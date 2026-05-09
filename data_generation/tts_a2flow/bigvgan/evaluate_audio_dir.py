import math
from env import AttrDict
from time import time
from tqdm import tqdm
import argparse
import os, glob
from pathlib import Path
import soundfile as sf
from meldataset import MAX_WAV_VALUE, load_full_wav_to_torch
import json

import torch
import numpy as np
import torchaudio as ta
import auraloss
import librosa
from pesq import pesq, NoUtterancesError
from metrics.utmos import UTMOSScore
from metrics.periodicity import calculate_periodicity_metrics
from metrics.mcd_dtw import get_metrics_wavs_from_numpy, calculate_mcd_v1

from evaluate import setup_visqol

import soundfile as sf

def check_sampling_rates(real_dir, generated_dir):
    sampling_rates = set()
    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.wav')])
    generated_files = sorted([f for f in os.listdir(generated_dir) if f.endswith('.wav')])

    for file in real_files + generated_files:
        file_path = os.path.join(real_dir if file in real_files else generated_dir, file)
        track = sf.SoundFile(file_path)
        sr = track.samplerate
        sampling_rates.add(sr)

    if len(sampling_rates) > 1:
        raise ValueError("Different sample rates detected in audio files. Check that all waveforms (real/generated) are in the same sample rate!")
    return sampling_rates.pop()

def main(args):
    if not args.skip_mcd_v1:
        print(f"WARNING: --skip_mcd_v1 is not set. mcd_v1 is very slow and prone to lots of errors. ARE YOU ABSOLUTELY CERTAIN?")
        
    print(f"Evaluating audio from audio_dir: {args.audio_dir}")
    real_dir = os.path.join(args.audio_dir, 'real')
    generated_dir = os.path.join(args.audio_dir, 'generated')

    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.wav')])
    generated_files = sorted([f for f in os.listdir(generated_dir) if f.endswith('.wav')])

    assert len(real_files) == len(generated_files), "Mismatch in number of files between real and generated directories."
    assert real_files == generated_files, "Mismatch in file names between real and generated directories."
    
    # Determine the sample rate from audio files
    sampling_rate = check_sampling_rates(real_dir, generated_dir)
    print(f"Detected sample rate: {sampling_rate}")
    
    # modules for evaluation metrics
    # torchaudio resampler used by original bigvgan. uses hann window with bad aliasing,
    # but values get higher than librosa 0.10.1's better sox resampling so let's keep it like before during eval (shrug)
    resampler_16k = ta.transforms.Resample(sampling_rate, 16000).cuda() # for pesq, utmos, visqol-speech, torchcrepe
    resampler_22k = ta.transforms.Resample(sampling_rate, 22050).cuda() # for torchcrepe and mcd_v1
    resampler_48k = ta.transforms.Resample(sampling_rate, 48000).cuda() # for visqol-audio
    
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")
    loss_sisdr = auraloss.time.SISDRLoss()
    utmos_model = UTMOSScore(device="cuda")
    # Setup ViSQOL API for both speech and audio
    visqol_api_speech = setup_visqol(mode="speech")
    visqol_api_audio = setup_visqol(mode="audio")

    # metrics placeholder
    val_pesq_tot = 0.
    val_mrstft_tot = 0.
    val_utmos_gt_tot = 0.
    val_utmos_tot = 0.
    val_visqol_speech_tot = 0.
    val_visqol_audio_tot = 0.
    val_mcd_v1_tot = 0.
    val_mcd_tot = 0.
    val_sisdr_tot = 0.
    val_periodicity_tot = 0.
    val_pitch_tot = 0.
    val_vuv_f1_tot = 0.

    # batch counters
    num_batches = 0  # Counter for number of batches processed
    num_batches_for_pesq = 0
    num_batches_for_mcd_v1 = 0
    num_batches_for_mcd = 0
    num_batches_for_visqol_speech = 0
    num_batches_for_visqol_audio = 0
    num_batches_for_pitch_vuv_f1 = 0
    
    for file_name in tqdm(real_files, total=len(real_files)):
        real_file_path = os.path.join(real_dir, file_name)
        generated_file_path = os.path.join(generated_dir, file_name)

        # load real and generated audio as y and y_g_hat
        real_track = sf.SoundFile(real_file_path)
        y = load_full_wav_to_torch(real_track, real_file_path, sampling_rate, stereo=False).cuda() # [1, T]
                    
        generated_track = sf.SoundFile(generated_file_path)
        y_g_hat = load_full_wav_to_torch(generated_track, generated_file_path, sampling_rate, stereo=False).cuda() # [1, T]
        
        # sanity check that y_g_hat is the actual audio from the model
        assert not torch.allclose(y, y_g_hat), "y and y_g_hat is unreasonably similar. is y_g_hat correct?"
        
        # normalize volume post-load if set. This can potentially alter the metrics
        if args.normalize_volume:
            y = y / (y.abs().max() + 1e-5) * 0.95
            y_g_hat = y_g_hat / (y_g_hat.abs().max() + 1e-5) * 0.95
            
        # resample to 16000 for pesq / utmos, shape [1(B), T]
        y_16k = resampler_16k(y)
        y_g_hat_16k = resampler_16k(y_g_hat)
        y_int_16k = (y_16k * MAX_WAV_VALUE).short()
        y_g_hat_int_16k = (y_g_hat_16k * MAX_WAV_VALUE).short()
        
        # resample to 22050 for torchcrepe
        y_22k = resampler_22k(y)
        y_g_hat_22k = resampler_22k(y_g_hat)
        
        # resample to 48000 for visqol-audio
        y_48k = resampler_48k(y)
        y_g_hat_48k = resampler_48k(y_g_hat)
        
        # PESQ calculation. need to use int 16k. receives [T] in int16 ndarray
        if not args.skip_pesq:
            try:
                val_pesq_tot += pesq(16000, y_int_16k.squeeze().cpu().numpy(), y_g_hat_int_16k.squeeze().cpu().numpy(), 'wb')
                num_batches_for_pesq += 1
            except NoUtterancesError as e: # only evaluate PESQ if it's speech signal (nonspeech PESQ will error out)
                print(f"pesq NoUtterancesError during filename {file_name}: {e}")
                pass
            
        # MCD v1 calculation (used by original bigvgan & PwC leaderboard). Receives [T] in double ndarray. uses dtw
        if not args.skip_mcd_v1:
            try:
                mcd = calculate_mcd_v1((y_22k * MAX_WAV_VALUE).squeeze().double().cpu().numpy(), (y_g_hat_22k * MAX_WAV_VALUE).squeeze().double().cpu().numpy(), sr=22050)
                val_mcd_v1_tot += mcd
                num_batches_for_mcd_v1 += 1
            except Exception as e: # corner cases in non-speech data
                print(f"mcd_v1 error during filename {file_name}: {e}")
                pass
            
        # MCD v2 calculation (using recent implementation), receives [T] in ndarray. disable use_dtw assuming audio is aligned in time
        try:
            mcd, penalty, _ = get_metrics_wavs_from_numpy(y.squeeze().cpu().numpy(), y_g_hat.squeeze().cpu().numpy(), sr=sampling_rate, use_dtw=False)
            val_mcd_tot += mcd
            num_batches_for_mcd += 1
        except Exception as e: # corner cases in non-speech data
            print(f"mcd_v2 error during filename {file_name}: {e}")
            pass
            
        # MRSTFT calculation, receives [1(B), 1, T] in torch.tensor
        val_mrstft_tot += loss_mrstft(y_g_hat.unsqueeze(1), y.unsqueeze(1)).item()
        
        # SI-SDR calculation, receives [1(B), 1, T] in torch.tensor. - to be interpreted as hihger-the-better metric in DB (not as minimization as target loss)
        sisdr = -loss_sisdr(y_g_hat.unsqueeze(1), y.unsqueeze(1)).item()
        val_sisdr_tot += sisdr
        
        # UTMOSScore calculation. model needs to use 16k. receives [1(B), T] in torch.tensor
        val_utmos_gt_tot += utmos_model.score(y_16k).item()
        val_utmos_tot += utmos_model.score(y_g_hat_16k).item()
        
        # VISQOL api receives [T] with np.float64 https://github.com/google/visqol/issues/67
        y_visqol_speech = y_16k.squeeze().cpu().numpy().astype(np.float64)
        y_g_hat_visqol_speech = y_g_hat_16k.squeeze().cpu().numpy().astype(np.float64)

        y_visqol_audio = y_48k.squeeze().cpu().numpy().astype(np.float64)
        y_g_hat_visqol_audio = y_g_hat_48k.squeeze().cpu().numpy().astype(np.float64)
        
        # visqol speech mode.
        try:
            similarity_result_speech = visqol_api_speech.Measure(y_visqol_speech, y_g_hat_visqol_speech)
            val_visqol_speech_tot += similarity_result_speech.moslqo
            num_batches_for_visqol_speech += 1
        except Exception as e: # no clue
            print(f"visqol-s error during filename {file_name}: {e}")
            pass
        
        # visqol audio mode
        try:
            similarity_result_audio = visqol_api_audio.Measure(y_visqol_audio, y_g_hat_visqol_audio)
            val_visqol_audio_tot += similarity_result_audio.moslqo
            num_batches_for_visqol_audio += 1
        except Exception as e: # no clue
            print(f"visqol-a error during filename {file_name}: {e}")
            pass
        
        # Periodicity & V/UV metrics calculation based on torchcrepe. need to use 22k. receives [1(B), T] in torch.tensor
        periodicity_loss, pitch_loss, f1 = calculate_periodicity_metrics(y_22k, y_g_hat_22k)
        val_periodicity_tot += periodicity_loss
        if not math.isnan(pitch_loss) and not math.isnan(f1):
            val_pitch_tot += pitch_loss
            val_vuv_f1_tot += f1
            num_batches_for_pitch_vuv_f1 += 1
        else:
            print(f"nan detected for pitch and vuv f1 during filename {file_name}")
            pass
        
        num_batches += 1
    
    # Calculate averages for each metric
    avg_pesq = val_pesq_tot / num_batches_for_pesq if not args.skip_pesq else 0.
    avg_mcd_v1 = val_mcd_v1_tot / num_batches_for_mcd_v1 if not args.skip_mcd_v1 and num_batches_for_mcd_v1 != 0 else 0.
    avg_mcd = val_mcd_tot / num_batches_for_mcd if num_batches_for_mcd != 0 else 0.
    avg_mrstft = val_mrstft_tot / num_batches
    avg_sisdr = val_sisdr_tot / num_batches
    avg_utmos_gt = val_utmos_gt_tot / num_batches
    avg_utmos = val_utmos_tot / num_batches
    avg_visqol_speech = val_visqol_speech_tot / num_batches_for_visqol_speech if num_batches_for_visqol_speech != 0 else 0.
    avg_visqol_audio = val_visqol_audio_tot / num_batches_for_visqol_audio if num_batches_for_visqol_audio != 0 else 0.
    avg_periodicity = val_periodicity_tot / num_batches 
    avg_pitch = val_pitch_tot / num_batches_for_pitch_vuv_f1 if num_batches_for_pitch_vuv_f1 != 0 else 0.
    avg_vuv_f1 = val_vuv_f1_tot / num_batches_for_pitch_vuv_f1 if num_batches_for_pitch_vuv_f1 != 0 else 0.
    
    metrics = ["PESQ↑", "UTMOS(GT)↑", "UTMOS↑", "VISQOL-S↑", "VISQOL-A↑", "MRSTFT↓", "MCD-v1↓", "MCD-v2↓", "SI-SDR↑", "Periodicity↓", "Pitch↓", "V/UV F1↑"]
    values = [avg_pesq, avg_utmos_gt, avg_utmos, avg_visqol_speech, avg_visqol_audio, avg_mrstft, avg_mcd_v1, avg_mcd, avg_sisdr, avg_periodicity, avg_pitch, avg_vuv_f1]
    
    # Generate the formatted string
    formatted_metrics = ''.join([f"{metric:<12}" for metric in metrics])
    formatted_values = ''.join([f"{value:<12.4f}" for value in values])

    print(f"####################################")
    print(f"Evaluation of audio_dir: {args.audio_dir}")
    print(f"####################################")
    formatted_table = f"{formatted_metrics}\n{formatted_values}"
    print(formatted_table)
    
    metrics_dict = {metric[:-1]:value for metric, value in zip(metrics, values)} # trim arrow for metric key

    with open(os.path.join(args.audio_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"evaluation results saved at {os.path.join(args.audio_dir, 'evaluation_results.json')}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default=None,
                        help="a parent path to real/generated audio pairs are saved. Assumes the dir contains (dir)/real/file.wav and (dir)/generated/file.wav. Useful for evaluating external models")
    parser.add_argument('--skip_pesq', default=False, action='store_true', 
                        help="whether to skip pesq calc for non-speech data.")
    parser.add_argument('--skip_mcd_v1', default=False, action='store_true',
                        help="whether to skip mcd_v1 (w/ DTW) used in BigVGAN-v1 to patch PapersWithCode benchmark. mcd_v1 is very slow and prone to lots of erros!")
    parser.add_argument('--normalize_volume', default=False, action='store_true',
                        help="normalize volume using audio = audio / (audio.abs().max() + 1e-5) * 0.95. this matches evaluate.py's loading logic for volume normed models.")
       
    args = parser.parse_args()
    
    main(args)