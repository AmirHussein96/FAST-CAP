import warnings
# Suppression for PyTorch and Hydra related warnings
warnings.filterwarnings('ignore', message="torch.nn.utils.weight_norm is deprecated")
warnings.filterwarnings('ignore', message="The version_base parameter is not specified.*")
warnings.filterwarnings('ignore', message="Usage of deprecated keyword in package header.*")
warnings.filterwarnings('ignore', message="'config' is validated against ConfigStore schema.*")
warnings.filterwarnings('ignore', message="The strict flag in the compose API is deprecated.*")
warnings.filterwarnings('ignore', message="This behavior is deprecated*")
warnings.filterwarnings('ignore', message="divide by zero encountered in log10", category=RuntimeWarning)
warnings.filterwarnings('ignore', message="Pytorch pre-release version.*", category=UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, module=r'.*utmos.*')
# To suppress all warnings from hydra.experimental.initialize
warnings.filterwarnings('ignore', message=".*is no longer experimental.*", category=UserWarning)
warnings.filterwarnings('ignore', message=".*The version_base parameter is not specified.*", category=UserWarning)
# To suppress warnings related to configuration validation and deprecation in Hydra and fairseq
warnings.filterwarnings('ignore', message=".*is validated against ConfigStore schema.*", category=UserWarning)
warnings.filterwarnings('ignore', message=".*changes to package header for more information.*", category=UserWarning)
# For any other unexpected warnings, consider suppressing by the module or a broader category
warnings.filterwarnings('ignore', module='hydra.*')
warnings.filterwarnings('ignore', module='fairseq.*')

import math
import json
from env import AttrDict
from time import time
from tqdm import tqdm
import argparse
import os, glob
from pathlib import Path
import re
import sys
import soundfile as sf
from meldataset import load_data, MelDataset, MAX_WAV_VALUE, load_full_wav_to_torch

from models import BigVGAN, LatentAutoEncoder
from train import load_generator
from models import apply_generator_forward
from inference import load_checkpoint, get_linear_and_mel_spectrogram
from xutils import save_checkpoint

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchaudio as ta
import auraloss
import librosa
from pesq import pesq, NoUtterancesError
from metrics.utmos import UTMOSScore
from metrics.periodicity import calculate_periodicity_metrics
from metrics.mcd_dtw import get_metrics_wavs_from_numpy
from scipy.io.wavfile import write
import dac
from audiotools import AudioSignal

def load_ema_checkpoint(cp_dir, prefix, ema_decay=0.999, device='cuda'):
    """
    Load the exponential moving average of generator checkpoints into the generator model,
    excluding 'g_last' and only considering the last 10 checkpoints.

    Args:
    - cp_dir (str): Directory containing the checkpoints.
    - prefix (str): Prefix of the checkpoint filenames.
    - ema_decay (float): Decay rate for the exponential moving average.
    - device (str): Device to map the loaded checkpoint.

    Returns:
    - A dictionary similar to what load_checkpoint returns, with keys including 'generator'
      containing the EMA state dict for the generator.
    - The numeric identifier of the first checkpoint used in the EMA computation.
    """
    all_files = glob.glob(os.path.join(cp_dir, prefix + '*'))
    # Filter files: keep only those ending with numbers, excluding 'g_last'
    cp_list = [f for f in all_files if f.split('_')[-1].isdigit()]

    if len(cp_list) == 0:
        raise FileNotFoundError(f"No numeric checkpoints found with prefix '{prefix}' in directory '{cp_dir}'.")

    # Sort and limit to the last 10 checkpoints
    cp_list = sorted(cp_list, key=lambda x: int(x.split('_')[-1]), reverse=True)[:10]

    ema_state_dict = None
    first_checkpoint_identifier = None
    ema_factor = 1.0

    for i, cp_path in enumerate(cp_list):
        checkpoint_identifier = cp_path.split('_')[-1]
        if first_checkpoint_identifier is None:
            first_checkpoint_identifier = checkpoint_identifier

        print(f"Loading checkpoint '{cp_path}' for EMA computation.")
        checkpoint = torch.load(cp_path, map_location=device)
        state_dict = checkpoint['generator']

        if ema_state_dict is None:
            ema_state_dict = state_dict
        else:
            # Update the EMA state dict
            for key in state_dict:
                if key in ema_state_dict:
                    ema_state_dict[key] = ema_factor * ema_state_dict[key] + (1.0 - ema_factor) * state_dict[key]
                else:
                    raise KeyError(f"Key '{key}' not found in the EMA state dictionary.")

        ema_factor *= ema_decay

    return {'generator': ema_state_dict}, first_checkpoint_identifier

def setup_visqol(mode: str):
    # setup visqol following example https://github.com/google/visqol
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2
    config = visqol_config_pb2.VisqolConfig()

    if mode == "audio":
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()

    api.Create(config)
    # use api like:
    # similarity_result = api.Measure(reference, degraded)
    # print(similarity_result.moslqo)
    # print(f"VISQOL mode set to {mode}")
    return api
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='/lustre/fsw/portfolios/adlr/users/sanggill/external_models/BigVGAN_checkpoints/bigvgan_24khz_100band/g_05000000')
    parser.add_argument('--dataset_config', default='datasets/libritts_24k.json')
    # parser.add_argument('--visqol_mode', default="speech", choices=["audio", "speech"], 
    #                     help="The mode for ViSQOL. Can be 'audio' (uses 48kHz) or 'speech' (uses 16kHz).")
    parser.add_argument('--skip_pesq', default=False, action='store_true', 
                        help="whether to skip pesq calc for non-speech data.")
    parser.add_argument('--ema_decay', type=float, default=None, help='Decay rate for the exponential moving average. If not set, loads the latest checkpoint.')
    parser.add_argument('--save_ema', default=False, action='store_true', help="save ema checkpoint as ema_(ema_decay)_g.pt in --checkpoint_path")
    
    # # TEMP
    # parser.add_argument('--dac_path', type=str, default=None)
    
    # load configs
    args = parser.parse_args()
    if args.ema_decay is not None and (not os.path.isdir(args.checkpoint_path)):
        raise ValueError("--checkpoint_path must be a directory when using --ema_decay.")
    
    # if args.dac_path is not None:
    #     print(f"!!!!!!!!!!!!!!RUNNING DAC EVAL!!!!!!!!!!!!!!!!")
    #     print(f"{args.dac_path}")
    #     model = dac.DAC.load(args.dac_path).eval()
    #     model.to('cuda')
    
    # Determine the correct path for config.json
    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.is_dir():
        # If the checkpoint_path is a directory, look for config.json inside this directory
        config_path = checkpoint_path / "config.json"
    else:
        # If the checkpoint_path is a file, look for config.json in the parent directory of the file
        config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at expected location: {config_path}")    
    with open(config_path) as f:
        config = f.read()
        
    # load dataset_config for evaluation
    dataset_config_path = args.dataset_config
    with open(dataset_config_path) as f:
        dataset_config = f.read()
    
    # load two jsons and merge into attrdict
    json_config = json.loads(config)
    json_dataset_config = json.loads(dataset_config)
    h = AttrDict({**json_config, **json_dataset_config})
    
    # load model. can be stand-alone vocoder or latent autoencoder
    h.model_type = getattr(h, "model_type", "vocoder") # fallback support
    generator = load_generator(h.model_type, h, "cuda")
    
    # Conditional logic for checkpoint loading
    if args.ema_decay is None:
        state_dict_g = load_checkpoint(args.checkpoint_path, device="cuda")
    else:
        state_dict_g, first_ckpt_id = load_ema_checkpoint(args.checkpoint_path, prefix="g_", ema_decay=args.ema_decay, device="cuda")
        if args.save_ema:
            ema_checkpoint_filename = f"ema_{args.ema_decay:.3f}_g_{first_ckpt_id}.pt"
            ema_checkpoint_path = os.path.join(args.checkpoint_path, ema_checkpoint_filename)
            save_checkpoint(ema_checkpoint_path, state_dict_g) # backward compatible to existing loading logic below
            
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    # prepare dataloader
    ignore_keys = ['training_files', 'validation_files']
    validation_filelist = load_data(
        h.data["validation_files"],
        pre_shuffle=False,
        **dict((k, v) for k, v in h.data.items() if k not in ignore_keys)
    )
    validset = MelDataset(
        validation_filelist, h, 
        split=False, shuffle=False, n_cache_reuse=0, device="cuda",
        fine_tuning=False, base_mels_path=None, is_seen=True, debug=False
    )
    validation_loader = DataLoader(
        validset, num_workers=1, shuffle=False,
        sampler=None, batch_size=1, pin_memory=True, drop_last=True
    )
    
    
    # modules for evaluation metrics
    # torchaudio resampler used by original bigvgan. uses hann window with bad aliasing,
    # but values get higher than librosa 0.10.1's better sox resampling so let's keep it like before during eval (shrug)
    resampler_16k = ta.transforms.Resample(h.sampling_rate, 16000).cuda() # for pesq, utmos, visqol-speech, torchcrepe
    resampler_22k = ta.transforms.Resample(h.sampling_rate, 22050).cuda() # for torchcrepe
    resampler_48k = ta.transforms.Resample(h.sampling_rate, 48000).cuda() # for visqol-audio
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")
    loss_sisdr = auraloss.time.SISDRLoss()
    utmos_model = UTMOSScore(device="cuda")
    # Setup ViSQOL API for both speech and audio
    visqol_api_speech = setup_visqol(mode="speech")
    visqol_api_audio = setup_visqol(mode="audio")
    # if args.visqol_mode == "audio" and not args.skip_pesq:
    #     print("WARNING: visqol_mode is audio. if you are using non-speech data for evaluation, you probably want to add --skip_pesq in CLI!")
        
    if "musdbhq" in args.dataset_config:
        print(f"INFOL: musdbhq eval. using 10s chunk starting at 1 minute of each audio files!")
    
    # metrics placeholder
    val_pesq_tot = 0.
    val_mrstft_tot = 0.
    val_utmos_gt_tot = 0.
    val_utmos_tot = 0.
    val_visqol_speech_tot = 0.
    val_visqol_audio_tot = 0.
    val_mcd_tot = 0.
    val_sisdr_tot = 0.
    val_periodicity_tot = 0.
    val_pitch_tot = 0.
    val_vuv_f1_tot = 0.

    # batch counters
    num_batches = 0  # Counter for number of batches processed
    num_batches_for_pesq = 0
    num_batches_for_mcd = 0
    num_batches_for_visqol_speech = 0
    num_batches_for_visqol_audio = 0
    num_batches_for_pitch_vuv_f1 = 0
    
    # loop over validation set and compute metrics
    for j, batch in tqdm(enumerate(validation_loader), total=len(validation_loader), desc="Evaluating"):
        x_linear, x_mel, y, audiopath, _ = batch["linear_spec"], batch["mel"], batch["audio"], batch["audiopath"], batch["mel_loss"]
        
        # (nasty) ad-hoc logic for musdbhq test: load 10 sec chunk starting from 1 minute
        # all samples from musdbhq test is at least 3m+ long
        # this can potentially have silence-only for non-mixture. just skip it: fine as long as the logic is consistent across models anyway
        if "musdbhq" in args.dataset_config:
            MUSDBHQ_START_SEC = 60
            MUSDBHQ_CHUNK_SEC = 10
            y = y[..., int(MUSDBHQ_START_SEC*h.sampling_rate):int((MUSDBHQ_START_SEC+MUSDBHQ_CHUNK_SEC)*h.sampling_rate)]
            if (y.shape[-1] % h.hop_size) != 0:
                y = y[..., :-(y.shape[-1] % h.hop_size)]
            if abs(y).max() < 1e-4:
                print(f"WARNING: skipping silent audio segment from {audiopath}")
                continue
            # # TEMP: save musdb samples for inference usage later
            # base_path, filename = os.path.split(audiopath[0])
            # artist, song = os.path.split(base_path)[-1].split(' - ')
            # tracks = os.path.splitext(filename)[0]
            # new_filename = f"{artist}_{song}_{tracks}.wav".replace(' ', '_')
            # write(os.path.join('tmp', new_filename), h.sampling_rate, (y*MAX_WAV_VALUE).squeeze().cpu().numpy().astype('int16'))
            # continue
            x_linear, x_mel = get_linear_and_mel_spectrogram(y, h)    
        
        # choose input representation
        if getattr(h, "use_wav_as_input", False):
            x = y.clone()
        elif getattr(h, "use_linear_spec_as_input", False):
            x = x_linear
        else:
            x = x_mel
        x = x.cuda()
        y = y.squeeze(1).cuda()  #[1(B), T]
        
        with torch.inference_mode():
            # Apply model forward. encoder_out and latent are available only for autoencoder (for vocoder, both are None)
            return_dict = apply_generator_forward(h.model_type, generator, x)
            y_g_hat = return_dict["decoder_out"].squeeze(1)  # [1(B), T]
    
        # # DAC inference
        # if args.dac_path is not None:
        #     y = AudioSignal(y, h.sampling_rate)
        #     # n_quantizers = model.n_codebooks
        #     with torch.inference_mode():
        #         y = model.preprocess(y.audio_data, y.sample_rate)
        #         z, codes, latents, _, _ = model.encode(
        #             y,
        #             # n_quantizers=n_quantizers
        #             )
        #         # Decode audio signal from z
        #         y_g_hat = model.decode(z).squeeze(1) # [1(B), T]
        #     y = y.squeeze(1)
            
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
        
        # PESQ calculation. need to use int 16k. receives [T]
        if not args.skip_pesq:
            try:
                val_pesq_tot += pesq(16000, y_int_16k.squeeze().cpu().numpy(), y_g_hat_int_16k.squeeze().cpu().numpy(), 'wb')
                num_batches_for_pesq += 1
            except NoUtterancesError as e: # only evaluate PESQ if it's speech signal (nonspeech PESQ will error out)
                print(e)
                pass
        
        # MCD calculation (using recent implementation), receives [T]. disable use_dtw assuming audio is aligned in time
        try:
            mcd, penalty, _ = get_metrics_wavs_from_numpy(y.squeeze().cpu().numpy(), y_g_hat.squeeze().cpu().numpy(), sr=h.sampling_rate, use_dtw=False)
            val_mcd_tot += mcd
            num_batches_for_mcd += 1
        except ValueError as e: # corner cases in non-speech data
            print(e)
            pass
        
        # MRSTFT calculation, receives [1(B), 1, T]
        val_mrstft_tot += loss_mrstft(y_g_hat.unsqueeze(1), y.unsqueeze(1)).item()
        
        # SI-SDR calculation, receives [1(B), 1, T]. - to be interpreted as hihger-the-better metric in DB (not as minimization as target loss)
        sisdr = -loss_sisdr(y_g_hat.unsqueeze(1), y.unsqueeze(1)).item()
        val_sisdr_tot += sisdr
        
        # UTMOSScore calculation. model needs to use 16k. receives [1(B), T]
        val_utmos_gt_tot += utmos_model.score(y_16k).item()
        val_utmos_tot += utmos_model.score(y_g_hat_16k).item()
        
        # VISQOL api receives [T] with np.float64 https://github.com/google/visqol/issues/67
        y_visqol_speech = y_16k.squeeze().cpu().numpy().astype(np.float64)
        y_g_hat_visqol_speech = y_g_hat_16k.squeeze().cpu().numpy().astype(np.float64)

        y_visqol_audio = y_48k.squeeze().cpu().numpy().astype(np.float64)
        y_g_hat_visqol_audio = y_g_hat_48k.squeeze().cpu().numpy().astype(np.float64)
        
        try:
            similarity_result_speech = visqol_api_speech.Measure(y_visqol_speech, y_g_hat_visqol_speech)
            val_visqol_speech_tot += similarity_result_speech.moslqo
            num_batches_for_visqol_speech += 1
        except Exception as e: # no clue
            print(f"Error during visqol_spech: {e}")
            pass
            
        try:
            similarity_result_audio = visqol_api_audio.Measure(y_visqol_audio, y_g_hat_visqol_audio)
            val_visqol_audio_tot += similarity_result_audio.moslqo
            num_batches_for_visqol_audio += 1
        except Exception as e: # no clue
            print(f"Error during visqol_audio: {e}")
            pass
        
        # Periodicity & V/UV metrics calculation based on torchcrepe. need to use 22k. receives [1(B), T]
        periodicity_loss, pitch_loss, f1 = calculate_periodicity_metrics(y_22k, y_g_hat_22k)
        val_periodicity_tot += periodicity_loss
        if not math.isnan(pitch_loss) and not math.isnan(f1):
            val_pitch_tot += pitch_loss
            val_vuv_f1_tot += f1
            num_batches_for_pitch_vuv_f1 += 1
        else:
            print(f"nan detected for pitch and vuv f1")
            pass
        
        num_batches += 1
    
    # Calculate averages for each metric
    if not args.skip_pesq:
        avg_pesq = val_pesq_tot / num_batches_for_pesq
    else:
        avg_pesq = 0.
    avg_mcd = val_mcd_tot / num_batches_for_mcd
    avg_mrstft = val_mrstft_tot / num_batches
    avg_sisdr = val_sisdr_tot / num_batches
    avg_utmos_gt = val_utmos_gt_tot / num_batches
    avg_utmos = val_utmos_tot / num_batches
    avg_visqol_speech = val_visqol_speech_tot / num_batches_for_visqol_speech
    avg_visqol_audio = val_visqol_audio_tot / num_batches_for_visqol_audio
    avg_periodicity = val_periodicity_tot / num_batches
    avg_pitch = val_pitch_tot / num_batches_for_pitch_vuv_f1
    avg_vuv_f1 = val_vuv_f1_tot / num_batches_for_pitch_vuv_f1
    
    metrics = ["PESQ↑", "UTMOS(GT)↑", "UTMOS↑", "VISQOL-S↑", "VISQOL-A↑", "MRSTFT↓", "MCD↓", "SI-SDR↑", "Periodicity↓", "Pitch↓", "V/UV F1↑"]
    values = [avg_pesq, avg_utmos_gt, avg_utmos, avg_visqol_speech, avg_visqol_audio, avg_mrstft, avg_mcd, avg_sisdr, avg_periodicity, avg_pitch, avg_vuv_f1]
    
    # Generate the formatted string
    formatted_metrics = ' '.join([f"{metric:<12}" for metric in metrics])
    formatted_values = ' '.join([f"{value:<12.4f}" for value in values])

    print(f"####################################")
    print(f"Evaluation of checkpoint: {args.checkpoint_path}")
    print(f"using validation_files from: {args.dataset_config}")
    if args.ema_decay is not None:
        print(f"EMA decay used: {args.ema_decay}")
    print(f"####################################")
    formatted_table = f"{formatted_metrics}\n{formatted_values}"
    print(formatted_table)