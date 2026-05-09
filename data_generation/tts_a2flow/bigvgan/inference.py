# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import linear_and_mel_spectrogram, mel_spectrogram, MAX_WAV_VALUE, load_full_wav_to_torch, load_random_wav_segment_to_torch
import librosa
from train import load_generator
from models import apply_generator_forward
from tqdm import tqdm
import soundfile as sf
# import dac
# from audiotools import AudioSignal

torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_linear_and_mel_spectrogram(x, h):
    return linear_and_mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def get_mel_spectrogram(x, h):
    stereo = getattr(h, "stereo", False)
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, h, device):
    # if a.dac_path is not None:
    #     print(f"!!!!!!!!!!!!!!RUNNING DAC EVAL!!!!!!!!!!!!!!!!")
    #     print(f"{a.dac_path}")
    #     model = dac.DAC.load(a.dac_path).eval()
    #     model.to('cuda')
        
    # load model. can be stand-alone vocoder or latent autoencoder
    h.model_type = getattr(h, "model_type", "vocoder") # default to vocoder if not set
    generator = load_generator(h.model_type, h, device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    filelist = os.listdir(a.input_wavs_dir)
    
    stereo = getattr(h, "stereo", False)
    if a.forced_stereo:
        assert generator.stereo is False, "forced_stereo mode only supports mono model!"
        print("WARNING: turning on EXPERIMENTAL forced_stereo mode. The mono model will process left/right mel independently then will concatenate channels.")
        stereo = True
        a.output_dir += "_forced_stereo"
    if a.flip_phase:
        assert stereo is True
        print("WARNING: turning on EXPERIMENTAL flip_phase mode. The ground-truth right channel will be phase-flipped (-x)")
        a.output_dir += "_flip_phase"

    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(os.path.join(a.output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(a.output_dir, "generated"), exist_ok=True)

    print(f"Writing generated output files with file format: {a.output_dir}/real/(reference).wav and {a.output_dir}/generated/(reference).wav")
        
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # load the ground truth audio and resample if necessary
            # wav, sr = librosa.load(os.path.join(a.input_wavs_dir, filname), sr=h.sampling_rate, mono=True)
            audiopath = os.path.join(a.input_wavs_dir, filname)
            track = sf.SoundFile(audiopath)
            wav = load_full_wav_to_torch(track, audiopath, h.sampling_rate, stereo=stereo)
            wav = torch.FloatTensor(wav).to(device)
            
            # trim last elements for the wav length to be multiple of hop_size. this is requird for external evaluation
            if (wav.shape[-1] % h.hop_size) != 0:
                wav = wav[..., :-(wav.shape[-1] % h.hop_size)]
            
            if a.flip_phase:
                wav[1, :] = -wav[1, :]
            
            # normalize volume, same as training
            if getattr(h, "normalize_volume", True):
                wav = wav / (wav.abs().max() + 1e-5) * 0.95 # L-inf volume normalization as in public hifi-gan & bigvgan
            # compute linear and mel spectrogram from the ground truth audio
            x_linear, x_mel = get_linear_and_mel_spectrogram(wav.unsqueeze(0), h)
            # choose input representation
            if getattr(h, "use_wav_as_input", False):
                x = wav.clone()
            elif getattr(h, "use_linear_spec_as_input", False):
                x = x_linear
            else:
                x = x_mel

            # generator inference
            if a.forced_stereo: # forced stereo mode for mono models
                with torch.inference_mode():
                    x_left, x_right = torch.split(x, h.num_mels, dim=1)
                    return_dict_left = apply_generator_forward(h.model_type, generator, x_left)
                    return_dict_right = apply_generator_forward(h.model_type, generator, x_right)
                    y_g_hat_left = return_dict_left["decoder_out"][0] # [C, T]
                    y_g_hat_right = return_dict_right["decoder_out"][0] # [C, T]
                    y_g_hat = torch.cat([y_g_hat_left, y_g_hat_right], dim=0)
                    if return_dict_left["latent"] is not None:
                        latent_left = return_dict_left["latent"].squeeze() # [C, T]
                        latent_right = return_dict_right["latent"].squeeze() # [C, T]
                        latent = torch.cat([latent_left, latent_right], dim=0)
                        print(f"latent min {latent.min():.2f} max {latent.max():.2f} mean {latent.mean():.2f} std {latent.std():.2f}")
            else:
                with torch.inference_mode():
                    # apply model forward. encoder_out and latent are avilable only for autoencoder (for vocoder, both are None)
                    return_dict = apply_generator_forward(h.model_type, generator, x)
                    y_g_hat = return_dict["decoder_out"][0] # [C, T]
                if return_dict["latent"] is not None:
                    latent = return_dict["latent"].squeeze() # [C, T]
                    print(f"latent min {latent.min():.2f} max {latent.max():.2f} mean {latent.mean():.2f} std {latent.std():.2f}")
                    
            # # DAC inference
            # if a.dac_path is not None:
            #     wav = AudioSignal(wav, h.sampling_rate)
            #     # n_quantizers = model.n_codebooks
            #     with torch.inference_mode():
            #         wav = model.preprocess(wav.audio_data, wav.sample_rate)
            #         z, codes, latents, _, _ = model.encode(
            #             wav,
            #             # n_quantizers=n_quantizers
            #             )
            #         # Decode audio signal from z
            #         y_g_hat = model.decode(z).squeeze(1) # [1(B), T]
            #     wav = wav.squeeze(1)

            audio_gt = wav * MAX_WAV_VALUE
            audio_gt = audio_gt.permute(1, 0).cpu().numpy().astype('int16')
            
            audio = y_g_hat
            if audio.min() < -1. or audio.max() > 1.:
                print("WARNING: clamping output to [-1, 1]. this would have caused from models not using tanh output in the end")
                audio = torch.clamp(audio, -1, 1)
            audio = audio * MAX_WAV_VALUE
            audio = audio.permute(1, 0).cpu().numpy().astype('int16')

            output_file_gt = os.path.join(a.output_dir, "real", os.path.splitext(filname)[0] + '.wav')
            write(output_file_gt, h.sampling_rate, audio_gt)
            
            output_file = os.path.join(a.output_dir, "generated", os.path.splitext(filname)[0] + '.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--forced_stereo', action="store_true", help="(EXPERIMENTAL) try to generate stereo sound from mono models by processing left/right channels separately and concatenating them together.")
    parser.add_argument('--flip_phase', action="store_true", help="(EXPERIMENTAL) flip the phase in right channel to check model behavior.")
    
    # # TEMP
    # parser.add_argument('--dac_path', type=str, default=None)
    
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, h, device)


if __name__ == '__main__':
    main()

