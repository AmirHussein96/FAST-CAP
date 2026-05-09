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
from meldataset import linear_and_mel_spectrogram, mel_spectrogram, MAX_WAV_VALUE, load_full_wav_to_torch
import librosa
from train import load_generator
from models import apply_generator_forward
from tqdm import tqdm
import random
import soundfile as sf

h = None
device = None
torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_linear_and_mel_spectrogram(x):
    return linear_and_mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def get_mel_spectrogram(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, h):
    # load model. can be stand-alone vocoder or latent autoencoder
    h.model_type = getattr(h, "model_type", "vocoder") # default to vocoder if not set
    generator = load_generator(h.model_type, h, "cuda")

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    with open(a.input_filelist, 'r') as f:
        filelist = f.readlines()
    list_audiopath = []
    for d in filelist:
        audiopath = os.path.join(a.input_wav_root, d.split("|")[0])
        list_audiopath.append(audiopath)
    list_audiopath = list_audiopath[:a.num_sample]
    
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    print(f"Writing generated output files with file format: {a.output_dir}/(reference)_generated.wav")
    with torch.no_grad():
        for i, audiopath in enumerate(list_audiopath):
            # load the ground truth audio and resample if necessary
            track = sf.SoundFile(audiopath)
            wav = load_full_wav_to_torch(track, audiopath, h.sampling_rate, stereo=False)
            wav = torch.FloatTensor(wav).to(device)
            
            # get 10 sec chunk if it's longer
            segment_length_sec = wav.shape[0] / sr
            if segment_length_sec > 10:
                # start_idx = random.randint(0, wav.shape[0] - 10 * sr)
                start_idx = 0
                wav = wav[start_idx:start_idx+10*sr]
                
            wav = torch.FloatTensor(wav).to(device)
            # normalize volume, same as training
            if getattr(h, "normalize_volume", True):
                wav = wav / (wav.abs().max() + 1e-5) * 0.95 # L-inf volume normalization as in public hifi-gan & bigvgan
                
            # compute linear and mel spectrogram from the ground truth audio
            x_linear, x_mel = get_linear_and_mel_spectrogram(wav.unsqueeze(0))
            # choose input representation
            if getattr(h, "use_wav_as_input", False):
                x = wav.clone()
            elif getattr(h, "use_linear_spec_as_input", False):
                x = x_linear
            else:
                x = x_mel

            # generator inference
            with torch.inference_mode():
                # apply model forward. encoder_out and latent are avilable only for autoencoder (for vocoder, both are None)
                return_dict = apply_generator_forward(h.model_type, generator, x)
                y_g_hat = return_dict["decoder_out"][0] # [C, T]
            if return_dict["latent"] is not None:
                latent = return_dict["latent"].squeeze() # [C, T]
                print(f"latent min {latent.min():.2f} max {latent.max():.2f} mean {latent.mean():.2f} std {latent.std():.2f}")
            
            audio_gt = wav * MAX_WAV_VALUE
            audio_gt = audio_gt.permute(1, 0).cpu().numpy().astype('int16')
            
            audio = y_g_hat
            if audio.min() < -1. or audio.max() > 1.:
                print("WARNING: clamping output to [-1, 1]. this would have caused from models not using tanh output in the end")
                audio = torch.clamp(audio, -1, 1)
            audio = audio * MAX_WAV_VALUE
            audio = audio.permute(1, 0).cpu().numpy().astype('int16')

            output_file_gt = os.path.join(a.output_dir, os.path.basename(os.path.splitext(audiopath)[0]) + '_real.wav')
            write(output_file_gt, h.sampling_rate, audio_gt)
            
            output_file = os.path.join(a.output_dir, os.path.basename(os.path.splitext(audiopath)[0]) + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filelist')
    parser.add_argument('--input_wav_root')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--num_sample', type=int, default=30)
    parser.add_argument('--checkpoint_file', required=True)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, h)


if __name__ == '__main__':
    main()

