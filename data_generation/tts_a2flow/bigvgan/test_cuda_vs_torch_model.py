import math
import torch
import json
from env import AttrDict
from models import BigVGAN
from time import time
from tqdm import tqdm
import os
from meldataset import mel_spectrogram, MAX_WAV_VALUE
import librosa
from scipy.io.wavfile import write

# for easier debugging
torch.set_printoptions(
    linewidth=200,
    threshold=10_000
)

def get_mel(x, h):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

if __name__ == "__main__":
    config_path = "configs/bigvgan_24khz_100band.json"
    ckpt_path = "/lustre/fsw/portfolios/adlr/users/sanggill/external_models/BigVGAN_checkpoints/bigvgan_24khz_100band/g_05000000"
    reference_audio_path = "/lustre/fsw/portfolios/adlr/users/sanggill/projects/playground/assets/reference_audio/celeb1.wav"
    
    with open(config_path) as f:
        config = f.read()
    json_config = json.loads(config)
    h = AttrDict({**json_config})
    
    h.use_cuda_kernel = False
    generator_original = BigVGAN(h).to("cuda")
    h.use_cuda_kernel = True
    generator_cuda_kernel = BigVGAN(h).to("cuda")
    
    state_dict_g = load_checkpoint(ckpt_path, "cuda")
    generator_original.load_state_dict(state_dict_g['generator'])
    generator_cuda_kernel.load_state_dict(state_dict_g['generator'])
    
    generator_original.eval()
    generator_original.remove_weight_norm()
    generator_cuda_kernel.eval()
    generator_cuda_kernel.remove_weight_norm()
        
    toc_total_original = 0.
    toc_total_cuda_kernel = 0.
    audio_length_total = 0.
    diff = 0.
    
    num_sample = 10
    num_mel_frame = 128
    for i in tqdm(range(num_sample)):
        # random mel: use large num_mel_frame to test peak gpu util performance
        data = torch.rand((1, h.num_mels, num_mel_frame), device='cuda')
        # original inference
        torch.cuda.synchronize()
        tic = time()
        with torch.inference_mode():
            audio_original = generator_original(data)
            torch.cuda.synchronize()
            toc = time() - tic
        toc_total_original += toc
        # cuda kernel inference
        torch.cuda.synchronize()
        tic = time()
        with torch.inference_mode():
            audio_cuda_kernel = generator_cuda_kernel(data)
            torch.cuda.synchronize()
            toc = time() - tic
        toc_total_cuda_kernel += toc
        audio_length_total += audio_cuda_kernel.shape[-1]

        # both outputs should be (almost) the same 
        test_result = (audio_original - audio_cuda_kernel).abs()
        diff += test_result.mean(dim=-1).item()

    diff /= num_sample
    if diff <= 1e-3:
        print(
            f"\n[Success] test CUDA fused vs. plain torch BigVGAN inference"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={audio_cuda_kernel[-1][-1][-30:].tolist()}"
            f"\n > torch_values={audio_original[-1][-1][-30:].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test CUDA fused vs. plain torch BigVGAN inference"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={audio_cuda_kernel[-1][-1][-30:].tolist()}, "
            f"\n > torch_values={audio_original[-1][-1][-30:].tolist()}"
        )
        
    audio_second = audio_length_total / h.sampling_rate      
    khz_original = audio_length_total / toc_total_original / 1000
    khz_cuda_kernel = audio_length_total / toc_total_cuda_kernel / 1000

    print('Original BigVGAN: took {:.2f} seconds to generate {:.2f} seconds of audio, {:.1f}kHz, {:.1f} faster than realtime'.format(toc_total_original, audio_second, khz_original, audio_second / toc_total_original))
    print('CUDA kernel BigVGAN: took {:.2f} seconds to generate {:.2f} seconds of audio, {:.1f}kHz, {:.1f} faster than realtime'.format(toc_total_cuda_kernel, audio_second, khz_cuda_kernel, audio_second / toc_total_cuda_kernel))
    print('speedup of CUDA kernel: {}'.format(khz_cuda_kernel/khz_original))
    
    # load the ground truth audio and resample if necessary
    wav, sr = librosa.load(reference_audio_path, sr=h.sampling_rate, mono=True)
    wav = torch.tensor(wav).to("cuda")
    # compute mel spectrogram from the ground truth audio
    x = get_mel(wav.unsqueeze(0), h)

    with torch.inference_mode():
        y_g_hat_original = generator_original(x)
        y_g_hat_cuda_kernel = generator_cuda_kernel(x)
    
    audio_original = y_g_hat_original.squeeze()
    audio_original = audio_original * MAX_WAV_VALUE
    audio_original = audio_original.cpu().numpy().astype('int16')
    
    audio_cuda_kernel = y_g_hat_cuda_kernel.squeeze()
    audio_cuda_kernel = audio_cuda_kernel * MAX_WAV_VALUE
    audio_cuda_kernel = audio_cuda_kernel.cpu().numpy().astype('int16')

    os.makedirs('tmp', exist_ok=True)
    output_file_original = os.path.join('tmp', 'audio_generated_original.wav')
    output_file_cuda_kernel = os.path.join('tmp', 'audio_generated_cuda_kernel.wav')
    write(output_file_original, h.sampling_rate, audio_original)
    write(output_file_cuda_kernel, h.sampling_rate, audio_cuda_kernel)
    print("Example generated audios of original vs. fused CUDA kernel written to tmp!")
    print("Done")