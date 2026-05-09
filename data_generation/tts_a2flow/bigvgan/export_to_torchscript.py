import glob
import os
import argparse
import json
import torch
from env import AttrDict
from meldataset import load_full_wav_to_torch
from train import load_generator
from models import apply_generator_forward
from inference import load_checkpoint, get_linear_and_mel_spectrogram
import soundfile as sf

def export(a, h, device):
    # load model. can be stand-alone vocoder or latent autoencoder
    h.model_type = getattr(h, "model_type", "vocoder") # default to vocoder if not set
    generator = load_generator(h.model_type, h, device)
    
    # load checkpoint
    state_dict_g = load_checkpoint(a.checkpoint_file, device=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    # define audiopath for tracing
    audiopath = a.trace_audiopath
    
    # load the ground truth audio and resample if necessary
    track = sf.SoundFile(audiopath)
    stereo = getattr(h, "stereo", False)
    wav = load_full_wav_to_torch(track, audiopath, h.sampling_rate, stereo=stereo)
    wav = torch.FloatTensor(wav).to(device)
    
    # compute linear and mel spectrogram from the ground truth audio
    x_linear, x_mel = get_linear_and_mel_spectrogram(wav.unsqueeze(0), h)
    x = x_linear if getattr(h, "use_linear_spec_as_input", False) else x_mel # choose input representation

    with torch.no_grad():
        if h.model_type == "vocoder":
            # trace generator.forward() of vocoder
            audio = generator(x)
            traced_vocoder = torch.jit.trace(generator, x, strict=False)
            traced_audio = traced_vocoder(x)
            
            # both outputs should be (almost) the same 
            test_result = (audio - traced_audio).abs()
            diff = test_result.mean(dim=-1).item()
            # Assertions to ensure outputs match between original and traced models
            assert diff <= 1e-3, "audio output between original and traced model is higher than 1e-3"
            
            save_path_vocoder = os.path.join(os.path.split(a.checkpoint_file)[0], 'torchscript_vocoder_'+os.path.basename(a.checkpoint_file))
            torch.jit.save(traced_vocoder, save_path_vocoder)
            print(f"saved traced model to\n{save_path_vocoder}") 
                       
        else:
            return_dict_encoder = generator.encode(x) # x_mel is [B, C_mel, T_frame]
            latent = return_dict_encoder["latent"] # latent is [B, C_latent, T_frame]
            return_dict_decoder = generator.decode(latent)
            # trace generator.encode() and generator.decode()
            traced_encoder = torch.jit.trace_module(generator, {"encode": x}, strict=False)
            traced_decoder = torch.jit.trace_module(generator, {"decode": latent}, strict=False)
            
            # sanity check of encoder except for the latent (random variable)
            traced_return_dict_encoder = traced_encoder.encode(x)
            mu, traced_mu = return_dict_encoder["mu"], traced_return_dict_encoder["mu"]
            diff_mu = (mu - traced_mu).abs().mean().item()
            assert diff_mu <= 1e-3, "mu between original and traced encoder is higher than 1e-3"
            logvar, traced_logvar = return_dict_encoder["logvar"], traced_return_dict_encoder["logvar"]
            diff_logvar = (logvar - traced_logvar).abs().mean().item()
            assert diff_logvar <=1e-3, "logvar between original and traced encoder is higher than 1e-3"
            
            # sanity check of decoder using the same latent
            traced_return_dict_decoder = traced_decoder.decode(latent)
            # both outputs should be (almost) the same 
            test_result = (return_dict_decoder["decoder_out"] - traced_return_dict_decoder["decoder_out"]).abs()
            diff = test_result.mean(dim=-1).item()
            # Assertions to ensure outputs match between original and traced models
            assert diff <= 1e-3, "audio output between original and traced decoder is higher than 1e-3"
            
            save_path_encoder = os.path.join(os.path.split(a.checkpoint_file)[0], 'torchscript_encoder_'+os.path.basename(a.checkpoint_file))
            save_path_decoder = os.path.join(os.path.split(a.checkpoint_file)[0], 'torchscript_decoder_'+os.path.basename(a.checkpoint_file))
            torch.jit.save(traced_encoder, save_path_encoder)
            torch.jit.save(traced_decoder, save_path_decoder)
            print(f"saved traced model to\n{save_path_encoder}\n{save_path_decoder}")

def main():
    print('Initializing Export Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--trace_audiopath', type=str,
                        default="/lustre/fsw/portfolios/adlr/users/sanggill/projects/playground/assets/reference_audio/jensen_gtc24.wav")

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

    export(a, h, device)
    
if __name__ == "__main__":
    main()