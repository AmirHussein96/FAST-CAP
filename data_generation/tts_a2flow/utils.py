import ast
import argparse
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.io.wavfile import read
import sys
from model import A2Flow, DP
from textlesslib.textless.data.speech_encoder import SpeechEncoder

                   
def update_params(config, params):
    for param in params:
        print(param)
        k, v = param.split("=")
        try:
            v = ast.literal_eval(v)
        except:
            pass

        k_split = k.split('.')
        if len(k_split) > 1:
            parent_k = k_split[0]
            cur_param = ['.'.join(k_split[1:])+"="+str(v)]
            update_params(config[parent_k], cur_param)
        elif k in config and len(k_split) == 1:
            print(f"overriding {k} with {v}")
            config[k] = v
        elif len(k_split) == 1:
            print(f"new params {k} with {v}")
            config[k] = v
        else:
            print("{}, {} params not updated".format(k, v))

            
def load_dp(checkpoint_dir, hps, n_language=1):
    n_vocab=hps.data.nvocab
    if getattr(hps.data, "add_bos_eos_to_text", False):
        n_vocab = n_vocab + 2
    model = DP(
            n_vocab=n_vocab,
            n_feats=128 if getattr(hps.train, "latent", False) else hps.data.n_feats,
            n_language=n_language,
            **hps.model).cuda()
    
    model_path = latest_checkpoint_path(checkpoint_dir, regex="dp_*.pt")
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict['model'], strict=False)
    return model
            

def load_a2flow(checkpoint_dir, hps, n_language=1, return_iteration=False, num=-1):
    n_vocab=hps.data.nvocab
    if getattr(hps.data, "add_bos_eos_to_text", False):
        n_vocab = n_vocab + 2
        
    unit_encoder = SpeechEncoder.by_name(
        dense_model_name=getattr(hps.model, "dense_model_name", "hubert-base-ls960"),
        quantizer_model_name=getattr(hps.model, "quantizer_name", "kmeans"),
        vocab_size=getattr(hps.model, "n_unit", 200),
        deduplicate=True
    ).eval().cuda()
    
    model = A2Flow(
            n_vocab=n_vocab,
            n_feats=hps.data.n_feats,
            n_language=n_language,
            unit_encoder=unit_encoder,
            **hps.model)
    if num == -1:
        model_path = latest_checkpoint_path(checkpoint_dir, regex="fm_*.pt")
    else:
        model_path = os.path.join(checkpoint_dir, f"fm_{num}.pt")
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict['model'], strict=False)
    if return_iteration:
        iteration = int(model_path.split("_")[-1][:-3])
        return model, iteration
    else:
        return model
    

def load_vocoder(config_path, checkpoint_path, device='cpu'):
    sys.path.append('./bigvgan/')
    from env import AttrDict
    from models import BigVGAN as Generator
    
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    vocoder = Generator(h)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['generator']
    vocoder.load_state_dict(state_dict)
    vocoder.remove_weight_norm()
    return vocoder


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip() for line in f
                              if line.strip().split(split_char)[-1] != '']
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="fm_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


def load_checkpoint(rank, logdir, model, num=None, optimizer=None, name="fm", module=None, scheduler=None):
    new_model_dict = model.state_dict()
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex=f"{name}_*.pt")
    else:
        model_path = os.path.join(logdir, f"{name}_{num}.pt")
    if rank == 0:
        print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    new_model_dict.update(model_dict['model'])
    model.load_state_dict(new_model_dict, strict=False)
    if optimizer is not None and 'optimizer' in model_dict.keys(): # should be removed in final version of code
        optimizer.load_state_dict(model_dict['optimizer'])
    if scheduler is not None and 'scheduler' in model_dict.keys():
        scheduler.load_state_dict(model_dict['scheduler'])
    return model, int(model_path.split('_')[-1].split('.')[0]), optimizer, scheduler


def get_hparams(train_dp=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/baseline.json")
    parser.add_argument('--address', type=int, default=50000)
    parser.add_argument('--params', nargs='+', default=[])

    args = parser.parse_args()

    # load and save config from predefined path (args.config)
    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, args.params)
    output_dir = config['train']['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    hparams = HParams(**config)

    if train_dp:
        config_save_path = os.path.join(output_dir, "config_dp.json")
    else:
        config_save_path = os.path.join(output_dir, "config.json")
    with open(config_save_path, "w") as fp:
        json.dump(config, fp)

    hparams.address = args.address
    return hparams


def set_hparams(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, key, default):
        return self[key] if key in self else default
