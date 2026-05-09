# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torchaudio.transforms import Spectrogram

import activations
from xutils import init_weights, get_padding
from alias_free_torch.act import Activation1d as TorchActivation1d

from modules import AMPBlock1, AMPBlock2

# added dependency from DAC
# from audiotools import AudioSignal
# from audiotools import STFTParams
from einops import rearrange
import typing
from typing import List, Optional, Tuple

import auraloss

def kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor
):
    """
    Calculate the Kullback-Leibler divergence loss for VAE.

    This function computes the KL divergence between the learned
    latent variable distribution Q(z|X) and its prior distribution P(z),
    which is assumed to be a standard Gaussian distribution N(0, I).
    
    The KL divergence loss acts as a regularization term, encouraging
    the learned distribution Q(z|X) to be close to the standard Gaussian
    distribution. This loss is a key component of the VAE loss function,
    alongside the reconstruction loss.

    Parameters:
    - mu (torch.Tensor): The mean of the latent variable distribution Q(z|X),
                         shape [batch_size, ...].
    - logvar (torch.Tensor): The logarithm of the variance of the latent
                             variable distribution Q(z|X),
                             shape [batch_size, ...].

    Returns:
    - torch.Tensor: Scalar tensor representing the KL divergence loss, summed
                    over all dimensions.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def feature_loss(
    fmap_r: List[List[torch.Tensor]],
    fmap_g: List[List[torch.Tensor]]
):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2 # this equates to lambda=2.0, prob not necessary to add it into hyperparams at this point


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
    loss_type: str = "l2" # default LSGAN as used by HiFi-GAN/BigVGAN
):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        if loss_type == "l2":
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
        elif loss_type == "hinge": # used by recent models like EnCodec, DAC, Vocos...
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
        else:
            raise NotImplementedError(f"unknown discriminator_loss loss_type {loss_type}")
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor],
    loss_type: str = "l2" # default LSGAN as used by HiFi-GAN/BigVGAN
):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        if loss_type == "l2":
            l = torch.mean((1-dg)**2)
        elif loss_type == "hinge": # used by recent models like EnCodec, DAC, Vocos...
            l = torch.mean(torch.clamp(1 - dg, min=0))
        else:
            raise NotImplementedError(f"unknown generator_loss loss_type {loss_type}")
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


# from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py
# and also changing default params to final setup in DAC
class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part, as in DAC)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0 (NOT 2.0 in original, as in DAC)
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = None,
        perceptual_weighting: bool = False
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow
        
        # new impl: perceptual_weighting impl from auraloss
        # uses A-weighting with FIRFilter and optimized defaults for 44khz. Will it help?
        self.perceptual_weighting = perceptual_weighting
        if self.perceptual_weighting:
            print("WARNING: turning on perceptual_weighting of DAC's MelSpectrogramLoss. Note that this is NOT the original impl of DAC!")
            if self.sampling_rate is None:
                raise ValueError(
                    f"`sample_rate` must be supplied when `perceptual_weighting = True`."
                )
            self.prefilter = auraloss.freq.FIRFilter(filter_type="aw", fs=self.sampling_rate)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        bs, chs, seq_len = x.size()
        if self.perceptual_weighting:  # apply optional A-weighting via FIR filter
            # since FIRFilter only support mono audio we will move channels to batch dim
            x = x.view(bs * chs, 1, -1)
            y = y.view(bs * chs, 1, -1)

            # now apply the filter to both
            self.prefilter.to(x.device)
            x, y = self.prefilter(x, y)

            # now move the channels back
            x = x.view(bs, chs, -1)
            y = y.view(bs, chs, -1)
            
        # wrap signals to AudioSignal
        x = AudioSignal(x, self.sampling_rate)
        y = AudioSignal(y, self.sampling_rate)
        
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
            
        return loss