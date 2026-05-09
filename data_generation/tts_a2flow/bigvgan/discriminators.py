# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torchaudio
from torchaudio.transforms import Spectrogram

import activations
from xutils import init_weights, get_padding
from alias_free_torch.act import Activation1d as TorchActivation1d

import librosa

# added dependency from DAC
# from audiotools import AudioSignal
# from audiotools import STFTParams
from einops import rearrange
import typing
from typing import List, Optional, Tuple

# default leaky relu slope used in discriminators
LRELU_SLOPE = 0.1



class DiscriminatorP(torch.nn.Module):
    def __init__(self, h, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        
        self.stereo = getattr(h, "stereo", False)
                
        self.period = period
        self.d_mult = h.mpd_channel_mult
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, int(32*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(32*self.d_mult), int(128*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))
        
        # New in ADLR BigVGAN: option to volume normalize to MPD as well
        # this will match the behavior of recent models (DAC, Vocos, etc.) that applied volume norm across all discriminator modules
        self.mpd_normalize_volume = getattr(h, "mpd_normalize_volume", False)
        if self.mpd_normalize_volume:
            print(f"WARNING: mpd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in MPD!")
            
    def forward(self, x):
        if self.mpd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
            
        fmap = []
        
        # handle stereo same as sa2.0: put stereo channel to batch dim
        x = rearrange(x, "b ch t -> (b ch) 1 t", ch=2 if self.stereo else 1)
        
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        x = x.view(b, c, t // self.period, self.period)
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiPeriodDiscriminator, self).__init__()
        self.mpd_reshapes = h.mpd_reshapes
        print("mpd_reshapes: {}".format(self.mpd_reshapes))
        discriminators = [DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm) for rs in self.mpd_reshapes]
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, cfg, resolution):
        super().__init__()
        
        self.stereo = getattr(cfg, "stereo", False)

        self.resolution = resolution
        assert len(self.resolution) == 3, \
            "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE

        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, "mrd_use_spectral_norm"):
            print("INFO: overriding MRD use_spectral_norm as {}".format(cfg.mrd_use_spectral_norm))
            norm_f = weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
        self.d_mult = cfg.mpd_channel_mult
        if hasattr(cfg, "mrd_channel_mult"):
            print("INFO: overriding mrd channel multiplier as {}".format(cfg.mrd_channel_mult))
            self.d_mult = cfg.mrd_channel_mult
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, int(32*self.d_mult), (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)  # [B, ch, F, TT, 2]
        x = rearrange(x, "b ch f t c -> (b ch) c t f", ch=2 if self.stereo else 1) # handle stereo same as sa2.0
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert len(self.resolutions) == 3,\
            "MRD requires list of list with len=3, each element having a list with len=3. got {}".\
                format(self.resolutions)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    


# from https://github.com/gemelo-ai/vocos/blob/main/vocos/discriminators.py
class DiscriminatorRDAC(nn.Module):
    def __init__(
        self,
        h,
        window_length: int,
        num_embeddings: Optional[int] = None,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
    ):
        super().__init__()
        self.h = h
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(
            n_fft=window_length, hop_length=int(window_length * hop_factor), win_length=window_length, power=None
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        self.stereo = getattr(self.h, "stereo", False)
        self.in_channels = 2 # complex spectrogram
        if self.stereo:
            self.in_channels *= 2
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(self.in_channels, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=channels)
            torch.nn.init.zeros_(self.emb.weight)

        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = rearrange(x, "b ch f t c -> b (ch c) t f", ch=2 if self.stereo else 1) # handle stereo same as sa2.0
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
        x_bands = self.spectrogram(x)
        
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, LRELU_SLOPE)
                if i > 0:
                    fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h

        return x, fmap
    

class MultiResolutionDiscriminatorDAC(nn.Module):
    def __init__(
        self,
        h,
        fft_sizes: Tuple[int, ...] = (2048, 1024, 512),
        num_embeddings: Optional[int] = None,
    ):
        """
        Multi-Resolution Discriminator module adapted from https://github.com/descriptinc/descript-audio-codec.
        Additionally, it allows incorporating conditional information with a learned embeddings table.

        Args:
            fft_sizes (tuple[int]): Tuple of window lengths for FFT. Defaults to (2048, 1024, 512).
            num_embeddings (int, optional): Number of embeddings. None means non-conditional discriminator.
                Defaults to None.
        """

        super().__init__()
        self.h = h
        self.channels = int(32 * getattr(self.h, "mrd_channel_mult", 1))
        self.discriminators = nn.ModuleList(
            [DiscriminatorRDAC(h=h, window_length=w, channels=self.channels, num_embeddings=num_embeddings) for w in fft_sizes]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
# from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/discriminator.py
# also using fix https://github.com/descriptinc/descript-audio-codec/issues/36
# applies strided large kernel on time axis, NOT on freq. axis (current main on DAC)!

def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(LRELU_SLOPE))

# for default bands list for DiscriminatorCB otherwise specified
BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
class DiscriminatorCB(nn.Module):
    def __init__(
        self,
        cfg,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
        apply_large_kernel_to_freq: bool = False
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        
        self.stereo = getattr(cfg, "stereo", False)
        self.in_channels = 2
        if self.stereo:
            self.in_channels *= 2

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(self.in_channels, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)
        
        # if False, applies "fix" in https://github.com/descriptinc/descript-audio-codec/issues/36
        self.apply_large_kernel_to_freq = apply_large_kernel_to_freq
        if self.apply_large_kernel_to_freq:
            print("INFO: applying large kernel of MCBD to freq axis, same as official DAC")
        else:
            print("WARNING: aplying large kernel of MCBD to TIME axis, different to official DAC!")

    def spectrogram(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        # wrap to AudioSignal
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        if self.apply_large_kernel_to_freq:
            x = rearrange(x, "b ch f t c -> b (ch c) t f", ch=2 if self.stereo else 1) # handle stereo same as sa2.0
            # Split into bands
            x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        else:
            x = rearrange(x, "b ch f t c -> b (ch c) f t", ch=2 if self.stereo else 1) # t f became f t in the after
            # Split into bands
            x_bands = [x[..., b[0] : b[1], :] for b in self.bands] # frequency is now 2nd to last dim
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        if self.apply_large_kernel_to_freq:
            x = torch.cat(x, dim=-1)
        else:
            x = torch.cat(x, dim=-2) # concatenate on the frequency axis
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap
    

class MultiComplexBandDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        # to keep original terms, i know it's ugly
        self.sample_rate = cfg.sampling_rate
        
        self.mcbd_fft_sizes = cfg.mcbd_fft_sizes
        self.mcbd_hop_factor = cfg.mcbd_hop_factor
        self.mcbd_bands = cfg.mcbd_bands
        self.mcbd_apply_large_kernel_to_freq = cfg.mcbd_apply_large_kernel_to_freq
        
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCB(
                    cfg=cfg,
                    window_length=mcbd_fft_size, # element to loop over
                    hop_factor=self.mcbd_hop_factor,
                    sample_rate=self.sample_rate,
                    bands=self.mcbd_bands,
                    apply_large_kernel_to_freq=self.mcbd_apply_large_kernel_to_freq
                )
                for mcbd_fft_size in self.mcbd_fft_sizes
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


    
# https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/cqtd.py

class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg, hop_length, n_octaves, bins_per_octave):
        super(DiscriminatorCQT, self).__init__()
        self.cfg = cfg
        
        self.stereo = getattr(cfg, "stereo", False)

        self.filters = self.cfg.cqtd_filters
        self.max_filters = self.cfg.cqtd_max_filters
        self.filters_scale = self.cfg.cqtd_filters_scale
        self.kernel_size = (3, 9)
        self.dilations = self.cfg.cqtd_dilations
        self.stride = (1, 2)

        self.in_channels = self.cfg.cqtd_in_channels
        if self.stereo:
            self.in_channels *= 2
        self.out_channels = self.cfg.cqtd_out_channels
        self.fs = self.cfg.sampling_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # lazy-load
        from nnAudio import features
        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                ))
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            ))
        )

        self.conv_post = weight_norm(nn.Conv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        ))

        self.activation = torch.nn.LeakyReLU(negative_slope=LRELU_SLOPE)
        self.resample = torchaudio.transforms.Resample(orig_freq=self.fs, new_freq=self.fs * 2)
        
        # New in ADLR BigVGAN: option to volume normalize to CQTD as well
        # this will match the behavior of recent models (DAC, Vocos, etc.) that applied volume norm across all discriminator modules
        self.cqtd_normalize_volume = getattr(cfg, "cqtd_normalize_volume", False)
        if self.cqtd_normalize_volume:
            print(f"WARNING: cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!")
    
    def get_2d_padding(
            self, kernel_size: typing.Tuple[int, int], dilation: typing.Tuple[int, int] = (1, 1)
        ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x):
        fmap = []
        
        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        
        if self.stereo:
            # put stereo channel to batch dim for cqt processing
            x = rearrange(x, "b ch t -> (b ch) 1 t", ch=2)
            
        x = self.resample(x)
        
        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = rearrange(z, "b c w t -> b c t w")
        
        if self.stereo:
            # put stereo channel back to actual channel dim
            z = rearrange(z, "(b ch) c w t -> b (ch c) w t", ch=2)

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleSubbandCQTDiscriminator, self).__init__()

        self.cfg = cfg
        # Using getattr with defaults
        self.cfg.cqtd_filters = getattr(self.cfg, "cqtd_filters", 32)
        self.cfg.cqtd_max_filters = getattr(self.cfg, "cqtd_max_filters", 1024)
        self.cfg.cqtd_filters_scale = getattr(self.cfg, "cqtd_filters_scale", 1)
        self.cfg.cqtd_dilations = getattr(self.cfg, "cqtd_dilations", [1, 2, 4])
        self.cfg.cqtd_in_channels = getattr(self.cfg, "cqtd_in_channels", 1)
        self.cfg.cqtd_out_channels = getattr(self.cfg, "cqtd_out_channels", 1)
        # multi-scale params to loop over
        self.cfg.cqtd_hop_lengths = getattr(self.cfg, "cqtd_hop_lengths", [512, 256, 256])
        self.cfg.cqtd_n_octaves = getattr(self.cfg, "cqtd_n_octaves", [9, 9, 9])
        self.cfg.cqtd_bins_per_octaves = getattr(self.cfg, "cqtd_bins_per_octaves", [24, 36, 48])

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    cfg,
                    hop_length=self.cfg.cqtd_hop_lengths[i],
                    n_octaves=self.cfg.cqtd_n_octaves[i],
                    bins_per_octave=self.cfg.cqtd_bins_per_octaves[i],
                )
                for i in range(len(self.cfg.cqtd_hop_lengths))
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
class CombinedDiscriminator(nn.Module):
    # wrapper of chaining multiple discrimiantor
    # ex: combine mrd and cqtd
    def __init__(self, list_discriminator):
        super().__init__()
        self.discrimiantor = nn.ModuleList(list_discriminator)
        
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for disc in self.discrimiantor:
            y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)
            y_d_rs.extend(y_d_r)
            fmap_rs.extend(fmap_r)
            y_d_gs.extend(y_d_g)
            fmap_gs.extend(fmap_g)
            
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs