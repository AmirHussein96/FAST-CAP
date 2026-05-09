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
import math

# added dependency from DAC
# from audiotools import AudioSignal
# from audiotools import STFTParams
from einops import rearrange
import typing
from typing import List, Optional, Tuple, Literal, Dict, Any, Callable

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

# https://github.com/jaywalnut310/vits/blob/main/modules.py
# removed mask
class WaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = activations.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts)
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


# ConvNextBlock from https://github.com/gemelo-ai/vocos/blob/main/vocos/modules.py
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x

# AMP blocks for BigVGAN
class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h
        
        self.anti_aliasing = getattr(self.h, "anti_aliasing", True)

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers
        
        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        self.use_cuda_kernel = getattr(h, "use_cuda_kernel", False)
        if self.use_cuda_kernel:
            assert self.anti_aliasing, "This model is not trained with anti-aliasing. Custom CUDA kernels not supported!"
            # faster CUDA kernel implementation of Activation1d
            from alias_free_cuda.activation1d import Activation1d as CudaActivation1d
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.Snake(channels, alpha_logscale=h.snake_logscale)) if self.anti_aliasing else activations.Snake(channels, alpha_logscale=h.snake_logscale)
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)) if self.anti_aliasing else activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h
        
        self.anti_aliasing = getattr(self.h, "anti_aliasing", True)

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs) # total number of conv layers
        
        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        self.use_cuda_kernel = getattr(h, "use_cuda_kernel", False)
        if self.use_cuda_kernel:
            assert self.anti_aliasing, "This model is not trained with anti-aliasing. Custom CUDA kernels not supported!"
            # faster CUDA kernel implementation of Activation1d
            from alias_free_cuda.activation1d import Activation1d as CudaActivation1d
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.Snake(channels, alpha_logscale=h.snake_logscale)) if self.anti_aliasing else activations.Snake(channels, alpha_logscale=h.snake_logscale)
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)) if self.anti_aliasing else activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip (self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
            
            
            
            
# from stable-audio-tools
def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None, use_cuda_kernel=False) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = activations.SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            # faster CUDA kernel implementation of Activation1d
            from alias_free_cuda.activation1d import Activation1d as CudaActivation1d
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d
        
        act = Activation1d(act)
    
    return act
            

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=7, use_snake=False, antialias_activation=False, causal=False, padding_mode='zeros'):
        super().__init__()

        self.dilation = dilation
        self.causal = causal
        self.kernel_size = kernel_size

        if causal:
            self.padding = dilation * (kernel_size - 1)
        else:
            self.padding = (dilation * (kernel_size - 1)) // 2
            
        # original non-causal impl used zero padding (DAC, SAVAE)
        # reflect padding may be better to reduce edge artifacts (EnCodec's default), but it increases VRAM usage during training (erm)
        self.padding_mode = padding_mode

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, dilation=dilation, padding=self.padding, padding_mode=self.padding_mode),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                     kernel_size=1, padding=0)
        )

    def forward(self, x):
        res = x

        # Disable checkpoint until tensor mismatch is fixed
        # x = checkpoint(self.layers, x)

        # apply conv layers
        x = self.layers(x)
        
        if self.causal:
            # Trim right padding to get the causal output
            x = x[:, :, :-self.padding]
            
        return x + res


class OobleckEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, causal=False, padding_mode='zeros'):
        super().__init__()

        self.causal = causal
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=in_channels,
                         dilation=1, use_snake=use_snake, causal=causal, padding_mode=padding_mode),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels,
                         dilation=3, use_snake=use_snake, causal=causal, padding_mode=padding_mode),
            ResidualUnit(in_channels=in_channels, out_channels=in_channels,
                         dilation=9, use_snake=use_snake, causal=causal, padding_mode=padding_mode),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            self._create_downsample_layer(in_channels, out_channels, stride, causal, padding_mode)
        )

    def _create_downsample_layer(self, in_channels, out_channels, stride, causal, padding_mode):
        if causal: # use EnCodec's SConv1d for convenience without reinventing the wheels. padding_mode is reflect by default
            downsample_layer = SConv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=2*stride, stride=stride, causal=True, norm='weight_norm')
        else: # original non-causal implmentation
            downsample_layer = WNConv1d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2), padding_mode=padding_mode)
        return downsample_layer

    def forward(self, x):
        return self.layers(x)
    
    def remove_weight_norm(self):
        for l in self.layers:
            remove_weight_norm(l)

class OobleckDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False, causal=False, padding_mode='zeros'):
        super().__init__()

        self.causal = causal

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            self._create_upsample_layer(in_channels, out_channels, stride, use_nearest_upsample, causal, padding_mode),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake, causal=causal, padding_mode=padding_mode),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake, causal=causal, padding_mode=padding_mode),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake, causal=causal, padding_mode=padding_mode),
        )
    
    def _create_upsample_layer(self, in_channels, out_channels, stride, use_nearest_upsample, causal, padding_mode):
        # NOTE: padding_mode parameter is not used in this function!
        
        if causal: # use EnCodec's SConvTransposed1d for convenience without reinventing the wheels. padding_mode is reflect by default
            assert not use_nearest_upsample, "use_nearest_upsample is not implemented for causal mode!"
            upsample_layer = SConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=2*stride, stride=stride, causal=True, norm='weight_norm')
        else:
            if use_nearest_upsample:
                upsample_layer = nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode="nearest"),
                    WNConv1d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=2*stride, stride=1, bias=False, padding='same')
                    )
            else:
                # WVConvTranspose1d only supports zeros padding mode so it's hardcoded
                upsample_layer = WNConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2), padding_mode='zeros')

        return upsample_layer

    def forward(self, x):
        return self.layers(x)
    
    def remove_weight_norm(self):
        for l in self.layers:
            remove_weight_norm(l)