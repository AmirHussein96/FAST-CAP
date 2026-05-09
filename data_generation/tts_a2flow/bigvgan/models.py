# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torchaudio.transforms import Spectrogram

import activations
from xutils import init_weights, get_padding
from alias_free_torch.act import Activation1d as TorchActivation1d

from modules import AMPBlock1, AMPBlock2, AdaLayerNorm, ConvNeXtBlock, WaveNet, OobleckEncoderBlock, OobleckDecoderBlock, WNConv1d, get_activation

# added dependency from DAC
# from audiotools import AudioSignal
# from audiotools import STFTParams
from einops import rearrange
import typing
from typing import List, Dict, Optional, Tuple

# function that defines generator.forward() logic
# return_dict_all contains k-v pairs according to model_type
def apply_generator_forward(model_type, generator, x):
    # Handle potential DataParallel/ DistributedDataParallel wrappers
    generator_for_forward = generator.module if hasattr(generator, 'module') else generator
    
    # Initialize default output dictionary
    # currently it contains 5 keys
    return_dict_all = {
        "encoder_out": None,
        "latent": None,
        "decoder_out": None,
        "mu": None,
        "logvar": None
    }
    
    if model_type == "vocoder":
        # For vocoder, only decode the input without encoding
        return_dict_all["decoder_out"] = generator_for_forward(x)
    else:
        # For autoencoder and VAE, process through the full model
        return_dict_model = generator_for_forward(x)
        # Update the return dictionary with model outputs
        return_dict_all.update({
            "encoder_out": return_dict_model.get("encoder_out"),
            "latent": return_dict_model.get("latent"),
            "decoder_out": return_dict_model.get("decoder_out"),
            "mu": return_dict_model.get("mu", None),
            "logvar": return_dict_model.get("logvar", None)
        })
        
    return return_dict_all

class TrimPadding(nn.Module):
    """
    Used for causal convolution support of a conv layer wrapped with nn.Sequential
    """
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return x[:, :, :-self.padding]

# from stable-audio-tools
class OobleckEncoder(nn.Module):
    def __init__(
            self,
            h
        ):
        super().__init__()        
        
        self.h = h
        
        in_channels = self.h.input_channels
        if getattr(h, "stereo", False):
            in_channels *= 2
            
        latent_dim = self.h.vocoder_input_dim
        if self.h.model_type == "vae":
            latent_dim *= 2

        channels = self.h.enc_dim
        c_mults = self.h.c_mults
        strides = self.h.strides
        use_snake = self.h.use_snake
        antialias_activation = self.h.anti_aliasing
        causal = self.h.causal
        
        padding_mode = self.h.padding_mode
        
        self.output_channels = c_mults[-1]*channels
        
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        # Padding for the first convolution layer
        self.first_padding = 6 if causal else 3
        first_conv = WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=self.first_padding, padding_mode=padding_mode)
        
        if causal:
            first_conv = nn.Sequential(first_conv, TrimPadding(self.first_padding))
            
        layers = [first_conv]
        
        for i in range(self.depth - 1):
            layers += [OobleckEncoderBlock(
                in_channels=c_mults[i] * channels,
                out_channels=c_mults[i + 1] * channels,
                stride=strides[i],
                use_snake=use_snake,
                antialias_activation=antialias_activation,
                causal=causal,
                padding_mode=padding_mode,
            )]

        # Padding for the final convolution layer
        self.final_padding = 2 if causal else 1
        final_conv = WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=self.final_padding, padding_mode=padding_mode)
        
        if causal:
            final_conv = nn.Sequential(final_conv, TrimPadding(self.final_padding))

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            final_conv
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x.transpose(1, 2) # [B, T, C] to apply linear projection outside
    
    def remove_weight_norm(self):
        print(f"INFO: Removing all weight norm from OobleckEncoder")
        for module in self.modules():
            if hasattr(module, "parametrizations"):  # for new WN implementation using parameterizations
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    print(f"[WARNING] No weight norm found in {module} with parameterizations. You can ignore this if you know that this module does not apply weight norm.")
            elif hasattr(module, "weight"):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    print(f"[WARNING] No weight norm found in {module} with legacy method. You can ignore this if you know that this module does not apply weight norm.")
                    
                    
class OobleckDecoder(nn.Module):
    def __init__(
        self,
        h,
    ):
        super().__init__()
        
        self.h = h
        
        latent_dim = self.h.vocoder_input_dim
        
        out_channels = self.h.input_channels
        if getattr(h, "stereo", False):
            out_channels *= 2
            
        channels = self.h.dec_dim
        c_mults = self.h.c_mults
        strides = self.h.strides
        use_snake = self.h.use_snake
        use_nearest_upsample = self.h.use_nearest_upsample
        antialias_activation = self.h.anti_aliasing
        causal = self.h.causal
        final_tanh = self.h.use_tanh_at_final
        padding_mode = self.h.padding_mode

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        # Padding for the first convolution layer
        self.first_padding = 6 if causal else 3
        first_conv = WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=self.first_padding, padding_mode=padding_mode)

        if causal:
            first_conv = nn.Sequential(first_conv, TrimPadding(self.first_padding))

        layers = [first_conv]
        
        for i in range(self.depth-1, 0, -1):
            layers += [OobleckDecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                use_snake=use_snake, 
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample,
                causal=causal,
                padding_mode=padding_mode,
            )]

        # Padding for the final convolution layer
        self.final_padding = 6 if causal else 3
        final_conv = WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=self.final_padding, padding_mode=padding_mode, bias=False)
        
        if causal:
            final_conv = nn.Sequential(final_conv, TrimPadding(self.final_padding))
            
        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            final_conv,
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
    def remove_weight_norm(self):
        print(f"INFO: Removing all weight norm from OobleckDecoder")
        for module in self.modules():
            if hasattr(module, "parametrizations"):  # for new WN implementation using parameterizations
                try:
                    remove_parametrizations(module, "weight")
                except ValueError:
                    print(f"[WARNING] No weight norm found in {module} with parameterizations. You can ignore this if you know that this module does not apply weight norm.")
            elif hasattr(module, "weight"):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    print(f"[WARNING] No weight norm found in {module} with legacy method. You can ignore this if you know that this module does not apply weight norm.")


class BigVGAN(nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(
        self,
        h
    ):
        super().__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.stereo = getattr(self.h, "stereo", False)
        self.anti_aliasing = getattr(self.h, "anti_aliasing", True)
        if not self.anti_aliasing:
            print("WARNING: turning off anti-aliased activation in BigVGAN. This model is NOT compatible with CUDA kernels!")
        if "vocoder_input_dim" in h: # used by autoencoder
            self.in_channels = h.vocoder_input_dim
        else: # standalone vocoder
            self.in_channels = h.num_mels
            if self.stereo:
                self.in_channels *= 2
        
        # pre conv
        self.conv_pre = weight_norm(Conv1d(
            self.in_channels,
            h.upsample_initial_channel,
            7,
            1,
            padding=3
        ))

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # for odd-numbered upsampling support
            op = (k - u) % 2
            p = (k - u + op) // 2
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k, u,
                    padding=p,
                    output_padding=op
                ))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        self.use_cuda_kernel = getattr(h, "use_cuda_kernel", False)
        if self.use_cuda_kernel:
            assert self.anti_aliasing, "This model is not trained with anti-aliasing. Custom CUDA kernels not supported!"
            # faster CUDA kernel implementation of Activation1d
            from alias_free_cuda.activation1d import Activation1d as CudaActivation1d
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # post conv
        if h.activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = activations.Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post) if self.anti_aliasing else activation_post
        elif h.activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post) if self.anti_aliasing else activation_post
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.use_bias_at_final = getattr(h, "use_bias_at_final", True)
        if not self.use_bias_at_final:
            print("WARNING: BigVGAN's use_bias_at_final set to False!")
        self.conv_post = weight_norm(Conv1d(
            ch, 2 if self.stereo else 1, 7, 1, padding=3, bias=self.use_bias_at_final
            )) # whether to add bias=false? but that will break legacy models

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)
        
        # final tanh activation
        self.use_tanh_at_final = getattr(h, "use_tanh_at_final", True)
        if not self.use_tanh_at_final:
            print("WARNING: BigVGAN's use_tanh_at_final set to False!")

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1., max=1.) # bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        print('Removing weight norm of BigVGAN...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# good old spec posterior encoder from vits https://github.com/jaywalnut310/vits/blob/main/models.py
# removes mask since there's no mask in bigvgan
# removes proj & vae reparam. we'll do these outside
class WaveNetEncoder(nn.Module):
    """
    A WaveNet-based encoder module adapted from the VITS model, designed for
    encoding spectrogram inputs into a latent representation.
    
    Attributes:
        h: A configuration object containing model hyperparameters.
        in_channels (int): The number of input channels (e.g., mel bins).
        hidden_channels (int): The number of hidden channels in the WaveNet model.
        kernel_size (int): The kernel size for convolutions within the WaveNet model.
        dilation_rate (int): The dilation rate for convolutions within the WaveNet model.
        n_layers (int): The number of layers in the WaveNet model.
        gin_channels (int): The number of global condition input channels, if any.

    Inputs:
        x (torch.Tensor): The input tensor representing a batch of mel spectrograms
                          with shape [batch_size, num_mels, sequence_length].
        g (torch.Tensor, optional): A global conditioning tensor, if required by
                                    the model. Defaults to None.

    Outputs:
        torch.Tensor: The encoded tensor representing the latent representations
                      of the input spectrograms, with shape
                      [batch_size, encoded_sequence_length, hidden_channels] ([B, T, C]).

    Example usage:
        encoder = WaveNetEncoder(h)
        mel_spectrograms = torch.randn(batch_size, num_mels, sequence_length)
        latent_representations = encoder(mel_spectrograms)
    """
    def __init__(
        self,
        h,
    ):
        super().__init__()
        self.h = h
        self.input_channels = self.h.input_channels
        self.stereo = getattr(self.h, "stereo", False)
        if self.stereo:
            self.in_channels *= 2
        self.hidden_channels = h.enc_dim
        self.kernel_size = h.enc_kernel_size
        self.dilation_rate = h.enc_dilation_rate
        self.n_layers = h.enc_num_layers
        self.gin_channels = getattr(self.h, "enc_gin_channels", 0)
        
        self.output_channels = h.enc_dim

        self.pre = nn.Conv1d(self.input_channels, self.hidden_channels, 1)
        self.enc = WaveNet(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            gin_channels=self.gin_channels,
        )
        # self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, g=None):
        x = self.pre(x)
        x = self.enc(x, g=g)
        # stats = self.proj(x)
        return x.transpose(1, 2) # [B, T, C] to apply linear projection outside
    
    def remove_weight_norm(self):
        print('Removing weight norm of WaveNetEncoder...')
        self.enc.remove_weight_norm()


# convnext encoder using arch of vocos backbone https://github.com/gemelo-ai/vocos/blob/main/vocos/models.py
class ConvNeXTEncoder(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        h,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.h = h
        self.input_channels = self.h.input_channels
        self.stereo = getattr(self.h, "stereo", False)
        if self.stereo:
            self.input_channels *= 2
        self.dim = self.h.enc_dim
        self.intermediate_dim = self.h.enc_intermediate_dim
        self.num_layers = self.h.enc_num_layers
        
        self.output_channels = self.h.enc_dim
        
        self.embed = nn.Conv1d(self.input_channels, self.dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, self.dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.layer_scale_init_value = layer_scale_init_value or 1 / self.num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=self.layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x # return shape is [B, T, C]

    def remove_weight_norm(self):
        print('ConvNeXTEncoder does not include weight_norm. skipping!')
        pass


class LatentAutoEncoder(nn.Module):
    """
    A Latent AutoEncoder class supporting both Variational AutoEncoder (VAE)
    and standard autoencoder configurations.
    Attributes:
        h: Configuration object containing model hyperparameters.
        model_type (str): Type of model to use, either "autoencoder" or "vae".
        encoder (nn.Module): The encoder module, specified here as ConvNeXTEncoder.
        encoder_proj_dim (int): Dimension of the projected encoder output.
        encoder_proj (nn.Linear): Linear projection layer applied to the encoder output.
        quantizer: Placeholder for potential future quantizer modules (e.g., VQ-VAE).
        decoder (nn.Module): The decoder module, specified here as BigVGAN.
    """
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.model_type = self.h.model_type
        assert self.model_type in ["autoencoder", "vae"], f"model_type {self.model_type} is not implemented"
        
        self.stereo = getattr(self.h, "stereo", False)
        
        self.input_type = None
        if getattr(self.h, "use_wav_as_input", False):
            print(f"INFO: Encoder's input feature is waveform")
            self.input_type = "waveform"
            self.h.input_channels = 1
        elif getattr(self.h, "use_linear_spec_as_input", False):
            print(f"INFO: Encoder's input feature is linear")
            self.input_type = "linear"
            self.h.input_channels = self.h.num_linears
        else:
            print(f"INFO: Encoder's input feature is mel")
            self.input_type = "mel"
            self.h.input_channels = self.h.num_mels
            
        self.enc_type = getattr(self.h, "enc_type", "convnext")
        print(f"INFO: using {self.enc_type} as encoder")
        # Initialize the encoder module
        if self.enc_type == "convnext":
            self.encoder = ConvNeXTEncoder(self.h)
        elif self.enc_type == "wavenet":
            self.encoder = WaveNetEncoder(self.h)
        elif self.enc_type == "oobleck":
            self.encoder = OobleckEncoder(self.h)
        else:
            raise NotImplementedError(f"unknown enc_type {self.enc_type}")
        
        if self.enc_type == "oobleck":
            self.encoder_proj = nn.Identity()
        else:
            # Initialize the encoder projection layer. For VAE, the dimension is doubled
            # to accommodate both mean (mu) and log variance (logvar) of the latent space.
            self.encoder_proj_dim = self.h.vocoder_input_dim * 2 if self.model_type == "vae" else self.h.vocoder_input_dim
            self.encoder_proj_bias = getattr(self.h, "encoder_proj_bias", False)
            if self.encoder_proj_bias:
                print("INFO: turning on bias term in encoder_proj")
            self.encoder_proj = nn.Linear(self.encoder.output_channels, self.encoder_proj_dim, bias=self.encoder_proj_bias)
    
        # Placeholder for quantization layers, if needed in the future
        self.quantizer = None
        
        # Initialize the decoder module
        self.dec_type = getattr(self.h, "dec_type", "bigvgan")
        print(f"INFO: using {self.dec_type} as decoder")
        if self.dec_type == "oobleck":
            self.decoder = OobleckDecoder(self.h)
        elif self.dec_type == "bigvgan":
            self.decoder = BigVGAN(self.h)
        else:
            raise NotImplementedError(f"unknown dec_type {self.dec_type}")
        
        # vae specific switches
        if self.model_type == "vae":
            self.use_vae_reparameterize_v2 = getattr(self.h, "use_vae_reparameterize_v2", False)
            if self.use_vae_reparameterize_v2:
                print("INFO: using vae_reparameterize_v2 (using softplus to scale to improve stability)")
                
        # whether to freeze encoder
        self.freeze_encoder =  getattr(self.h, "freeze_encoder", False)
        if self.freeze_encoder:
            print("WARNING: freeze_encoder set to true. The encoder will not be updated during training, make sure you load a pretrained checkpoint!")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def vae_reparameterize(self, encoder_out_proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for VAE to sample from N(mu, var) using N(0, 1).
        This version is used for first Audio VAE models in this repo.
        Args:
            encoder_out_proj (torch.Tensor): Encoded output projections with shape [B, T, 2*C] that contain both mean
                                            and log variance concatenated together.
        Returns:
            torch.Tensor: Sampled latent vector with shape [B, T, C].
            torch.Tensor: Mean of the latent Gaussian distribution with shape [B, T, C].
            torch.Tensor: Log variance of the latent Gaussian distribution with shape [B, T, C].
        """
        mu, logvar = torch.split(encoder_out_proj, self.h.vocoder_input_dim, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = eps * std + mu
        return latent, mu, logvar

    def vae_reparameterize_v2(self, encoder_out_proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        An alternative version of the reparameterization trick for VAEs where variance is derived
        from a scale parameter transformed by a softplus function to ensure positivity.
        refelcts stable-audio-tools VAE:
        https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/bottleneck.py
        Args:
            encoder_out_proj (torch.Tensor): Encoded output projections with shape [B, T, 2*C] that contain both mean
                                            and scale values concatenated together.
        Returns:
            torch.Tensor: Sampled latent vector with shape [B, T, C].
            torch.Tensor: Mean of the latent Gaussian distribution with shape [B, T, C].
            torch.Tensor: Log variance of the latent Gaussian distribution with shape [B, T, C].
        """
        mu, scale = torch.split(encoder_out_proj, self.h.vocoder_input_dim, dim=-1)
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latent = torch.randn_like(mu) * stdev + mu
        return latent, mu, logvar
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model with shape [B, C, T].
        Returns:
            Dict[str, torch.Tensor]: Dictionary of output tensors all with shape [B, C, T], including
            encoder output, latent representation, decoder output, and optionally mu and logvar.
        """

        encoder_out = self.encoder(x)  # Shape: [B, T_frame, encoder_out_dim]
        encoder_out_proj = self.encoder_proj(encoder_out)  # Shape: [B, T_frame, vocoder_input_dim(*2)]
        
        if self.model_type == "vae":
            if self.use_vae_reparameterize_v2:
                latent, mu, logvar = self.vae_reparameterize_v2(encoder_out_proj) # [B, T_frame, vocoder_input_dim]
            else:
                latent, mu, logvar = self.vae_reparameterize(encoder_out_proj) # [B, T_frame, vocoder_input_dim]

            decoder_out = self.decoder(latent.transpose(1, 2)) # [B, C, T_frame] -> [B, 1, T_time]
            return { # all with shape [B, C, T]
                "encoder_out": encoder_out.transpose(1, 2),
                "latent": latent.transpose(1, 2),
                "decoder_out": decoder_out,
                "mu": mu.transpose(1, 2),
                "logvar": logvar.transpose(1, 2)
            }
        elif self.model_type == "autoencoder":
            # For standard autoencoder, use the projected encoder output directly as latent
            latent = encoder_out_proj
            decoder_out = self.decoder(latent.transpose(1, 2)) # [B, C, T_frame] -> [B, 1, T_time]
            return { # all with shape [B, C, T]
                "encoder_out": encoder_out.transpose(1, 2),
                "latent": latent.transpose(1, 2),
                "decoder_out": decoder_out
            }
        else:
            raise NotImplementedError
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encodes input x into latent representation. This method supports
        both the VAE and standard autoencoder configurations.
        Args:
            x (torch.Tensor): Input tensor with shape [B, C, T], where B is the batch size,
                            C is the number of channels (e.g., number of mel frequency bins),
                            and T is the temporal dimension (e.g., time or sequence length).
        Returns:
            Dict[str, torch.Tensor]: Contains the latent representation and, for VAE, 'mu' and 'logvar'.
                                    All outputs are in the shape [B, C, T]:
                                    - "latent": The latent representation. For VAE, this is the reparameterized latent space.
                                    - "mu": Mean of the latent Gaussian distribution for VAE (not present for standard autoencoder).
                                    - "logvar": Log variance of the latent Gaussian distribution for VAE (not present for standard autoencoder).
                                    The C dimension is `encoder_proj_dim` for autoencoder and `vocoder_input_dim` for VAE.
        """
        encoder_out = self.encoder(x)
        encoder_out_proj = self.encoder_proj(encoder_out)
        
        if self.model_type == "vae":
            if self.use_vae_reparameterize_v2:
                latent, mu, logvar = self.vae_reparameterize_v2(encoder_out_proj)
            else:
                latent, mu, logvar = self.vae_reparameterize(encoder_out_proj)
            return {
                "latent": latent.transpose(1, 2),
                "mu": mu.transpose(1, 2),
                "logvar": logvar.transpose(1, 2)
            }
        else:  # autoencoder
            latent = encoder_out_proj
            return {
                "latent": latent.transpose(1, 2)
            }

    def decode(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decodes latent representation into output. This method applies the decoder
        to the given latent representation to produce the output tensor.
        Args:
            latent (torch.Tensor): Latent representation with shape [B, C, T] where
                                C is the dimension of the encoded latent space which
                                is `encoder_proj_dim` for autoencoder configurations
                                and `vocoder_input_dim` for VAE configurations. T is
                                the temporal dimension after encoding.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the decoded output.
                                    The output tensor is in the shape [B, C, T], where
                                    C may differ from the input channels based on decoder specifics,
                                    and T is the temporal dimension of the decoded output.
                                    - "decoder_out": The output from the decoder.
        """
        # For both VAE and autoencoder, decode the latent representation
        decoder_out = self.decoder(latent) # [B, C, T_frame] -> [B, 1, T_time]
        return {
            "decoder_out": decoder_out
        }
        
    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()
        self.decoder.remove_weight_norm()
        
def test_autoencoder():
    import json
    from env import AttrDict
    print(f"running unit test of {__file__}")
    
    # Load configuration
    config_path = "configs_vae/vae_wavenet_v3loss_22k80b_bs128_ss16k.json"
    with open(config_path) as f:
        config = f.read()
    json_config = json.loads(config)
    h = AttrDict(**json_config)
    model = LatentAutoEncoder(h).cuda()

    # Define random data (e.g., batch size of 4 for simplicity)
    random_data = torch.randn(4, h['num_mels'], int(h['segment_size']/h['hop_size'])).cuda()  # Adjust dimensions as needed

    # Forward pass
    forward_output = model.forward(random_data)
    print("Forward output shapes:")
    for key, value in forward_output.items():
        print(f"{key}: {value.shape}")

    # Encode
    encoded_output = model.encode(random_data)
    print("\nEncoded output shapes:")
    for key, value in encoded_output.items():
        print(f"{key}: {value.shape}")

    # Decode (using latent from encode as an example)
    latent = encoded_output.get('latent')
    if latent is not None:
        decoded_output = model.decode(latent)
        print("\nDecoded output shape:", decoded_output['decoder_out'].shape)
    else:
        print("\nNo latent representation available for decoding.")

if __name__ == "__main__": # unit test
    test_autoencoder()