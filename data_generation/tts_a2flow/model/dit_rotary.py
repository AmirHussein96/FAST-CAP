import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from model.base import BaseModule
from model.rotary_attention import RotaryEmbedding, Attention

Linear = nn.Linear


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = 1000 * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def modulate(x, shift, scale):
    dtype = x.dtype
    with amp.autocast(enabled=True, dtype=torch.float32):
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x.to(dtype)


#################################################################################
#                                 Core DiT Model                                #
# ################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, use_norm=False, 
        norm_type="RMS", causal=False, chunk_size=1, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim=hidden_size, dim_head=hidden_size//num_heads, heads=num_heads, flash=False, bias=True, use_norm=use_norm, norm=norm_type)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
            
    def forward(self, x, c, mask, rotary_emb=None, c_p=None, l_mask=None):
        # [B, C] -> [B, 6 * C] -> 3
        if c.ndim == 3 and c.shape[1] == 1:
            c = c.squeeze(1)
        if c_p is not None and c_p.ndim == 3 and c_p.shape[1] == 1:
            c_p = c_p.squeeze(1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        if c_p is not None:
            shift_msa_p, scale_msa_p, gate_msa_p, shift_mlp_p, scale_mlp_p, gate_mlp_p = self.adaLN_modulation(c_p).chunk(6, dim=1)
        # x: [B, T, C]
        # mask: [B, T, 1]
        
        x_norm1 = self.norm1(x) # [B, T, C]
        if c_p is not None:
            x_in = modulate(x_norm1, shift_msa_p, scale_msa_p) * (mask - l_mask) +  modulate(x_norm1, shift_msa, scale_msa) * l_mask
        else:
            x_in = modulate(x_norm1, shift_msa, scale_msa) * mask
        
        # rotary embedding attention
        x_in = self.attn(x_in, mask=mask.squeeze(-1).bool(), rotary_emb=rotary_emb) * mask
        if c_p is not None:
            x = x + (gate_msa_p.unsqueeze(1) * x_in) * (mask - l_mask) + (gate_msa.unsqueeze(1) * x_in) * l_mask
            x_norm2 = self.norm2(x)
            x = x + gate_mlp_p.unsqueeze(1) * self.mlp(modulate(x_norm2, shift_mlp_p, scale_mlp_p)) * (mask - l_mask)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(x_norm2, shift_mlp, scale_mlp)) * l_mask
        else:
            x = x + gate_msa.unsqueeze(1) * x_in
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x * mask


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, c, mask, c_p=None, l_mask=None):
        if c.ndim == 3 and c.shape[1] == 1:
            c = c.squeeze(1)
        if c_p is not None and c_p.ndim == 3 and c_p.shape[1] == 1:
            c_p = c_p.squeeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        if c_p is not None:
            shift_p, scale_p = self.adaLN_modulation(c_p).chunk(2, dim=1)
            x_norm = self.norm_final(x)
            x = modulate(x_norm, shift_p, scale_p) * (mask - l_mask) + modulate(x_norm, shift, scale) * l_mask
        else:    
            x = modulate(self.norm_final(x), shift, scale) * mask
        x = self.linear(x)
        return x * mask


class DiT_rotary(BaseModule):
    def __init__(
        self, n_feats=80, c_dim=80, hidden_size=1024, num_heads=16, 
        depth=12, mlp_ratio=4, n_language=1, use_norm=True, norm_type="LN", 
        use_prompt_emb=True, causal=False, chunk_size=1
    ):
        super(DiT_rotary, self).__init__()
        self.input_projection = Conv1d(n_feats + c_dim, hidden_size, 1)
        self.time_emb_size = 128
        self.time_pos_emb = SinusoidalPosEmb(self.time_emb_size)
        time_emb_in = self.time_emb_size * 2 if n_language > 1 else self.time_emb_size
        self.mlp = torch.nn.Sequential(torch.nn.Linear(time_emb_in, hidden_size), Mish(),
                                       torch.nn.Linear(hidden_size, hidden_size))

        self.rotary_emb = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_norm=use_norm, norm_type=norm_type) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, n_feats)
        self.initialize_weights()
        
        self.n_channels = hidden_size
        self.use_prompt_emb = use_prompt_emb
        if self.use_prompt_emb:
            self.prompt_emb = nn.Embedding(2, hidden_size)
            torch.nn.init.normal_(self.prompt_emb.weight, 0.0, hidden_size**-0.5)
            
        # language conditioning
        self.n_language = n_language
        if self.n_language > 1:
            self.lang_emb = torch.nn.Embedding(n_language, 128)
            torch.nn.init.normal_(self.lang_emb.weight, 0.0, 128**-0.5)
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, x, mask, mu, t, l_mask=None, language_id=None, 
        prompt_language_id=None, synth_language_id=None
    ):
        
        t_p = self.time_pos_emb(t) if prompt_language_id is not None else None
        t = self.time_pos_emb(t)
        if self.n_language > 1:
            lang = self.lang_emb(language_id) * math.sqrt(self.time_emb_size)
            # if t.ndim == 3:
            #     lang = lang.unsqueeze(1)  # [B, 1, 128]
            #     t = torch.cat((t, lang), dim=2)  # [B, 1, 256]
            # else:
                # Fallback for non-batch mode (e.g. t is [1, 128])
            t = torch.cat((t, lang), dim=1)  # [1, 256]

            if t_p is not None:
                prompt_lang = self.lang_emb(prompt_language_id) * math.sqrt(self.time_emb_size)
                # if t_p.ndim == 3:
                #     prompt_lang = prompt_lang.unsqueeze(1)  # [B, 1, 128]
                #     t_p = torch.cat((t_p, prompt_lang), dim=2)  # [B, 1, 256]
                # else:
                t_p = torch.cat((t_p, prompt_lang), dim=1)  # [1, 256]
            
        # Flatten time embeddings if needed for MLP input
        # if t.ndim == 3:
        #     t = t.squeeze(1)  # [B, D]
        # if t_p is not None and t_p.ndim == 3:
        #     t_p = t_p.squeeze(1)

        # Apply MLP to time embeddings
        t = self.mlp(t)  # [B, D]
        if t_p is not None:
            t_p = self.mlp(t_p)  # [B, D]

        # # Restore shape to [B, 1, D] after MLP
        # t = t.unsqueeze(1)
        # if t_p is not None:
        #     t_p = t_p.unsqueeze(1)

        # Project input features
        x = torch.cat([mu, x], dim=1)                  # [B, C_in, T]
        x = self.input_projection(x) * mask            # [B, C, T]

        # Add prompt embeddings based on local/global mask
        if self.use_prompt_emb:
            x += self.prompt_emb.weight[0].unsqueeze(-1) * math.sqrt(self.n_channels) * (mask - l_mask) 
            x += self.prompt_emb.weight[1].unsqueeze(-1) * math.sqrt(self.n_channels) * l_mask


        # Rotary embedding for attention
        rotary_emb = self.rotary_emb(x.size(-1))        # [H, T, D]
        # B, C, T -> B, T, C
        x = x.transpose(1, 2)
        mask = mask.transpose(1, 2)
        l_mask = l_mask.transpose(1, 2)

        # Transformer blocks
        for block in self.blocks:
            
            x = block(x, t, mask, rotary_emb, t_p, l_mask)  # [B, T, C]

        # Final output projection
        x = self.final_layer(x, t, mask, t_p, l_mask)       # [B, T, C_out]
        x = x.transpose(1, 2)                               # [B, C_out, T]
        return x