import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda import amp
from torch.nn import Module, ModuleList

from einops import rearrange

from model.attend import Attend

# rmsnorm

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# rotary embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# fp32
def apply_rotary_pos_emb(pos, t):
    dtype = t.dtype
    with amp.autocast(enabled=True, dtype=torch.float32):
        t = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return t.to(dtype)

# Attention

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, flash=True, bias=False, use_norm=False, norm="RMS"):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.use_norm = use_norm
        if self.use_norm:
            if norm == "RMS":
                self.q_norm = RMSNorm(dim)
                self.k_norm = RMSNorm(dim)
            elif norm == "LN":
                self.q_norm = nn.LayerNorm(dim)
                self.k_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None
    ):
        q, k, v = (self.to_qkv(x).chunk(3, dim = -1))
        if self.use_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = apply_rotary_pos_emb(rotary_emb, q)
        k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# https://github.com/lucidrains/spear-tts-pytorch/blob/main/spear_tts_pytorch/spear_tts_pytorch.py
    
# residual wrapper

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_dropout = 0.,
        attn_flash = False,
        bias=False,
        norm_type="RMS"
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, flash=attn_flash, bias=bias, use_norm=True, norm=norm_type),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))
        self.final_norm = RMSNorm(dim)

    def forward(self, x, mask = None, rotary_emb=None):
        for ind, (self_attn, ff) in enumerate(self.layers):
            residual = x
            attn_out = self_attn(x, mask=mask, rotary_emb=rotary_emb)
            x = attn_out + residual
            x = ff(x) + x
        out = self.final_norm(x)   
        return out
