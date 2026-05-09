import math
import torch
import torch.nn as nn
from model.base import BaseModule
from model.utils import sequence_mask, convert_pad_shape
from functools import partial
from model.rotary_attention import RotaryEmbedding, Attention, Transformer


class LayerNorm(BaseModule):
    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, 
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, 
                                                kernel_size, padding=kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels, 
                                                    kernel_size, padding=kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

    
class DurationPredictor_Mel(BaseModule):
    def __init__(self, n_vocab, n_feats, n_channels, n_heads, n_layers, n_language=1):
        super(DurationPredictor_Mel, self).__init__()
        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_language = n_language

        self.emb = torch.nn.Embedding(n_vocab + 1, n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        self.proj_p = torch.nn.Conv1d(n_feats, n_channels, 1)
        self.prenet = ConvReluNorm(n_channels, n_channels, n_channels, kernel_size=5, n_layers=3, p_dropout=0.1)
        self.prompt_emb = nn.Embedding(2, n_channels)
        torch.nn.init.normal_(self.prompt_emb.weight, 0.0, n_channels**-0.5)
        
        self.rotary_emb = RotaryEmbedding(n_channels // n_heads)
        self.transformer_encoder = Transformer(dim=n_channels, depth=n_layers, dim_head=n_channels//n_heads, heads=n_heads)
        self.proj = nn.Sequential(
            nn.Linear(n_channels, 1, bias = False),
            nn.Softplus()
        )
        if n_language > 1:
            self.lang_emb = torch.nn.Embedding(n_language, n_channels)
            torch.nn.init.normal_(self.lang_emb.weight, 0.0, n_channels**-0.5)
            self.proj_lang = torch.nn.Conv1d(2 * n_channels, n_channels, 1)

    def forward(self, x, x_lengths, p=None, p_lengths=None, language_id=None):
        # p: mel 3-second audio
        p_mask = sequence_mask(p_lengths, p.shape[-1]).unsqueeze(1).to(p)
        p = self.proj_p(p) * p_mask
        p_max_length = p.size(-1)
        p = self.prenet(p, p_mask)
        p += self.prompt_emb.weight[0].unsqueeze(-1) * math.sqrt(self.n_channels)
        rotary_emb_p = self.rotary_emb(p.size(-1))
        
        predict_token = torch.ones((x.size(0), 1)).to(x) * self.n_vocab
        x = torch.cat((predict_token, x), dim=1)
        x_lengths = x_lengths + 1
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)            
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        if self.n_language > 1:
            lang = self.lang_emb(language_id) * math.sqrt(self.n_channels)
            lang = lang.unsqueeze(-1).repeat(1, 1, x.size(-1))
            x = torch.cat((x, lang), dim=1)
            x = self.proj_lang(x) * x_mask
        x += self.prompt_emb.weight[1].unsqueeze(-1) * math.sqrt(self.n_channels)
         
        rotary_emb = self.rotary_emb(x.size(-1))
        
        x = torch.cat((p, x), dim=-1)
        x_mask = torch.cat((p_mask, x_mask), dim=-1)
        rotary_emb = torch.cat((rotary_emb_p, rotary_emb), dim=0)
        
        x = self.transformer_encoder(x.transpose(1, 2), mask=x_mask.squeeze(1).bool(), rotary_emb=rotary_emb).transpose(1, 2) * x_mask
        # take predict_token
        x = x[:, :, p_max_length]
        w = self.proj(x)
        return w