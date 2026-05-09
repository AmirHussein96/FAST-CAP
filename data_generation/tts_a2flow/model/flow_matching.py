import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModule
from model.dit_rotary import DiT_rotary


class FlowMatching(BaseModule):
    def __init__(
        self, n_vocab, n_feats, c_dim, hidden_size, 
        num_heads, num_layers=12, sigma=0.01, n_language=1
    ):
        super(FlowMatching, self).__init__()
        self.n_feats = n_feats
        self.c_dim = c_dim
        self.sigma = sigma
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mask_token = nn.Parameter(torch.randn(1, c_dim))
        self.proj_mu = nn.Linear(c_dim, c_dim)
        self.proj_x = nn.Linear(n_feats, c_dim)
        self.emb = torch.nn.Embedding(n_vocab, c_dim)
        torch.nn.init.normal_(self.emb.weight, 0.0, c_dim**-0.5)
        
        self.estimator = DiT_rotary(n_feats=n_feats, c_dim=c_dim, hidden_size=hidden_size, num_heads=num_heads, 
                                    depth=num_layers, n_language=n_language)
        
    def skewed_sampling(self, alpha, t):
        # for inference only
        return t / (1 + 1e-4 + (alpha - 1) * (1 - t))
    
    @torch.no_grad()
    def forward(self, x0, mask, mu=None, gradient_scale=0., n_timesteps=32, p=None, p_mask=None, 
            language_id=None, prompt_language_id=None, synth_language_id=None, alpha=1.
        ):
        # x0: [B, C, T_p+T_y]
        # mu: [B, C, T_p+T_y]
        # mask: [B, 1, T_p+T_y], p_mask: [B, 1, T_p]
        x = x0
        x_prompt = torch.zeros_like(x)
        x_p = torch.randn_like(p) * p_mask
        x[:, :, :p.size(-1)] = x_p
        l_mask = torch.zeros_like(mask)
        l_mask[:, :, p.size(-1):] = mask[:, :, p.size(-1):]
        x_prompt[:, :, :p.size(-1)] = p
        
        mu_c, mu_u = self.process_input(x_prompt, mask, mu, l_mask)
        # mu_c: [B, C', T_p+T_y] C' here is 512
        # mu_u: [B, C', T_p+T_y]
        
        for i in range(n_timesteps):
            t = torch.tensor([i / n_timesteps], dtype=x0.dtype, device=x0.device, requires_grad=False)
            t_p1 = torch.tensor([(i + 1) / n_timesteps], dtype=x0.dtype, device=x0.device, requires_grad=False)
            t = self.skewed_sampling(alpha, t)
            t_p1 = self.skewed_sampling(alpha, t_p1)
            t_ = t.unsqueeze(-1).unsqueeze(-1)
            if p is not None:
                xt_p = (1 - (1 - self.sigma) * t_) * x_p + t_ * p
                x[:, :, :p.size(-1)] = xt_p
            
            vt = self.vt(x, mask, mu_c, mu_u, t, l_mask, gradient_scale=gradient_scale,
                language_id=language_id,
                prompt_language_id=prompt_language_id,
                synth_language_id=synth_language_id
            )
            x += vt * (t_p1 - t)
        x[:, :, :p.size(-1)] = p
        x_cut = x[:, :, p.size(-1):]
        return x, x_cut

    def vt(self, x, mask, mu_c, mu_u, t, l_mask, gradient_scale=0.,
        language_id=None, prompt_language_id=None, synth_language_id=None
    ):  
        if t.size(0) != x.size(0):
            # t = t.repeat(x.shape[0],1)
            t = t.repeat(x.shape[0])
        if gradient_scale == 0:
            if language_id is None:
                language_id = synth_language_id
            vt = self.estimator(x, mask, mu_c, t, l_mask,
                language_id=language_id,
                prompt_language_id=prompt_language_id,
                synth_language_id=synth_language_id
            )
        else:
            x_g = torch.cat((x, x), dim=0)
            mask_g = torch.cat((mask, mask), dim=0)
            mu_g = torch.cat((mu_c, mu_u), dim=0)
            l_mask_g = torch.cat((l_mask, l_mask), dim=0)
            if language_id is not None:
                language_id_g = torch.cat((language_id, language_id), dim=0)
            else:
                language_id_g = torch.cat((synth_language_id, synth_language_id), dim=0)
                synth_language_id_g = language_id_g
            if prompt_language_id is not None:
                prompt_language_id_g = torch.cat((prompt_language_id, prompt_language_id), dim=0)
            else:
                prompt_language_id_g = None
            if synth_language_id is not None:
                synth_language_id_g = torch.cat((synth_language_id, synth_language_id), dim=0)
            else:
                synth_language_id_g = None
            if t.dim() == 1:
                t = t.repeat(2)
            else:
                t = t.repeat(2,1)
            vt = self.estimator(x_g, mask_g, mu_g, t, l_mask_g,
                language_id=language_id_g,
                prompt_language_id=prompt_language_id_g,
                synth_language_id=synth_language_id_g
            )
            B = vt.size(0) // 2
            vt, vt_u = vt[0:B], vt[B:]
            vt += gradient_scale * (vt - vt_u) * mask
        return vt

    def loss_t(self, x1, mask, mu=None, t=None, l_mask=None,
        language_id=None, prompt_language_id=None, synth_language_id=None
    ):
        # l_mask: 1 -> compute loss, 0 -> prompt
        B, C, T = x1.size()
        idx_u = (torch.rand((B,), device=x1.device) <= 0.2)
        l_mask[idx_u] = mask[idx_u]
        mu_c, mu_u = self.process_input(x1, mask, mu, l_mask)
        mu_c[idx_u] = mu_u[idx_u]
        
        t_ = t.unsqueeze(-1).unsqueeze(-1)
        x0 = torch.randn_like(x1)
        x = (1 - (1 - self.sigma) * t_) * x0 + t_ * x1
        ut = (x1 - x0 * (1 - self.sigma)) * mask
            
        vt = self.estimator(x, mask, mu_c, t, l_mask,
            language_id=language_id,
            prompt_language_id=prompt_language_id,
            synth_language_id=synth_language_id
        )
        loss = torch.sum(((vt - ut) ** 2) * l_mask) / (torch.sum(l_mask) * self.n_feats)
        return loss

    def compute_loss(self, x1, mask, mu=None, l_mask=None, 
        language_id=None, prompt_language_id=None, synth_language_id=None
    ):
        t = torch.rand(x1.shape[0], dtype=x1.dtype, device=x1.device, requires_grad=False)
        return self.loss_t(x1, mask, mu=mu, t=t, l_mask=l_mask,
            language_id=language_id,
            prompt_language_id=prompt_language_id,
            synth_language_id=synth_language_id
        )
    def process_input(self, x, mask, mu=None, l_mask=None):
        B, C, T = x.size()
        x_proj = self.proj_x(x.transpose(1, 2)).transpose(1, 2)
        mu_u = (self.mask_token.unsqueeze(-1).repeat(B, 1, T) * mask).to(x)
        if mu is None:
            mu = mu_u
        else:
            mu = self.emb(mu) * math.sqrt(self.c_dim)
            mu = torch.transpose(mu, 1, -1) * mask
        
        mu = self.proj_mu(mu.transpose(1, 2)).transpose(1, 2) * mask
        mu_u = self.proj_mu(mu_u.transpose(1, 2)).transpose(1, 2) * mask
        
        mu_u = mu_u * mask
        mu = mu * mask + x_proj * (mask - l_mask)
        return mu, mu_u
