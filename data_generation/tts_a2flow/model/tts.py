import math

import torch
import torch.nn as nn

from torch.cuda import amp
from model.base import BaseModule
from model.duration_predictor import DurationPredictor_Mel
from model.flow_matching import FlowMatching
from model.utils import sequence_mask, get_mae_mask


class A2Flow(BaseModule):
    def __init__(self, n_vocab, n_feats, n_language=1, r_min_dec=0.7, r_max_dec=1.0, p_drop_dec=0.1, 
                 mode=2, n_unit=2000, unit_prob=0., num_heads=16, n_dec_layers=24, dec_hidden_size=1024, **kwargs): 
        super(A2Flow, self).__init__()
        self.n_vocab = n_vocab
        self.n_unit = n_unit
        self.n_token = self.n_vocab + self.n_unit + 1
        self.n_feats = n_feats
        self.mode = mode
        
        self.frac_lengths_mask_dec = (r_min_dec, r_max_dec)
        self.p_drop_dec = p_drop_dec
        self.unit_prob = unit_prob

        self.decoder = FlowMatching(
            self.n_token, n_feats, c_dim=512, hidden_size=dec_hidden_size, 
            num_heads=num_heads, num_layers=n_dec_layers, n_language=n_language
        )
        
    @torch.no_grad()
    def infer(self, x=None, x_lengths=None, p=None, p_lengths=None, y_lengths=None, n_timesteps=32, gradient_scale=0., 
              language_id=None, alpha=1., prompt_language_id=None, synth_language_id=None):
        p_mask = sequence_mask(p_lengths, p.shape[-1]).unsqueeze(1).to(p)
        x, x_lengths = self.relocate_input([x, x_lengths])
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(p)
        y_mask = torch.cat((p_mask, y_mask), dim=-1)
        
        x0 = torch.randn((y_mask.size(0), self.n_feats, y_mask.size(-1))).to(p) * y_mask
        mu_y = (y_mask.squeeze(1) * (self.n_vocab + self.n_unit)).long()
        mu_y[:, :x.size(-1)] = x
        
        x1, x1_cut = self.decoder(
            x0, y_mask, mu_y, n_timesteps=n_timesteps, gradient_scale=gradient_scale, 
            p=p, p_mask=p_mask, language_id=language_id, alpha=alpha,
            prompt_language_id=prompt_language_id, synth_language_id=synth_language_id
        )
        return x1, x1_cut
    @torch.no_grad()
    def infer_batch(self, x=None, x_lengths=None, p=None, p_lengths=None, y_lengths=None,
          n_timesteps=32, gradient_scale=0., language_id=None,
          alpha=1., prompt_language_id=None, synth_language_id=None):
        # Create mask for prompt mel (p) of shape [B, 1, Tp]
        p_mask = sequence_mask(p_lengths, p_lengths.max()).unsqueeze(1).to(p)  # [B, 1, Tp]

        # Relocate inputs to model device
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Create mask for target length (y) of shape [B, 1, Ty]
        y_mask = sequence_mask(y_lengths, y_lengths.max()).unsqueeze(1).to(p)  # [B, 1, Ty]

        # Concatenate p_mask and y_mask → shape: [B, 1, Tp + Ty]
        full_mask = torch.cat((p_mask, y_mask), dim=-1)  # [B, 1, T_total]

        # Sample x0 from normal distribution and apply full mask
        B, _, T_total = full_mask.shape
        x0 = torch.randn((B, self.n_feats, T_total), device=p.device) * full_mask  # [B, n_feats, T_total]

        # Prepare mu_y by filling with vocab offset, then inserting x
        mu_y = (full_mask.squeeze(1) * (self.n_vocab + self.n_unit)).long()  # [B, T_total]
        # mu_y[:, :x.size(-1)] = x
        for i in range(len(x)):
            mu_y[i, :x_lengths[i]] = x[i]

        # Decode
        x1, x1_cut = self.decoder(
            x0, full_mask, mu_y, n_timesteps=n_timesteps, gradient_scale=gradient_scale,
            p=p, p_mask=p_mask, language_id=language_id, alpha=alpha,
            prompt_language_id=prompt_language_id, synth_language_id=synth_language_id
        )

        return x1, x1_cut

    def forward(self, x=None, x_lengths=None, y=None, y_lengths=None, language_id=None,
                prompt_language_id=None, synth_language_id=None):
        # mode == 2  -> mu_y = concat (x, filler_tokens)
        
        # y is required for all modes.
        y, y_lengths = self.relocate_input([y, y_lengths])
        # y_mask [B, 1, T]
        y_mask = sequence_mask(y_lengths, y.shape[-1]).unsqueeze(1).to(y)
        x, x_lengths = self.relocate_input([x, x_lengths])
        loss_dict = {}
        
        mu_y[:, :x.size(-1)] = x

        # l_mask = [1 1 1 0 0 0 1 1 1 0 0 0]
        l_mask = get_mae_mask(y_mask, frac_lengths_mask=self.frac_lengths_mask_dec, p_drop=self.p_drop_dec)      
        fm_loss = self.decoder.compute_loss(y, y_mask, mu_y, y_mask - l_mask, prompt_language_id=prompt_language_id, synth_language_id=synth_language_id)
        loss_dict['fm_loss'] = fm_loss
        return loss_dict


class DP(BaseModule):
    def __init__(self, n_vocab, n_feats=80, n_language=1, n_unit=200, num_heads=8, n_dp_layers=8, dp_hidden_size=512, scale_factor=6, **kwargs):
        super(DP, self).__init__()
        self.n_feats = n_feats
        self.n_vocab = n_vocab
        self.n_unit = n_unit
        self.n_token = self.n_vocab + self.n_unit + 1
        self.scale_factor = scale_factor

        self.dp = DurationPredictor_Mel(self.n_token, n_feats=n_feats, n_channels=dp_hidden_size, 
                                        n_layers=n_dp_layers, n_heads=8, n_language=n_language)
    
    def forward(self, x=None, x_lengths=None, p=None, p_lengths=None, y_lengths=None, language_id=None):
        logw = self.dp(
            x, x_lengths, 
            p, p_lengths, 
            language_id=language_id, 
        ).squeeze(-1)
        w = (torch.exp(logw * self.scale_factor)).floor()
        
        loss_dict = {}
        if y_lengths is not None:
            logw_ = torch.log(y_lengths) / self.scale_factor
            dur_loss = torch.mean((logw - logw_)**2)
            loss_dict['duration_loss'] = dur_loss
            return loss_dict, w.long()
        else:
            return w.long()
