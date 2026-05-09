import torch
import transformers
from typing import List
from transformers import T5EncoderModel, T5Config, AutoTokenizer
from einops import rearrange
transformers.logging.set_verbosity_error()


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class T5Conditioner(torch.nn.Module):
    """T5-based TextConditioner.
    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        device (str): Device for T5 Conditioner.
    """

    def __init__(self, name, n_hidden, max_length=None):
        super(T5Conditioner, self).__init__()
        model, t5_dim = self.get_model(name)
        self.model = model
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.proj = torch.nn.Conv1d(t5_dim, n_hidden, 1)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def get_model(self, name):
        t5_dim = T5Config.from_pretrained(name).d_model
        model = T5EncoderModel.from_pretrained(name)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model, t5_dim

    def tokenize_and_encode(self, texts: List[str], device: str,
                            attn_mask=None, pad_id=None):
        # tokenize first
        tokenized = self.tokenizer.batch_encode_plus(
            texts, return_tensors="pt", padding='longest',
            max_length=self.max_length, truncation=True)

        token_ids = tokenized.input_ids.to(device)
        attn_mask = tokenized.attention_mask.to(device)

        attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())
        attn_mask = attn_mask.bool()

        # encode with t5
        output = self.model(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()

        # force padded embeddings to 0
        encoded_text = encoded_text.masked_fill(
            ~rearrange(attn_mask, '... -> ... 1'), 0.)

        return encoded_text, attn_mask

    def forward(self, text, device):
        mask_empty = [True if t_i == '' else False for t_i in text]
        with torch.no_grad():
            h, mask = self.tokenize_and_encode(text, device)
        h = h.transpose(1, 2).to(device)
        h[mask_empty] *= 0
        mask = mask.unsqueeze(1).long().to(device)
        h = self.proj(h) * mask
        return h, mask
