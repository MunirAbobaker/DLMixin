import torch 
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math



@dataclass 
class config:
    n_embd = 768
    n_head = 12
    n_layer = 12
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    afn = 'gelu'
    block_size = 1024 
    vocab_size = 50257 


# multihead self attention
class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        assert self.n_embd % self.n_head == 0
        self.head_size = self.n_embd // self.n_head

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.bias = self.register_buffer(
            'bias', torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)
        k = k.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)
        v = v.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)

        y = y = F.scaled_dot_product_attention(q, k, v,  is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)    
        return y



