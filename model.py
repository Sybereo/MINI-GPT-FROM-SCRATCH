import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 32000
    multiple_of: int = 256  # Make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.1

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes rotary positional embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to Queries and Keys."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # SwiGLU architecture requires expanding and gating
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: (F.silu(xW1) * xW3)W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis[:seqlen])
        
        # PyTorch 2.0 Flash Attention: requires [batch, heads, seq, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=mask, 
            is_causal=True if mask is None else False
        )
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(output))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, 
            hidden_dim=4 * args.dim, 
            multiple_of=args.multiple_of,
            dropout=args.dropout
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class ProfessionalLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Weight tying (standard in production LLMs to save memory)
        self.tok_embeddings.weight = self.output.weight
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))
        
        for layer in self.layers:
            h = layer(h, self.freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            # Flatten to calculate cross entropy efficiently
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss

    def get_num_params(self, non_embedding=True):
        """Returns the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params
