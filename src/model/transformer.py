import torch
import torch.nn as nn
from einops import rearrange
from typing import Callable

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float=0.):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(), # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int=8, head_dim: int=64, dropout: float=0.):
        super().__init__()

        inner_dim = head_dim * num_heads

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        depth: int=6, 
        num_heads: int=8, 
        head_dim: int=64, 
        mlp_hidden_dim: int=128, 
        dropout: float=0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, SelfAttention(embed_dim, num_heads, head_dim, dropout)),
                PreNorm(embed_dim, FeedForward(embed_dim, mlp_hidden_dim, dropout)),
            ]))

    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        depth: int=6, 
        num_heads: int=8, 
        head_dim: int=64, 
        mlp_hidden_dim: int=128, 
        n_features: int=1,
        hidden_dim: int=128,
        output_dim: int=2,
        dropout: float=0.
    ):
        super().__init__()

        self.encoder = TransformerEncoderLayer(embed_dim, depth, num_heads, head_dim, mlp_hidden_dim, dropout)
        self.output = nn.Sequential(
            nn.Linear(n_features * embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.view_embed = nn.Parameter(torch.zeros(1, n_features, embed_dim))

    def forward(self, x):
        # x = x + self.view_embed
        x = self.encoder(x)
        x = rearrange(x, 'b n d -> b (n d)')
        return self.output(x)