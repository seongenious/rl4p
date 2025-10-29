import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()