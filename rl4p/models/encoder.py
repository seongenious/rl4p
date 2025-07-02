import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class CNNEncoder(nn.Module):
    """CNN encoder that downsamples occupancy grid to (16x16x128)."""
    @nn.compact
    def __call__(self, x):  # x: (B, 256, 256, 1)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)  # -> (128x128)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)  # -> (64x64)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)  # -> (32x32)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)  # -> (16x16)
        x = nn.relu(x)
        return x  # (B, 16, 16, 128)


class StateEmbedding(nn.Module):
    embed_dim: int = 128

    @nn.compact
    def __call__(self, state):  # state: (B, 5)
        x = nn.Dense(self.embed_dim)(state)
        x = nn.relu(x)
        return x[:, None, :]  # (B, 1, 128)


class TransformerBlock(nn.Module):
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256

    @nn.compact
    def __call__(self, x):  # (B, N, D)
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.embed_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)
        return x


class BEVFormerLite(nn.Module):
    num_tokens: int = 256
    embed_dim: int = 128
    num_transformer_layers: int = 3

    @nn.compact
    def __call__(self, occupancy, state):
        x = CNNEncoder()(occupancy)  # (B, 16, 16, 128)
        x = x.reshape(x.shape[0], -1, self.embed_dim)  # (B, 256, 128)

        state_token = StateEmbedding(embed_dim=self.embed_dim)(state)  # (B, 1, 128)

        tokens = jnp.concatenate([state_token, x], axis=1)  # (B, 257, 128)

        for _ in range(self.num_transformer_layers):
            tokens = TransformerBlock(embed_dim=self.embed_dim)(tokens)

        return tokens  # (B, 257, 128) with state token at index 0
