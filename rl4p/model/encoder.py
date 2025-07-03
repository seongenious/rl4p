import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple


class OccupancyEncoder(nn.Module):
    """CNN backbone for (T, H, W, C) input."""
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, x) -> jnp.ndarray:  # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = x.reshape((B, H, W, T*C))  # (B, H, W, T*C)
        
        x = nn.Conv(32, (5, 5), strides=1)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(128, (3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_dim, (3, 3), strides=2)(x)  # (B*T, 32, 32, 128)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_dim, (3, 3), strides=2)(x)  # (B*T, 16, 16, 128)

        x = x.reshape((B, T, 16, 16, self.hidden_dim))  # (B, T, 16, 16, D)
        x = jnp.mean(x, axis=1)  # Temporal average (B, 16, 16, D)
        x = x.reshape((B, 256, self.hidden_dim))  # grid tokens
        return x  # (B, 256, D)

class StateEncoder(nn.Module):
    """Encode temporal vehicle state sequence into a single feature vector."""
    hidden_dim: int = 64
    output_dim: int = 128

    @nn.compact
    def __call__(self, x):  # x: (B, T, 5)
        x = nn.Conv(
            features=self.hidden_dim, kernel_size=(3,), strides=(1,), padding='SAME')(x)  # (B, T, hidden)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.output_dim, kernel_size=(1,), strides=(1,), padding='SAME')(x)  # (B, T, output)
        x = jnp.mean(x, axis=1)  # Temporal pooling
        return x  # (B, 128)

class TransformerEncoder(nn.Module):
    num_layers: int = 3
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.num_layers):
            x = nn.LayerNorm()(tokens)
            x = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                out_features=self.embed_dim
            )(x)
            tokens = tokens + x

            x = nn.LayerNorm()(tokens)
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.embed_dim)(x)
            tokens = tokens + x
        return tokens

class SacEncoder(nn.Module):
    embed_dim: int = 128

    @nn.compact
    def __call__(self, occ_grid: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        # occ_grid: (B, 3, 256, 256, 2), state: (B, 5)
        grid_tokens = OccupancyEncoder(hidden_dim=self.embed_dim)(occ_grid)  # (B, 256, D)
        state_token = StateEncoder(embed_dim=self.embed_dim)(state)  # (B, 1, D)

        tokens = jnp.concatenate([state_token, grid_tokens], axis=1)  # (B, 257, D)
        tokens = TransformerEncoder(embed_dim=self.embed_dim)(tokens)  # (B, 257, D)

        # Feature aggregation
        state_out = tokens[:, 0]  # (B, D)
        grid_mean = jnp.mean(tokens[:, 1:], axis=1)  # (B, D)
        feat = jnp.concatenate([state_out, grid_mean], axis=-1)  # (B, 2*D = 256)
        return feat  # (B, 256)