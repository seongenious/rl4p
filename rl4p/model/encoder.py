import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple


class OccupancyEncoder(nn.Module):
    """CNN backbone for (B, T, H, W, C) input."""
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # x: (B, T, H, W, C)
        if x.ndim == 4:
            x = x[None, ...]
            
        B, T, H, W, C = x.shape
        
        # Reshape for 3D convolution: (B, T, H, W, C) -> (B, H, W, T*C), early fusion
        x = x.reshape((B, H, W, T*C))  # (B, 256, 256, T*2)
        
        # 2D CNN with temporal channels
        x = nn.Conv(32, (5, 5), strides=1, padding='SAME')(x)  # (B, 256, 256, 32)
        x = nn.gelu(x)
        x = nn.Conv(64, (3, 3), strides=2, padding='SAME')(x)  # (B, 128, 128, 64)
        x = nn.gelu(x)
        x = nn.Conv(128, (3, 3), strides=2, padding='SAME')(x)  # (B, 64, 64, 128)
        x = nn.gelu(x)
        x = nn.Conv(self.hidden_dim, (3, 3), strides=2, padding='SAME')(x)  # (B, 32, 32, hidden_dim)
        x = nn.gelu(x)
        x = nn.Conv(self.hidden_dim, (3, 3), strides=2, padding='SAME')(x)  # (B, 16, 16, hidden_dim)
        x = nn.gelu(x)
        
        # Reshape to grid tokens
        x = x.reshape((B, 256, self.hidden_dim))  # (B, 256, hidden_dim)
        
        return x  # (B, 256, hidden_dim)


class StateEncoder(nn.Module):
    """Encode temporal vehicle state sequence into a single feature vector."""
    hidden_dim: int = 64
    output_dim: int = 128
    
    POSITION_SCALE = 10.0
    YAW_SCALE = jnp.pi
    VELOCITY_SCALE = 10.0

    @nn.compact
    def __call__(self, x):  # x: (B, T, 5)
        if x.ndim == 2:
            x = x[None, ...]
        
        # Input normalization for state variables
        # x, y: normalize by a reasonable range (e.g., ±10m)
        # yaw: already in [-π, π], normalize by π
        # v: normalize by max speed (e.g., 10 m/s)
        # dir: already in [-1, 1]
        x_normalized = x.at[:, :, 0].set(x[:, :, 0] / self.POSITION_SCALE)  # x
        x_normalized = x_normalized.at[:, :, 1].set(x[:, :, 1] / self.POSITION_SCALE)  # y
        x_normalized = x_normalized.at[:, :, 2].set(x[:, :, 2] / self.YAW_SCALE)  # yaw
        x_normalized = x_normalized.at[:, :, 3].set(x[:, :, 3] / self.VELOCITY_SCALE)  # v
        
        # 1D CNN over T
        x = nn.Conv(
            features=self.hidden_dim, kernel_size=(3,), strides=(1,), padding='SAME')(x_normalized)  # (B, T, hidden)
        x = nn.gelu(x)  # Changed from ReLU to GELU
        x = nn.Conv(
            features=self.output_dim, kernel_size=(1,), strides=(1,), padding='SAME')(x)  # (B, T, output)
        
        # Temporal pooling
        x = jnp.mean(x, axis=1)  
        x = x[:, None, :]  # Unsqueeze to (B, 1, 128)
        return x  # (B, 1, 128)

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
            x = nn.gelu(x)  # Changed from ReLU to GELU
            x = nn.Dense(self.embed_dim)(x)
            tokens = tokens + x
        return tokens

class SacEncoder(nn.Module):
    feat_dim: int = 128

    @nn.compact
    def __call__(self, occ_grid: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        # occ_grid: (B, 3, 256, 256, 2), state: (B, 3, 5)
        grid_tokens = OccupancyEncoder(hidden_dim=self.feat_dim)(occ_grid)  # (B, 256, D)
        state_token = StateEncoder(output_dim=self.feat_dim)(state)  # (B, 1, D)
        tokens = jnp.concatenate([state_token, grid_tokens], axis=1)  # (B, 257, D)
        tokens = TransformerEncoder(embed_dim=self.feat_dim)(tokens)  # (B, 257, D)

        # Feature aggregation
        state_out = tokens[:, 0]  # (B, D)
        grid_mean = jnp.mean(tokens[:, 1:], axis=1)  # (B, D)
        feat = jnp.concatenate([state_out, grid_mean], axis=-1)  # (B, 2*D = 256)
        
        return feat  # (B, 256)