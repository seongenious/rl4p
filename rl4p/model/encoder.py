import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple


class OccupancyEncoder(nn.Module):
    """CNN backbone for (B, W, H, C) input."""
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # x: (B, W, H, C)
        B, W, H, C = x.shape
        
        # 2D CNN for feature extraction
        x = nn.Conv(32, (3, 3), strides=1, padding='SAME')(x)  # (B, W, H, 32)
        x = nn.gelu(x)
        x = nn.Conv(64, (3, 3), strides=2, padding='SAME')(x)  # (B, W/2, H/2, 64)
        x = nn.gelu(x)
        x = nn.Conv(128, (3, 3), strides=2, padding='SAME')(x)  # (B, W/4, H/4, 128)
        x = nn.gelu(x)
        x = nn.Conv(256, (3, 3), strides=2, padding='SAME')(x)  # (B, W/8, H/8, 256)
        x = nn.gelu(x)
        x = nn.Conv(256, (3, 3), strides=2, padding='SAME')(x)  # (B, W/16, H/16, 256)
        x = nn.gelu(x)
        return x  # (B, W/16, H/16, 256)

class StateEncoder(nn.Module):
    """Encode vehicle state into a single feature vector."""

    @nn.compact
    def __call__(self, x):  # x: (B, 4)
        x = jnp.expand_dims(x, axis=1)  # (B, 1, 4)
        x = jnp.expand_dims(x, axis=2)  # (B, 1, 1, 4)
        x = jnp.broadcast_to(x, (x.shape[0], 16, 16, 4))  # (B, 16, 16, 4)
        return x  # (B, 16, 16, 4)

class SacEncoder(nn.Module):
    """Encode occupancy grid and vehicle state into a single feature vector."""
    feat_dim: int = 256

    @nn.compact
    def __call__(self, occupancy_grid: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        # occ_grid: (B, 256, 256, 3), state: (B, 4)
        occupancy_feat = OccupancyEncoder()(occupancy_grid)  # (B, 16, 16, 256)
        kinematic_feat = StateEncoder()(state)  # (B, 16, 16, 4)
        feat = jnp.concatenate([occupancy_feat, kinematic_feat], axis=-1)  # (B, 16, 16, 260)
        
        # Fuse via 1x1 Conv
        feat = nn.Conv(self.feat_dim, kernel_size=(1, 1))(feat)  # (B, 16, 16, 256)
        feat = nn.gelu(feat)

        return feat  # (B, 16, 16, 256)