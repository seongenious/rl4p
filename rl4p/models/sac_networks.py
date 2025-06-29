from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Callable, Tuple


class PolicyNetwork(nn.Module):
    """Gaussian policy network that outputs mean and log_std."""
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mean, log_std


class QNetwork(nn.Module):
    """Q-function that outputs a scalar value for (obs, action)."""
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)
