from flax import linen as nn
import jax
import jax.numpy as jnp
import distrax
from typing import  Tuple, Optional, Dict
from flax.training import train_state

from model.encoder import SacEncoder


class PolicyNetwork(nn.Module):
    """Gaussian policy network that outputs mean and log_std."""
    action_dim: int = 2  # steering_angle, acceleration
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ x: (B, 256) """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        mu = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
    
        return mu, log_std

class QNetwork(nn.Module):
    """Q-function that outputs a scalar value for (obs, action)."""
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, feat: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """ x: (B, 1, 256) """
        x = jnp.concatenate([feat, action], axis=-1)  # (B, 256+2)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)  # (B,)
    
class TwinQNetwork(nn.Module):
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, feat: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = QNetwork(hidden_dims=self.hidden_dims)(feat, action)
        q2 = QNetwork(hidden_dims=self.hidden_dims)(feat, action)
        return q1, q2  # both shape: (B,)

def sample_action(actor: train_state.TrainState, 
                  params: Dict, 
                  feat: jnp.ndarray,
                  rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Perform policy inference to get actions. Dependent on whether rng is provided, 
    the action is deterministic or stochastic.
    
    Args:
        actor: Actor network definition.
        params: Actor network parameters.
        feat: Feature.
        rng: Random number generator key.

    Returns:
        action: Actions, shape: (batch_size, action_dim).
        log_prob: Log probabilities, shape: (batch_size, 1).
    """
    mu, log_std = actor.apply_fn({'params': params}, feat)
    action, log_prob = [0, 0], jnp.zeros_like(mu).sum(axis=-1)
        
    if rng is None:
        action = jnp.tanh(mu)
    else:
        std = jnp.exp(log_std)
        dist = distrax.Transformed(distrax.Normal(mu, std), distrax.Tanh())
        action = dist.sample(seed=rng)
        log_prob = dist.log_prob(action).sum(axis=-1)

    return action, log_prob