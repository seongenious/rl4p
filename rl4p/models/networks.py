from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Callable, Tuple, Optional, Dict, Any, List

from envs.datatypes import Observation

class MLP(nn.Module):
    """Multi-layer perceptron."""
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:        
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return x


class CNN(nn.Module):
    """Convolutional neural network for processing occupancy grid."""
    hidden_dims: Sequence[int] = (64, 128, 256)
    output_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (batch_size, 256, 256)
        if x.ndim == 2:
            x = x[None, ...]  # (1, 256, 256)
        if x.ndim == 3:
            x = x[..., None]  # (batch, 256, 256, 1)
        
        # Convolutional layers
        for dim in self.hidden_dims:
            x = nn.Conv(features=dim, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (batch_size, features)
        
        # Final dense layer
        x = nn.Dense(self.output_dim)(x)
        return x


class ObservationEncoder(nn.Module):
    """Encoder for processing observation dictionary."""
    vehicle_hidden_dims: Sequence[int] = (128, 256)
    grid_hidden_dims: Sequence[int] = (64, 128, 256)
    grid_output_dim: int = 256
    combined_hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        # Process vehicle state
        vehicle_state = obs.vehicle_state  # (batch_size, 5)
        if vehicle_state.ndim == 1:
            vehicle_state = vehicle_state[None, ...]  # (1, 5)
        vehicle_features = MLP(self.vehicle_hidden_dims)(vehicle_state)  # (batch_size, 256)
        
        # Process occupancy grid
        occupancy_grid = obs.occupancy_grid  # (batch_size, 256, 256)
        if occupancy_grid.ndim == 2:
            occupancy_grid = occupancy_grid[None, ...]  # (1, 256, 256)
        grid_features = CNN(self.grid_hidden_dims, self.grid_output_dim)(occupancy_grid)  # (batch_size, 256)
        
        # Combine features
        combined = jnp.concatenate([vehicle_features, grid_features], axis=-1)
        encoded = MLP(self.combined_hidden_dims)(combined)  # (batch_size, 256)
        
        return encoded


class PolicyNetwork(nn.Module):
    """Gaussian policy network that outputs mean and log_std."""
    action_dim: int = 2  # steering_angle, acceleration
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: Observation) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Encode observation
        x = ObservationEncoder()(obs)
        
        # Policy head
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        
        return mean, log_std


class QNetwork(nn.Module):
    """Q-function that outputs a scalar value for (obs, action)."""
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: Observation, action: jnp.ndarray) -> jnp.ndarray:
        # Encode observation
        x = ObservationEncoder()(obs)
        
        # Concatenate with action
        x = jnp.concatenate([x, action], axis=-1)
        
        # Q-function head
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


def sample_action(actor: PolicyNetwork, 
                  params: Dict, 
                  obs: Observation,
                  rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """Perform policy inference to get actions. Dependent on whether rng is provided, 
    the action is deterministic or stochastic.
    
    Args:
        actor: Actor network definition.
        params: Actor network parameters.
        obs: Observation.
        rng: Random number generator key.

    Returns:
        Actions, shape: (batch_size, action_dim).
    """
    mu, log_std = actor.apply(params, obs)
    
    if rng is None:
        # Deterministic action
        return jnp.tanh(mu)
    else:
        # Stochastic action
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, shape=mu.shape)
        return jnp.tanh(mu + noise * std)