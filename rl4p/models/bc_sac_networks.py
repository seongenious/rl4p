"""BC-SAC network architectures for RL4P project.

This module defines the network architectures for Behavior Cloned Soft Actor-Critic
(BC-SAC) algorithm, following the project specification where inputs include
initial state, goal state, and occupancy grid map.
"""

from typing import Tuple, Sequence, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn


class StateEncoder(nn.Module):
    """Encoder for vehicle state information.
    
    Encodes initial state and goal state into a latent representation.
    """
    hidden_dims: Sequence[int] = (128, 128)
    
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Encode vehicle state.
        
        Args:
            state: Vehicle state [x, y, yaw, v, dir], shape: (batch_size, 5).
            
        Returns:
            Encoded state representation, shape: (batch_size, hidden_dims[-1]).
        """
        x = state
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        return x


class OccupancyGridEncoder(nn.Module):
    """Encoder for occupancy grid map.
    
    Uses convolutional layers to encode 2D occupancy grid information.
    """
    conv_dims: Sequence[int] = (32, 64, 128)
    fc_dims: Sequence[int] = (256, 128)
    grid_size: int = 64  # 64x64 grid
    
    @nn.compact
    def __call__(self, grid: jnp.ndarray) -> jnp.ndarray:
        """Encode occupancy grid map.
        
        Args:
            grid: Occupancy grid map, shape: (batch_size, grid_size, grid_size, 1).
            
        Returns:
            Encoded grid representation, shape: (batch_size, fc_dims[-1]).
        """
        x = grid
        
        # Convolutional layers
        for dim in self.conv_dims:
            x = nn.relu(nn.Conv(dim, kernel_size=(3, 3), padding='SAME')(x))
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten and fully connected layers
        x = x.reshape(x.shape[0], -1)  # Flatten
        for dim in self.fc_dims:
            x = nn.relu(nn.Dense(dim)(x))
        
        return x


class BC_SACPolicyNetwork(nn.Module):
    """Policy network for BC-SAC with proper input structure.
    
    Takes initial state, goal state, and occupancy grid as input.
    Outputs Gaussian policy parameters.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(self, 
                 initial_state: jnp.ndarray,
                 goal_state: jnp.ndarray,
                 occupancy_grid: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of the policy network.
        
        Args:
            initial_state: Initial vehicle state [x, y, yaw, v, dir], 
                          shape: (batch_size, 5).
            goal_state: Goal vehicle state [x, y, yaw, v, dir], 
                       shape: (batch_size, 5).
            occupancy_grid: Occupancy grid map, 
                           shape: (batch_size, grid_size, grid_size, 1).
            
        Returns:
            Tuple of (mean, std) for action distribution, 
            each of shape: (batch_size, action_dim).
        """
        # Encode states and grid
        initial_encoded = StateEncoder()(initial_state)
        goal_encoded = StateEncoder()(goal_state)
        grid_encoded = OccupancyGridEncoder()(occupancy_grid)
        
        # Concatenate all encodings
        x = jnp.concatenate([initial_encoded, goal_encoded, grid_encoded], axis=-1)
        
        # Policy head
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        
        # Output mean and log_std
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        std = jnp.exp(log_std)
        
        return mean, std


class BC_SACQNetwork(nn.Module):
    """Q-function network for BC-SAC with proper input structure.
    
    Takes initial state, goal state, occupancy grid, and action as input.
    """
    hidden_dims: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(self,
                 initial_state: jnp.ndarray,
                 goal_state: jnp.ndarray,
                 occupancy_grid: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the Q-network.
        
        Args:
            initial_state: Initial vehicle state [x, y, yaw, v, dir], 
                          shape: (batch_size, 5).
            goal_state: Goal vehicle state [x, y, yaw, v, dir], 
                       shape: (batch_size, 5).
            occupancy_grid: Occupancy grid map, 
                           shape: (batch_size, grid_size, grid_size, 1).
            action: Action [delta, accel], shape: (batch_size, action_dim).
            
        Returns:
            Q-values, shape: (batch_size,).
        """
        # Encode states and grid
        initial_encoded = StateEncoder()(initial_state)
        goal_encoded = StateEncoder()(goal_state)
        grid_encoded = OccupancyGridEncoder()(occupancy_grid)
        
        # Concatenate all encodings and action
        x = jnp.concatenate([initial_encoded, goal_encoded, grid_encoded, action], axis=-1)
        
        # Q-function head
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        
        # Output Q-value
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, axis=-1)


class BC_SACValueNetwork(nn.Module):
    """Value function network for BC-SAC (optional, for critic baseline).
    
    Takes initial state, goal state, and occupancy grid as input.
    """
    hidden_dims: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(self,
                 initial_state: jnp.ndarray,
                 goal_state: jnp.ndarray,
                 occupancy_grid: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the value network.
        
        Args:
            initial_state: Initial vehicle state [x, y, yaw, v, dir], 
                          shape: (batch_size, 5).
            goal_state: Goal vehicle state [x, y, yaw, v, dir], 
                       shape: (batch_size, 5).
            occupancy_grid: Occupancy grid map, 
                           shape: (batch_size, grid_size, grid_size, 1).
            
        Returns:
            Value estimates, shape: (batch_size,).
        """
        # Encode states and grid
        initial_encoded = StateEncoder()(initial_state)
        goal_encoded = StateEncoder()(goal_state)
        grid_encoded = OccupancyGridEncoder()(occupancy_grid)
        
        # Concatenate all encodings
        x = jnp.concatenate([initial_encoded, goal_encoded, grid_encoded], axis=-1)
        
        # Value function head
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        
        # Output value
        v = nn.Dense(1)(x)
        return jnp.squeeze(v, axis=-1) 