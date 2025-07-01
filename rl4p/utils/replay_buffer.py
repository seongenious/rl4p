from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    """Efficient JAX-compatible replay buffer with dictionary-based storage.

    Stores transition data in a dictionary format for better flexibility and efficiency.
    Supports both vehicle state only and full observation (vehicle state + occupancy grid).
    """

    def __init__(self, max_size: int, obs_dim: int = 5, act_dim: int = 2, 
                 include_occupancy_grid: bool = False, include_expert: bool = False):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of transitions to store.
            obs_dim: Dimension of the vehicle state (default: 5 for x, y, yaw, v, dir).
            act_dim: Dimension of the action vector (default: 2 for delta, accel).
            include_occupancy_grid: Whether to store occupancy grid data.
            include_expert: Whether to store expert action data.
        """
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.include_occupancy_grid = include_occupancy_grid
        self.include_expert = include_expert
        
        self.ptr = 0
        self.size = 0

        # Core transition data
        self.data = {
            'obs_vehicle_state': jnp.zeros((max_size, obs_dim)),
            'action': jnp.zeros((max_size, act_dim)),
            'next_obs_vehicle_state': jnp.zeros((max_size, obs_dim)),
            'reward': jnp.zeros((max_size,)),
            'done': jnp.zeros((max_size,), dtype=jnp.bool_),
            'truncated': jnp.zeros((max_size,), dtype=jnp.bool_)
        }
        
        # Optional data
        if include_occupancy_grid:
            self.data['obs_occupancy_grid'] = jnp.zeros((max_size, 256, 256))
            self.data['next_obs_occupancy_grid'] = jnp.zeros((max_size, 256, 256))
        
        if include_expert:
            self.data['expert_action'] = jnp.zeros((max_size, act_dim))

    def add_batch(self, data: Dict[str, jnp.ndarray]):
        """Add a batch of transitions to the buffer.

        Args:
            data: A dictionary with transition data. Required keys:
                - 'obs_vehicle_state': [N, obs_dim]
                - 'action': [N, act_dim]
                - 'next_obs_vehicle_state': [N, obs_dim]
                - 'reward': [N]
                - 'done': [N]
                - 'truncated': [N]
                
                Optional keys (if buffer supports them):
                - 'obs_occupancy_grid': [N, 256, 256]
                - 'next_obs_occupancy_grid': [N, 256, 256]
                - 'expert_action': [N, act_dim]
        """
        n = data['obs_vehicle_state'].shape[0]
        
        # Calculate indices for circular buffer
        indices = np.arange(self.ptr, self.ptr + n) % self.max_size
        
        # Add data for each key
        for key in self.data.keys():
            if key in data:
                # Handle case where we wrap around the buffer
                if self.ptr + n > self.max_size:
                    # Split the data
                    first_part = self.max_size - self.ptr
                    self.data[key] = self.data[key].at[self.ptr:].set(data[key][:first_part])
                    self.data[key] = self.data[key].at[:n-first_part].set(data[key][first_part:])
                else:
                    self.data[key] = self.data[key].at[self.ptr:self.ptr+n].set(data[key])
        
        # Update buffer state
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def add(
        self,
        obs_vehicle_state: jnp.ndarray,
        action: jnp.ndarray,
        next_obs_vehicle_state: jnp.ndarray,
        reward: float,
        done: bool,
        truncated: bool = False,
        obs_occupancy_grid: Optional[jnp.ndarray] = None,
        next_obs_occupancy_grid: Optional[jnp.ndarray] = None,
        expert_action: Optional[jnp.ndarray] = None
    ):
        """Add a single transition to the buffer.

        Args:
            obs_vehicle_state: Vehicle state vector at time t.
            action: Action vector taken at time t.
            next_obs_vehicle_state: Vehicle state vector at time t+1.
            reward: Reward received after taking the action.
            done: Whether the episode terminated after this transition.
            truncated: Whether the episode was truncated.
            obs_occupancy_grid: Occupancy grid at time t (optional).
            next_obs_occupancy_grid: Occupancy grid at time t+1 (optional).
            expert_action: Expert action for this transition (optional).
        """
        self.data['obs_vehicle_state'] = self.data['obs_vehicle_state'].at[self.ptr].set(obs_vehicle_state)
        self.data['action'] = self.data['action'].at[self.ptr].set(action)
        self.data['next_obs_vehicle_state'] = self.data['next_obs_vehicle_state'].at[self.ptr].set(next_obs_vehicle_state)
        self.data['reward'] = self.data['reward'].at[self.ptr].set(reward)
        self.data['done'] = self.data['done'].at[self.ptr].set(done)
        self.data['truncated'] = self.data['truncated'].at[self.ptr].set(truncated)
        
        if self.include_occupancy_grid and obs_occupancy_grid is not None:
            self.data['obs_occupancy_grid'] = self.data['obs_occupancy_grid'].at[self.ptr].set(obs_occupancy_grid)
            self.data['next_obs_occupancy_grid'] = self.data['next_obs_occupancy_grid'].at[self.ptr].set(next_obs_occupancy_grid)
        
        if self.include_expert and expert_action is not None:
            self.data['expert_action'] = self.data['expert_action'].at[self.ptr].set(expert_action)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, rng: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.
            rng: A JAX PRNGKey used for random sampling.

        Returns:
            A dictionary with sampled transition data.
        """
        idx = jax.random.randint(rng, (batch_size,), minval=0, maxval=self.size)
        return {key: value[idx] for key, value in self.data.items()}

    def sample_vehicle_only(self, batch_size: int, rng: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Sample a batch of transitions with vehicle state only (for SAC training).

        Args:
            batch_size: Number of transitions to sample.
            rng: A JAX PRNGKey used for random sampling.

        Returns:
            A dictionary with keys: 'obs', 'actions', 'next_obs', 'rewards', 'dones'.
        """
        idx = jax.random.randint(rng, (batch_size,), minval=0, maxval=self.size)
        return {
            'obs': self.data['obs_vehicle_state'][idx],
            'actions': self.data['action'][idx],
            'next_obs': self.data['next_obs_vehicle_state'][idx],
            'rewards': self.data['reward'][idx],
            'dones': self.data['done'][idx]
        }

    def get_all_data(self) -> Dict[str, jnp.ndarray]:
        """Get all data in the buffer.

        Returns:
            A dictionary with all transition data.
        """
        return {key: value[:self.size] for key, value in self.data.items()}

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
        # Reset all arrays to zeros
        for key in self.data.keys():
            if key in ['done', 'truncated']:
                self.data[key] = jnp.zeros_like(self.data[key], dtype=jnp.bool_)
            else:
                self.data[key] = jnp.zeros_like(self.data[key])

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
