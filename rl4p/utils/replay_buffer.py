from typing import Optional, Dict, Any, Sequence
import jax
import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    """Efficient JAX-compatible replay buffer with dictionary-based storage.

    Stores transition data in a dictionary format for better flexibility and efficiency.
    Supports both vehicle state only and full observation (vehicle state + occupancy grid).
    """
    
    state_dim: int = 5
    grid_dim: Sequence[int] = (256, 256)
    action_dim: int = 2

    def __init__(self, max_size: int):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of transitions to store.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Core transition data
        self.data = {
            'obs_vehicle_state': jnp.zeros((max_size, self.state_dim), dtype=jnp.float32),
            'obs_occupancy_grid': jnp.zeros((max_size, 256, 256), dtype=jnp.float32),
            'action': jnp.zeros((max_size, self.action_dim), dtype=jnp.float32),
            'next_obs_vehicle_state': jnp.zeros((max_size, self.state_dim), dtype=jnp.float32),
            'next_obs_occupancy_grid': jnp.zeros((max_size, 256, 256), dtype=jnp.float32),
            'reward': jnp.zeros((max_size,), dtype=jnp.float32),
            'done': jnp.zeros((max_size,), dtype=jnp.bool_),
            'truncated': jnp.zeros((max_size,), dtype=jnp.bool_),
            'expert_action': jnp.zeros((max_size, self.action_dim), dtype=jnp.float32)
        }

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
        obs_occupancy_grid: jnp.ndarray,
        action: jnp.ndarray,
        next_obs_vehicle_state: jnp.ndarray,
        next_obs_occupancy_grid: jnp.ndarray,
        reward: float,
        done: bool,
        truncated: bool = False,
        expert_action: Optional[jnp.ndarray] = None
    ):
        """Add a single transition to the buffer.

        Args:
            obs_vehicle_state: Vehicle state vector at time t.
            obs_occupancy_grid: Occupancy grid at time t.
            action: Action vector taken at time t.
            next_obs_vehicle_state: Vehicle state vector at time t+1.
            next_obs_occupancy_grid: Occupancy grid at time t+1.
            reward: Reward received after taking the action.
            done: Whether the episode terminated after this transition.
            truncated: Whether the episode was truncated.
            expert_action: Expert action for this transition (optional).
        """
        self.data['obs_vehicle_state'] = self.data['obs_vehicle_state'].at[self.ptr].set(obs_vehicle_state)
        self.data['obs_occupancy_grid'] = self.data['obs_occupancy_grid'].at[self.ptr].set(obs_occupancy_grid)
        self.data['action'] = self.data['action'].at[self.ptr].set(action)
        self.data['next_obs_vehicle_state'] = self.data['next_obs_vehicle_state'].at[self.ptr].set(next_obs_vehicle_state)
        self.data['next_obs_occupancy_grid'] = self.data['next_obs_occupancy_grid'].at[self.ptr].set(next_obs_occupancy_grid)
        self.data['reward'] = self.data['reward'].at[self.ptr].set(reward)
        self.data['done'] = self.data['done'].at[self.ptr].set(done)
        self.data['truncated'] = self.data['truncated'].at[self.ptr].set(truncated)
        
        if expert_action is not None:
            self.data['expert_action'] = self.data['expert_action'].at[self.ptr].set(expert_action)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, rng: jax.random.PRNGKey) -> Dict[str, Any]:
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.
            rng: A JAX PRNGKey used for random sampling.

        Returns:
            A dictionary with keys:
                - obs: dict(vehicle_state, occupancy_grid)
                - action
                - next_obs: dict(vehicle_state, occupancy_grid)
                - reward
                - done
        """
        idx = jax.random.randint(rng, (batch_size,), minval=0, maxval=self.size)

        obs = {
            "vehicle_state": self.data["obs_vehicle_state"][idx],
            "occupancy_grid": self.data["obs_occupancy_grid"][idx]
        }

        next_obs = {
            "vehicle_state": self.data["next_obs_vehicle_state"][idx],
            "occupancy_grid": self.data["next_obs_occupancy_grid"][idx]
        }

        batch = {
            "obs": obs,
            "action": self.data["action"][idx],
            "next_obs": next_obs,
            "reward": self.data["reward"][idx],
            "done": self.data["done"][idx]
        }

        return batch


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
