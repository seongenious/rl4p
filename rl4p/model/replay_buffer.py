from typing import Optional, Dict, Any, Sequence, NamedTuple

import chex
import jax
import jax.numpy as jnp

from env.datatypes import Observation

BUFFER_SIZE = 1000
BATCH_SIZE = 64

@chex.dataclass
class ReplayBuffer:
    """Replay buffer for storing and sampling experience."""
    obs: Observation       # (buffer size, obs dim)
    action: jnp.ndarray    # (buffer size, act dim)
    reward: jnp.ndarray    # (buffer size,)
    done: jnp.ndarray      # (buffer size,)
    next_obs: Observation  # (buffer size, obs dim)
    capacity: int          # total buffer size
    size: int              # current size
    position: int          # current position

@jax.jit
def create_buffer() -> ReplayBuffer:
    """Create a replay buffer. 

    Returns:
        ReplayBuffer: replay buffer
    """    
    return ReplayBuffer(
        obs=Observation(
            occupancy_grid=jnp.zeros((BUFFER_SIZE, 3, 256, 256, 2), dtype=jnp.int8),
            state=jnp.zeros((BUFFER_SIZE, 3, 5)),
        ),
        action=jnp.zeros((BUFFER_SIZE, 2)),
        reward=jnp.zeros((BUFFER_SIZE,)),
        done=jnp.zeros((BUFFER_SIZE,)),
        next_obs=Observation(
            occupancy_grid=jnp.zeros((BUFFER_SIZE, 3, 256, 256, 2), dtype=jnp.int8),
            state=jnp.zeros((BUFFER_SIZE, 3, 5)),
        ),
        capacity=BUFFER_SIZE,
        size=0,
        position=0,
    )

@jax.jit
def add_buffer(buffer: ReplayBuffer, 
               obs: Observation, 
               action: jnp.ndarray, 
               reward: float, 
               done: bool, 
               next_obs: Observation) -> ReplayBuffer:
    """Add a transition to the replay buffer.

    Args:
        buffer: replay buffer
        obs: observation
        action: action
        reward: reward
        done: done
        next_obs: next observation

    Returns:
        ReplayBuffer: updated replay buffer
    """
    idx = buffer.position

    # Update observation
    new_obs = buffer.obs.replace(
        occupancy_grid=buffer.obs.occupancy_grid.at[idx].set(obs.occupancy_grid),
        state=buffer.obs.state.at[idx].set(obs.state),
    )

    # Update next observation
    new_next_obs = buffer.next_obs.replace(
        occupancy_grid=buffer.next_obs.occupancy_grid.at[idx].set(next_obs.occupancy_grid),
        state=buffer.next_obs.state.at[idx].set(next_obs.state),
    )

    # Update buffer
    buffer = buffer.replace(
        obs=new_obs,
        action=buffer.action.at[idx].set(action),
        reward=buffer.reward.at[idx].set(reward),
        done=buffer.done.at[idx].set(done),
        next_obs=new_next_obs,
        position=(buffer.position + 1) % buffer.capacity,
        size=jnp.minimum(buffer.size + 1, buffer.capacity),
    )

    return buffer

@jax.jit
def sample_batch_transitions(buffer: ReplayBuffer, rng: jax.random.PRNGKey):
    """Sample a batch of transitions from the replay buffer.

    Args:
        buffer: replay buffer
        rng: random number generator

    Returns:
        dict: batch of transitions
    """
    idxs = jax.random.randint(rng, (BATCH_SIZE,), 0, buffer.size)

    return dict(
        obs=Observation(
            occupancy_grid=buffer.obs.occupancy_grid[idxs],
            state=buffer.obs.state[idxs],
        ),
        action=buffer.action[idxs],
        reward=buffer.reward[idxs],
        done=buffer.done[idxs],
        next_obs=Observation(
            occupancy_grid=buffer.next_obs.occupancy_grid[idxs],
            state=buffer.next_obs.state[idxs],
        ),
    )    
    
    