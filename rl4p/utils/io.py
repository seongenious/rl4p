import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict

from envs.datatypes import State, Action, Transition
from utils.replay_buffer import ReplayBuffer


def save_transitions(transitions: List[Transition], path: str):
    """Save a list of transitions to disk as a .npz file.

    Each transition includes state, action, next_state, reward, and done.
    The saved file contains arrays:
        - obs: shape [N, 5] (x, y, yaw, v, dir)
        - actions: shape [N, 2] (delta, accel)
        - next_obs: shape [N, 5]
        - rewards: shape [N]
        - dones: shape [N]

    Args:
        transitions: List of Transition objects to save.
        path: Destination file path for saving the .npz archive.
    """
    # Convert list of transitions to Transition of batched arrays
    stacked = Transition(
        obs=jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[t.obs for t in transitions]),
        action=jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[t.action for t in transitions]),
        next_obs=jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[t.next_obs for t in transitions]),
        reward=jnp.stack([t.reward for t in transitions]),
        done=jnp.stack([t.done for t in transitions]),
    )

    # Use vmap to vectorize conversion to arrays
    def state_to_array(state: State):
        return jnp.array([state.x, state.y, state.yaw, state.v, state.dir])

    def action_to_array(action: Action):
        return jnp.array([action.delta, action.accel])

    obs = jax.vmap(state_to_array)(stacked.obs)
    next_obs = jax.vmap(state_to_array)(stacked.next_obs)
    actions = jax.vmap(action_to_array)(stacked.action)
    rewards = stacked.reward
    dones = stacked.done

    np.savez(path, obs=obs, actions=actions, next_obs=next_obs, rewards=rewards, dones=dones)


def load_transitions(path: str) -> Dict[str, jnp.ndarray]:
    """Load transitions from a .npz file into a JAX-compatible dictionary.

    Keys returned in the dictionary:
        - 'obs': jnp.ndarray of shape [N, 5]
        - 'actions': jnp.ndarray of shape [N, 2]
        - 'next_obs': jnp.ndarray of shape [N, 5]
        - 'rewards': jnp.ndarray of shape [N]
        - 'dones': jnp.ndarray of shape [N]

    Args:
        path: File path to a .npz archive containing saved transitions.

    Returns:
        A dictionary mapping keys to JAX arrays suitable for training.

    Example:
        data = load_transitions("offline_data.npz")
        print(data['obs'].shape)  # (N, 5)
    """
    data = np.load(path)
    return {k: jnp.array(data[k]) for k in data}


def load_transitions_into_buffer(
        folder_path: str, max_size: int, obs_dim: int, act_dim: int) -> ReplayBuffer:
    """Load all .npz transitions in a folder into a replay buffer.

    Args:
        folder_path: Path to the folder containing .npz transition files.
        max_size: Maximum size of the replay buffer.
        obs_dim: Observation dimension.
        act_dim: Action dimension.

    Returns:
        ReplayBuffer object filled with loaded transitions.
    """
    buffer = ReplayBuffer(max_size=max_size, obs_dim=obs_dim, act_dim=act_dim)
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npz"))
    print(f"Found {len(files)} transition files.")
    
    for f in files:
        file_path = os.path.join(folder_path, f)
        data = load_transitions(file_path)
        buffer.add_batch(data)

    print(f"Replay buffer populated with {buffer.size} transitions.")
    return buffer