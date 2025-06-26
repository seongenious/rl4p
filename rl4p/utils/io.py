import jax.numpy as jnp
import numpy as np
from typing import List, Dict

from envs.datatypes import Transition


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

    Example:
        save_transitions(transitions, "offline_data.npz")
    """
    s = jnp.stack([jnp.array([t.obs.x, t.obs.y, t.obs.yaw, t.obs.v, t.obs.dir]) for t in transitions])
    a = jnp.stack([jnp.array([t.action.delta, t.action.accel]) for t in transitions])
    s_next = jnp.stack([jnp.array([t.next_obs.x, t.next_obs.y, t.next_obs.yaw, t.next_obs.v, t.next_obs.dir]) for t in transitions])
    r = jnp.array([t.reward for t in transitions])
    d = jnp.array([t.done for t in transitions])

    np.savez(path, obs=s, actions=a, next_obs=s_next, rewards=r, dones=d)


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
