import chex
from typing import Dict, Any
from collections import deque

import jax
import jax.numpy as jnp

@chex.dataclass
class State:
    x: float    # [m]
    y: float    # [m]
    yaw: float  # [rad]
    v: float    # [m/s]
    dir: int=1  # 1: forward, -1: reverse


@chex.dataclass
class Action:
    delta: float  # [-1, 1] normalized
    accel: float  # [-1, 1] normalized


@chex.dataclass
class Transition:
    obs: Dict[str, Any]
    action: Action
    next_obs: Dict[str, Any]
    reward: float
    done: bool
    truncated: bool
    expert: Action


@chex.dataclass
class Observation:
    occupancy_grid: jnp.ndarray   # shape = (batch, N, H, W, C)
    

class HistoryBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)

    def append(self, item: jnp.ndarray) -> None:
        """Append an item to the buffer.
        
        Args:
            item: An item to append. (256, 256, C)
        """
        self.buffer.append(item)

    def get_stacked(self) -> jnp.ndarray:
        """Get stacked items from the buffer.
        
        Returns:
            A stacked array of items. (T, 256, 256, C)
        """
        return jnp.stack(self.buffer, axis=0)