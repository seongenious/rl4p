import chex
from typing import Dict, Any


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