import chex


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
    obs: State
    action: Action
    next_obs: State
    reward: float
    done: bool