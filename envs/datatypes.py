import chex


@chex.dataclass
class State:
    x: float
    y: float
    yaw: float  # radians
    v: float    # m/s


@chex.dataclass
class Action:
    delta: float   # [-1, 1]
    accel: float   # [-1, 1]


@chex.dataclass
class Transition:
    state: State
    action: Action
    next_state: State
    reward: float
    done: bool