import jax
import jax.numpy as jnp

from envs.datatypes import State, Action, Transition
from utils.unit import mod2pi

ZERO_SPEED = 0.03


def simulate(state: State, action: Action, dt: float, wheelbase: float) -> State:
    """Step forward in the environment using the given action.

    Args:
        state: Current state (x, y, yaw, v, dir)
        action: Action to apply (delta, accel)
        dt: Step time 
        wheelbase: Wheel base of the vehicle

    Returns:
        A next State object (x, y, yaw, v, dir)
    """
    v_next = state.v + action.accel * dt
    if v_next > ZERO_SPEED:
        dir_next = 1
    elif v_next < -ZERO_SPEED:
        dir_next = -1
    else:
        v_next = 0
        dir_next = state.dir

    x_next = state.x + v_next * jnp.cos(state.yaw) * dt 
    y_next = state.y + v_next * jnp.sin(state.yaw) * dt 
    yaw_next = mod2pi(state.yaw + v_next / wheelbase * jnp.tan(action.delta) * dt)

    return State(x=x_next, y=y_next, yaw=yaw_next, v=v_next, dir=dir_next)