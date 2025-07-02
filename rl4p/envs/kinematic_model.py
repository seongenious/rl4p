import jax
import jax.numpy as jnp

from envs.datatypes import State, Action, Transition
from utils.unit import mod2pi

ZERO_SPEED = 0.03


@jax.jit
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
    v = state.v + state.dir * action.accel * dt

    dir_next = jnp.where(v < -ZERO_SPEED, -state.dir, state.dir)
    v_next = jnp.where(v < ZERO_SPEED, 0., v)
    x_next = state.x + dir_next * v_next * jnp.cos(state.yaw) * dt 
    y_next = state.y + dir_next * v_next * jnp.sin(state.yaw) * dt 
    yaw_next = mod2pi(state.yaw + dir_next * v_next / wheelbase * jnp.tan(action.delta) * dt)

    return State(x=x_next, y=y_next, yaw=yaw_next, v=v_next, dir=dir_next)