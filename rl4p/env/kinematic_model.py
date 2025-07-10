import jax
import jax.numpy as jnp

from env.datatypes import State, Action
from util.unit import mod2pi, rad2deg, deg2rad

ZERO_SPEED = 0.1
MAX_YAWRATE = deg2rad(30.0)


@jax.jit
def simulate(state: State, action: Action, dt: float) -> State:
    """Step forward in the environment using the given action.

    Args:
        state: Current state (x, y, yaw, v)
        action: Action to apply (a_lon, a_lat)
        dt: Step time 

    Returns:
        A next State object (x, y, yaw, v)
    """
    yaw_rate = jnp.clip(
        jnp.where(
            jnp.abs(state.v) > ZERO_SPEED, action.a_lat / state.v, 0.), 
        -MAX_YAWRATE, 
        MAX_YAWRATE
    )
    yaw_next = mod2pi(state.yaw + yaw_rate * dt)
    x_next = state.x + state.v * jnp.cos(yaw_next) * dt 
    y_next = state.y + state.v * jnp.sin(yaw_next) * dt 
    v_next = state.v + action.a_lon * dt

    return State(x=x_next, y=y_next, yaw=yaw_next, v=v_next)