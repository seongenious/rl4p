import jax
import jax.numpy as jnp

from envs.datatypes import State, Action

import reeds_shepp as rs

TURNING_RADIUS = 10
STEP_SIZE = 0.1


def rs_path(start: State, goal: State):
    """
    Compute Reeds-Shepp path and its length
    """
    q0 = (start.x, start.y, start.yaw)
    q1 = (goal.x, goal.y, goal.yaw)
    
    l = rs.path_length(q0, q1, TURNING_RADIUS)
    points = rs.path_sample(q0, q1, TURNING_RADIUS, step_size=STEP_SIZE)

    return l, points