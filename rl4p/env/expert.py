from typing import Dict, Tuple, List

import jax
import jax.numpy as jnp
import reeds_shepp as rs
import numpy as np

from env.datatypes import State
from util.unit import mod2pi, kph2mps, rad2deg, deg2rad

MAX_SPEED = kph2mps(10)
STEP_TIME = 0.1


class Expert:
    def __init__(self, config: Dict):
        # Vehicle dynamics
        self.max_a_lat = config['vehicle_config']['max_a_lat']
        self.min_turning_radius = config['rs_path']['turning_radius']
        self.step_size = config['rs_path']['resolution']
        self.max_speed = config['vehicle_config']['max_speed']

    def get_path_length(self, start: State, goal: State) -> float:
        """Get Reeds-Shepp path length.
        
        Args:
            start: Start state. (x, y, yaw)
            goal: Goal state. (x, y, yaw)
        
        Returns:
            Path length.
        """
        radius = turning_radius_from_dynamics(
            start.v, self.max_a_lat, self.min_turning_radius)
        q0 = (start.x, start.y, start.yaw)
        q1 = (goal.x, goal.y, goal.yaw)
        path_length = rs.path_length(q0, q1, radius)
        
        return path_length
    
    def get_rs_path(self, start: State, goal: State) -> jnp.ndarray:
        """Get Reeds-Shepp path.
        
        Args:
            start: Start state. (x, y, yaw)
            goal: Goal state. (x, y, yaw)
        
        Returns:
            Path. N * (x, y, yaw, dir)
        """
        radius = turning_radius_from_dynamics(
            start.v, self.max_a_lat, self.min_turning_radius)
        q0 = (start.x, start.y, start.yaw)
        q1 = (goal.x, goal.y, goal.yaw)
        points = rs.path_sample(q0, q1, radius, self.step_size)
        points = jnp.array(points)

        # Recalculate direction at each point
        position = points[:, :2]
        dxdy = position[1:] - position[:-1]
        heading = points[:, 2]
        tangent = jnp.stack([jnp.cos(heading[:-1]), jnp.sin(heading[:-1])], axis=-1)
        
        dot = jnp.sum(dxdy * tangent, axis=-1)
        directions = jnp.sign(dot)  # shape: (N-1,), values: +1 or -1
        directions = jnp.concatenate([directions, directions[-1:]], axis=-1)
        
        # Reshape directions to (N, 1) to concatenate with points[:, :3] (N, 3)
        directions = directions.reshape(-1, 1)  # shape: (N, 1)
        path = jnp.concatenate([points[:, :3], directions], axis=-1)  # shape: (N, 4)

        return path

@jax.jit
def turning_radius_from_dynamics(v: float, 
                                 max_a_lat: float, 
                                 min_turning_radius: float) -> float:
    """Compute turning radius according to vehicle dynamics.
    
    Args:
        v: vehicle speed (m/s)
        max_a_lat: maximum lateral acceleration (m/s^2)
        min_turning_radius: minimum turning radius (m)
    
    Returns:
        Turning radius in m
    """
    return jnp.maximum(v**2 / max_a_lat, min_turning_radius)