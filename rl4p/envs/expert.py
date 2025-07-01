from typing import Dict, Tuple, List

import jax.numpy as jnp
import numpy as np
import reeds_shepp as rs

from envs.datatypes import State

class Expert:
    def __init__(self, config: Dict):
        # Vehicle dynamics
        self.max_lateral_accel = config['vehicle_config']['max_lateral_accel']
        self.min_turning_radius = config['rs_path']['turning_radius']
        self.step_size = config['rs_path']['resolution']
        self.max_speed = config['vehicle_config']['max_speed']
        self.max_steering_angle = config['vehicle_config']['max_steering_angle']
        self.max_accel = config['vehicle_config']['max_accel']

    def get_rs_path(self, start: State, goal: State) -> List[State]:
        """Get Reeds-Shepp path.
        
        Args:
            start: Start state. (x, y, yaw)
            goal: Goal state. (x, y, yaw)
        
        Returns:
            Path. (x, y, yaw, s, kappa)
        """
        radius = self.turning_radius_from_dynamics(start.v)
        q0 = (start.x, start.y, start.yaw)
        q1 = (goal.x, goal.y, goal.yaw)
        points = rs.path_sample(q0, q1, radius, self.step_size)
        path = [State(x=p[0], y=p[1], yaw=p[2], v=0, dir=1) for p in points]

        return path

    def turning_radius_from_dynamics(self, v: float) -> float:
        """Compute turning radius according to vehicle dynamics.
        
        Args:
            v: vehicle speed (m/s)
        
        Returns:
            Turning radius in m
        """
        return jnp.maximum(v**2 / self.max_lateral_accel, self.min_turning_radius)

    def find_direction_switch_idx(self, path: jnp.ndarray) -> int:
        """
        Estimate driving direction (+1: forward, -1: backward) at each point
        based on heading and displacement vector.
        
        Args:
            path: jnp.ndarray of (x, y, yaw, s, kappa).
        
        Returns:
            direction switch index: int.
        """
        pos = path[:, :2]  # (x, y)
        yaw = path[:, 2]   # yaw
        delta = pos[1:] - pos[:-1]  # displacement vector
        heading = jnp.stack([jnp.cos(yaw[:-1]), jnp.sin(yaw[:-1])], axis=-1)

        dot = jnp.sum(delta * heading, axis=-1)
        direction = jnp.sign(dot)  # shape: (N-1,), values: +1 or -1
        change = jnp.where(direction[1:] != direction[:-1])[0]
        
        return int(change[0] - 1) if len(change) > 0 else -1