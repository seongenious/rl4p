from typing import Dict, Tuple, List

import jax
import jax.numpy as jnp
import reeds_shepp as rs

from env.datatypes import State
from util.unit import mod2pi, kph2mps, rad2deg, deg2rad

MAX_SPEED = kph2mps(10)
STEP_TIME = 0.1


class Expert:
    def __init__(self, config: Dict):
        # Vehicle dynamics
        self.max_lateral_accel = config['vehicle_config']['max_lateral_accel']
        self.min_turning_radius = config['rs_path']['turning_radius']
        self.step_size = config['rs_path']['resolution']
        self.max_speed = config['vehicle_config']['max_speed']
        self.max_steering_angle = deg2rad(config['vehicle_config']['max_steering_angle'])
        self.max_accel = config['vehicle_config']['max_accel']
        self.wheelbase = config['vehicle_config']['wheelbase']
    
    def set_reference_path(self, path: List[State]):
        """Set reference path. Split path into segments by direction change.
        
        Args:
            path: List[State]
        """
        self.segments = []
        self.segment_idx = 0
        
        path = jnp.array([(p.x, p.y, p.yaw, p.v, p.dir) for p in path])

        direction = path[:, 4]
        change_points = jnp.where(direction[1:] != direction[:-1])[0]
        split_indices = jnp.concatenate([jnp.array([-1]), change_points, jnp.array([-1])])

        for i in range(len(split_indices) - 1):
            start = split_indices[i] + 1
            end = split_indices[i + 1]
            segment = path[start:end]
            self.segments.append(segment)

    def get_path_length(self, start: State, goal: State) -> float:
        """Get Reeds-Shepp path length.
        
        Args:
            start: Start state. (x, y, yaw)
            goal: Goal state. (x, y, yaw)
        
        Returns:
            Path length.
        """
        radius = turning_radius_from_dynamics(
            start.v, self.max_lateral_accel, self.min_turning_radius)
        q0 = (start.x, start.y, start.yaw)
        q1 = (goal.x, goal.y, goal.yaw)
        path_length = rs.path_length(q0, q1, radius)
        
        return path_length
    
    def get_rs_path(self, start: State, goal: State) -> List[State]:
        """Get Reeds-Shepp path.
        
        Args:
            start: Start state. (x, y, yaw)
            goal: Goal state. (x, y, yaw)
        
        Returns:
            Path. (x, y, yaw, s, kappa)
        """
        radius = turning_radius_from_dynamics(
            start.v, self.max_lateral_accel, self.min_turning_radius)
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
        
        path = [
            State(x=p[0], y=p[1], yaw=p[2], v=0, dir=dir) for p, dir in zip(points, directions)
        ]

        return path

    def get_control_input(self, state: State) -> Tuple[float, float]:
        """Get control input.
        
        Args:
            state: State

        Returns:
            control input: (delta, accel)
        """
        path = self.segments[self.segment_idx]

        # Find nearest point in the segment
        xy = jnp.array([state.x, state.y])
        dists = jnp.linalg.norm(xy - path[:, :2], axis=-1)
        idx = jnp.argmin(dists)
        curr_dir = path[0, 4]

        if self.segment_idx < len(self.segments) - 1:
            next_path = self.segments[self.segment_idx + 1]
            next_dir = next_path[0, 4]

            # Advance to next segment 
            if state.dir == next_dir and idx == len(path) - 1:
                self.segment_idx += 1

        # Compute remaining length
        seg_dist = jnp.linalg.norm(path[:, :2][1:] - path[:, :2][:-1], axis=-1)
        remaining_length = jnp.sum(seg_dist[idx:])
        dist_ego_to_start = jnp.linalg.norm(xy - path[:, :2][0])
        dist_ego_to_end = jnp.linalg.norm(xy - path[:, :2][-1])
        dist_start_to_end = jnp.linalg.norm(path[:, :2][0] - path[:, :2][-1])
        s = jnp.where(dist_ego_to_start > dist_start_to_end, -dist_ego_to_end, remaining_length)
        
        # Compute steering angle using stanley method
        delta = compute_steering_angle(state, path)

        # Compute acceleration using IDM
        if state.dir != curr_dir:
            accel = -0.5
        else:
            accel = compute_acceleration(state, s)

        # Normalize to [-1, 1]
        delta = jnp.clip(delta / self.max_steering_angle, -1.0, 1.0)
        accel = jnp.clip(accel / self.max_accel, -1.0, 1.0)
        return delta, accel

@jax.jit
def turning_radius_from_dynamics(v: float, 
                                 max_lateral_accel: float, 
                                 min_turning_radius: float) -> float:
    """Compute turning radius according to vehicle dynamics.
    
    Args:
        v: vehicle speed (m/s)
        max_lateral_accel: maximum lateral acceleration (m/s^2)
        min_turning_radius: minimum turning radius (m)
    
    Returns:
        Turning radius in m
    """
    return jnp.maximum(v**2 / max_lateral_accel, min_turning_radius)

def find_direction_switch_idx(path: jnp.ndarray) -> int:
    """
    Estimate driving direction (+1: forward, -1: backward) at each point
    based on heading and displacement vector.
    
    Args:
        path: jnp.ndarray of (x, y, yaw, v, dir).
    
    Returns:
        direction switch index: int.
    """
    direction = path[:, 4]
    change = jnp.where(direction[1:] != direction[:-1])[0]
    idx = change[0] if len(change) > 0 else -1
    
    return idx

@jax.jit
def project_point_to_segment(p: jnp.ndarray, 
                             p0: jnp.ndarray, 
                             p1: jnp.ndarray) -> Tuple[float, jnp.ndarray, float]:
    """Project a point to a segment.
    
    Args:
        p: (2,) - point
        p0: (2,) - segment start point
        p1: (2,) - segment end point
    
    Returns:
        distance: float
        projected point: (2,)
        t: float
    """
    v = p1 - p0
    w = p - p0
    v_norm_sq = jnp.dot(v, v) + 1e-6  # small epsilon to avoid divide-by-zero

    t = jnp.clip(jnp.dot(w, v) / v_norm_sq, 0.0, 1.0)
    proj = p0 + t * v
    dist = jnp.linalg.norm(p - proj)
    return dist, proj, t

@jax.jit
def compute_error_to_path(
    pose: jnp.ndarray, 
    path: jnp.ndarray
) -> Tuple[float, jnp.ndarray, int, float]:
    """
    Compute error to the path.
    
    Args:
        pose: (3,) - vehicle position (x, y, yaw)
        path: (N, 5) - path (x, y, yaw, s, kappa)
    
    Returns:
        lateral_error: float, lateral error
        heading_error: float, heading error
    """
    # Get closest point on the path
    path_xy = path[:, :2]
    path_yaw = path[:, 2]

    p0 = path_xy[:-1]
    p1 = path_xy[1:]
    yaw0 = path_yaw[:-1]
    yaw1 = path_yaw[1:]

    def project_segment(p0_i, p1_i, yaw0_i, yaw1_i):
        """Compute distance and heading error at projected point."""
        dist, proj, t = project_point_to_segment(pose[:2], p0_i, p1_i)
        tangent = p1_i - p0_i
        tangent = tangent / (jnp.linalg.norm(tangent) + 1e-6)

        delta = jnp.arctan2(jnp.sin(yaw1_i - yaw0_i), jnp.cos(yaw1_i - yaw0_i))
        yaw_interp = yaw0_i + t * delta

        vec = pose[:2] - proj
        cross = tangent[0] * vec[1] - tangent[1] * vec[0]
        signed_dist = jnp.sign(cross) * jnp.linalg.norm(vec)

        return signed_dist, yaw_interp

    dists, yaws = jax.vmap(project_segment)(p0, p1, yaw0, yaw1)
    idx = jnp.argmin(dists)

    lateral_error = dists[idx] 
    heading_error = jnp.arctan2(jnp.sin(yaws[idx] - pose[2]), jnp.cos(yaws[idx] - pose[2]))

    return lateral_error, heading_error

@jax.jit
def compute_steering_angle(state: State, 
                           path: jnp.ndarray, 
                           k: float = 0.5,
                           soft_eps: float = 1e-6) -> float:
    """Compute steering angle using stanley method.
    
    Args:
        state: State
        path: jnp.ndarray
        k: float
        soft_eps: float
    
    Returns:
        steering angle: float
    """
    # Compute error to the path
    pose = jnp.array([state.x, state.y, state.yaw])
    e_lat, e_yaw = compute_error_to_path(pose, path)
    
    # Stanley formula
    e_yaw = mod2pi(jnp.where(state.dir == 1, e_yaw, e_yaw - jnp.pi))
    v = jnp.where(state.dir == 1, state.v + soft_eps, -state.v - soft_eps)
    delta = e_yaw + jnp.arctan2(k * (e_lat), v)
    return delta

@jax.jit
def compute_acceleration(state: State, s: float) -> float:
    """Compute acceleration with dynamic idx, fully JAX-compatible."""
    a = 1.0  # max acceleration
    b = 1.0  # comfortable braking deceleration
    s0 = 0.0  # bumper-to-bumper distance at t=0
    t_gap = 0.0  # time gap
    v0 = MAX_SPEED
    v = jnp.maximum(state.v, 0.1)  # current speed
    dv = v0 - v  # speed difference to the front vehicle
    delta = 4.0  # max acceleration exponent
    s = jnp.maximum(s, 1e-3)

    # Compute acceleration by IDM
    s_star = s0 + jnp.maximum(0., v * t_gap + v * dv / (2.0 * jnp.sqrt(a * b)))
    accel = a * (1.0 - (v / v0)**delta - (s_star / s)**2.0)

    return accel

