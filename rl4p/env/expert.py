from typing import Dict, Tuple, List

import jax
import jax.numpy as jnp
import reeds_shepp as rs

from env.datatypes import State
from util.unit import mod2pi

class Expert:
    def __init__(self, config: Dict):
        # Vehicle dynamics
        self.max_lateral_accel = config['vehicle_config']['max_lateral_accel']
        self.min_turning_radius = config['rs_path']['turning_radius']
        self.step_size = config['rs_path']['resolution']
        self.max_speed = config['vehicle_config']['max_speed']
        self.max_steering_angle = config['vehicle_config']['max_steering_angle']
        self.max_accel = config['vehicle_config']['max_accel']
        self.wheelbase = config['vehicle_config']['wheelbase']

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
        path = [State(x=p[0], y=p[1], yaw=p[2], v=0, dir=1) for p in points]

        return path

    def get_control_input(self, state: State, path: List[State]) -> Tuple[float, float]:
        """Get control input.
        
        Args:
            state: State
            path: List[State]

        Returns:
            control input: (delta, accel)
        """
        # Convert path to jnp.ndarray
        path = jnp.array([(p.x, p.y, p.yaw, p.v, p.dir) for p in path])

        return compute_control_input(state, path, self.wheelbase)

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

@jax.jit
def find_direction_switch_idx(path: jnp.ndarray) -> int:
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
        dist, _, t = project_point_to_segment(pose[:2], p0_i, p1_i)
        delta = jnp.arctan2(jnp.sin(yaw1_i - yaw0_i), jnp.cos(yaw1_i - yaw0_i))
        yaw_interp = yaw0_i + t * delta
        return dist, yaw_interp

    dists, yaws = jax.vmap(project_segment)(p0, p1, yaw0, yaw1)
    idx = jnp.argmin(dists)

    return dists[idx], mod2pi(pose[2] - yaws[idx])

@jax.jit
def compute_control_input(state: State, 
                          path: jnp.ndarray, 
                          wheelbase: float) -> Tuple[float, float]:
    """Compute control input.
    
    Args:
        state: State
        path: jnp.ndarray
        wheelbase: float
    
    Returns:
        control input: (delta, accel)
    """
    # Compute error to the path
    pose = jnp.array([state.x, state.y, state.yaw])
    e_lat, e_yaw = compute_error_to_path(pose, path)

    # Compute steering angle by LQR
    v = jnp.maximum(state.v, 1.)
    dt = 0.1
    A = jnp.array([[1., v*dt], [0., 1.]])
    B = jnp.array([[0.], [v*dt / wheelbase]])

    Q = jnp.array([[1., 0.], [0., 2.]])
    R = jnp.array([[0.1]])
    
    K = discrete_lqr(A, B, Q, R)
    x = jnp.array([e_lat, e_yaw])
    delta = -K @ x
    
    accel = 0

    return delta, accel
    
@jax.jit
def solve_discrete_are(A: jnp.ndarray, 
                       B: jnp.ndarray, 
                       Q: jnp.ndarray, 
                       R: jnp.ndarray, 
                       max_iters: int = 100, 
                       tol: float = 1e-6) -> jnp.ndarray:
    """Iterative solution to discrete-time Algebraic Riccati Equation (DARE)."""
    def body_fn(P):
        BT_P = B.T @ P
        inv_term = jnp.linalg.inv(R + B.T @ P @ B)
        temp = BT_P.T @ (inv_term @ BT_P)
        P_new = Q + A.T @ (P - temp) @ A
        return P_new

    def cond_fn(val):
        i, P, P_prev = val
        diff = jnp.max(jnp.abs(P - P_prev))
        return (i < max_iters) & (diff > tol)

    def loop_fn(val):
        i, P, P_prev = val
        P_new = body_fn(P)
        return i + 1, P_new, P

    # Initial value: P = Q
    i, P_final, _ = jax.lax.while_loop(cond_fn, loop_fn, (0, Q, Q))
    return P_final

@jax.jit
def discrete_lqr(A: jnp.ndarray, 
                 B: jnp.ndarray, 
                 Q: jnp.ndarray, 
                 R: jnp.ndarray) -> jnp.ndarray:
    """Computes discrete-time LQR gain matrix K"""
    P = solve_discrete_are(A, B, Q, R)
    K = jnp.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K
