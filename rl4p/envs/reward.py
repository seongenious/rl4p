import jax
import jax.numpy as jnp
import reeds_shepp as rs

from typing import Tuple, List

from envs.datatypes import State, Action
from utils.unit import kph2mps, deg2rad


def reward_fn(start: State, goal: State, config: dict) -> Tuple[float, bool, List]:
    """Compute reward for a trajectory segment based on Reeds-Shepp path.

    Args:
        start: The current state of the vehicle (x, y, yaw).
        goal: The goal state to reach (x, y, yaw).
        config: Dictionary containing vehicle and reward configuration. Must not be None.
            Required fields:
              - config['reward']['turning_radius']: minimum turning radius.
              - config['reward']['step_size']: step size used for path sampling.
              - config['reward']['goal_reach']: reward when goal is reached.
              - config['reward']['collision']: penalty for collision.
              - config['reward']['heading_threshold']: acceptable yaw error.

    Returns:
        A tuple of:
            - reward (float): Combined reward from path length, goal, and collision penalty.
            - done (bool): True if goal is reached or a collision occurs.

    Raises:
        ValueError: If config is None.
    """
    if config is None:
        raise ValueError("`config` must not be None. Provide vehicle and reward settings.")

    # Load configuration
    r_min = config['vehicle_config']['min_turning_radius']
    a_lat_max = config['vehicle_config']['max_lateral_accel']
    path_resolution = config['reward']['path_resolution']
    goal_reward_val = config['reward']['goal_reach']
    collision_penalty_val = config['reward']['collision']
    yaw_thres = deg2rad(config['reward']['heading_threshold'])
    vel_thres = kph2mps(config['reward']['velocity_threshold'])

    # Initial and goal poses
    q0 = (start.x, start.y, start.yaw)
    q1 = (goal.x, goal.y, goal.yaw)
    
    # Get RS path
    radius = turning_radius_from_dynamics(start.v, a_lat_max, r_min)
    length = rs.path_length(q0, q1, radius)
    points = rs.path_sample(q0, q1, radius, step_size=path_resolution)
    
    # Path reward (shorter path = higher reward)
    path_reward = -length

    # Sub goal is direction switching point
    idx = find_direction_switch_idx(jnp.array(points))
    sub_goal = points[idx]
    
    # Goal reward
    dx = start.x - sub_goal[0]
    dy = start.y - sub_goal[1]
    distance = jnp.sqrt(dx**2 + dy**2)
    yaw_diff = jnp.abs(start.yaw - sub_goal[2])

    done = False
    goal_reward = 0.0
    if distance < path_resolution and yaw_diff < yaw_thres and start.v <= vel_thres:
        goal_reward = goal_reward_val
        done = True

    # Collision penalty
    collision_penalty = 0.0
    for (x, y, yaw, s, kappa) in points:
        # TODO: Replace `False` with actual collision check
        if False:
            collision_penalty = collision_penalty_val
            done = True
            break

    total_reward = path_reward + goal_reward + collision_penalty
    return total_reward, done


def turning_radius_from_dynamics(v: float, a_lat_max: float = 4., r_min: float = 6.) -> float:
    """Compute turning radius according to vehicle dynamics.
    
    Args:
        v: vehicle speed (m/s)
        a_lat_max: Maximum allowable lateral acceleration (m/s^2)
        r_min: Minimum turning raduis to prevent zero value
    
    Returns:
        Turning radius in m
    """
    return jnp.maximum(v**2 / a_lat_max, r_min)


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