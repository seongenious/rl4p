"""Parking environment for reinforcement learning.

This module implements a parking environment that inherits from gym.Env
for SAC-based reinforcement learning with continuous actions.
"""

import gym
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional, List
from collections import deque

from env.kinematic_model import simulate
from env.datatypes import State, Action, Observation, HistoryBuffer
from env.expert import Expert
from env.render import ParkingRenderer
from util.unit import kph2mps, mps2kph, deg2rad, rad2deg, mod2pi


class ParkingEnv(gym.Env):
    """Parking environment for reinforcement learning.
    
    This environment simulates a parking scenario where the agent must
    park a vehicle at the goal position (0, 0, 0) while avoiding obstacles.
    
    Observation:
        - Vehicle state: (x, y, yaw, v, dir) - 5 dimensions
        - Occupancy grid map: 256×256 binary grid
        - Total: 65541 dimensions
    
    Action:
        - Continuous action space: [steering_angle, acceleration]
        - steering_angle: [-1, 1]
        - acceleration: [-1, 1]
    
    Reward:
        - Based on distance to goal, heading alignment, and collision penalty
    """
    
    def __init__(self, config: Dict[str, Any], level: int = -1, render: bool = False):
        """Initialize the parking environment.
        
        Args:
            config: Environment configuration. If None, uses default config.
            level: Curriculum level.
            render: Whether to render the environment.
        """
        super().__init__()
        
        self.config = config
        self.level = level
        
        # Vehicle dimensions
        self.wheelbase = config['vehicle_config']['wheelbase']
        self.front_overhang = config['vehicle_config']['front_overhang']
        self.rear_overhang = config['vehicle_config']['rear_overhang']
        self.width = config['vehicle_config']['width']
        self.cg_to_front = self.wheelbase + self.front_overhang
        self.cg_to_rear = self.rear_overhang

        # Expert
        self.expert = Expert(config)
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32
        )
        
        # Observation space: {occupancy grid (256*256)}
        N = self.config['observation']['history_length']
        H = W = self.config['observation']['grid_size']
        C = 2  # Occupancy + vehicle 
        self.grid_size = (H, W)
        self.grid_resolution = self.config['observation']['grid_resolution']
        self.observation_space = gym.spaces.Dict(
            {
                'occupancy_grid': gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(N, H, W, C), dtype=np.float32
                ),
                'state': gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(N, 5), dtype=np.float32
                )
            }
        )
        
        # Environment state
        self.state: Optional[State] = None
        self.goal = State(x=0., y=0., yaw=0., v=0., dir=1.)
        self.occupancy_grid: Optional[jnp.ndarray] = None
        
        self.occupancy_buffer: Optional[HistoryBuffer] = None
        self.state_buffer: Optional[HistoryBuffer] = None
        self.step_count: int = 0
        self.path_length: float = 0.0
        
        # Renderer
        self.enable_render = render
        self.renderer: Optional[ParkingRenderer] = None
        
        # Initialize renderer if requested
        if self.enable_render:
            self.renderer = ParkingRenderer()
            self.renderer.set_vehicle_parameters(
                wheelbase=self.config['vehicle_config']['wheelbase'],
                front_overhang=self.config['vehicle_config']['front_overhang'],
                rear_overhang=self.config['vehicle_config']['rear_overhang'],
                width=self.config['vehicle_config']['width']
            )

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        
        # Reset variables
        self.step_count = 0
        self.occupancy_grid = None
        self.occupancy_buffer = HistoryBuffer(maxlen=self.config['observation']['history_length'])
        self.state_buffer = HistoryBuffer(maxlen=self.config['observation']['history_length'])
        self.path_length = 0.0
                
        # Clear trajectory if renderer exists
        if self.renderer is not None:
            self.renderer.clear_trajectory()
        
        # Generate random initial state without collision
        while True:
            if self.level == -1:
                yaw_max = deg2rad(self.config['env']['initial_heading'])
                x_max = self.config['env']['initial_position']
                y_max = x_max * np.sin(yaw_max)
            else:
                yaw_max = deg2rad(self.config['curriculum']['initial_heading'][self.level])
                x_max = self.config['curriculum']['initial_position'][self.level]
                y_max = x_max * np.sin(yaw_max)
                

            # Random state
            x = np.random.uniform(-x_max, x_max)
            y = np.random.uniform(-y_max, y_max)
            yaw = np.random.uniform(-yaw_max, yaw_max)
            v = 0
            dir = np.random.choice([-1, 1])
            
            # Create state
            state = State(x=x, y=y, yaw=yaw, v=v, dir=dir)
            
            # Compute path length for reward
            self.path_length = self.expert.get_path_length(state, self.goal)
                        
            # Get initial observation
            obs = self._get_observation(state)

            # Check if initial state is valid (no collision)
            if not check_collision(obs.occupancy_grid):
                self.state = state
                break
                
        info = {
            'vehicle_state': self.state,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action array [steering_angle, acceleration].
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        self.step_count += 1
        
        # Get expert trajectory
        rs_path = self.expert.get_rs_path(self.state, self.goal)
        delta, accel = self.expert.get_control_input(self.state, rs_path)
        
        # Parse action
        max_steering_angle = deg2rad(self.config['vehicle_config']['max_steering_angle'])
        steering_angle = action[0] * max_steering_angle
        
        max_accel = self.config['vehicle_config']['max_accel']
        acceleration = action[1] * max_accel
        
        # Create action object
        action = Action(delta=steering_angle, accel=acceleration)
        
        # Simulate next state
        next_state = simulate(
          state=self.state,
          action=action,
          dt=self.config['env']['dt'],
          wheelbase=self.config['vehicle_config']['wheelbase']
        )
        
        # Get observation
        obs = self._get_observation(next_state)
        
        # Check terminal condition
        done = self._check_goal_reached(next_state)
        truncated = self.step_count > self.config['env']['max_steps']
        collision = check_collision(obs.occupancy_grid) 
                
        # Compute reward
        reward = self._compute_reward(next_state, action, done, truncated, collision)
        
        # Prepare info
        info = {
            'vehicle_state': next_state,
            'rs_path': rs_path,
            'step_count': self.step_count,
        }
        
        # Update current state
        self.state = next_state
                
        return obs, reward, done, truncated, info
    
    def render(self, obs: Observation, **kwargs):
        """Render the environment.
        
        Args:
            obs: Observation.
            **kwargs: Additional arguments. (reward, done, truncated, rs_path)
        """
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        if not self.enable_render:
            return
        
        # Initialize renderer if not already done
        if self.renderer is None:
            self.renderer = ParkingRenderer()
            
            # Set vehicle parameters from config
            wheelbase = self.config['vehicle_config']['wheelbase']
            front_overhang = self.config['vehicle_config']['front_overhang']
            rear_overhang = self.config['vehicle_config']['rear_overhang']
            width = self.config['vehicle_config']['width']
            self.renderer.set_vehicle_parameters(wheelbase, front_overhang, rear_overhang, width)
        
        # Update trajectory
        self.renderer.update_trajectory(self.state)
        
        # Parse kwargs
        action, reward, done, truncated, rs_path = kwargs.values()
        
        # Render
        self.renderer.render(
            state=self.state,
            action=action,
            obs=obs,
            step_count=self.step_count,
            reward=reward,
            done=done,
            truncated=truncated,
            rs_path=rs_path,
        )
        
        # Handle events
        if not self.renderer.handle_events():
            self.close()

        return None
    
    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            
    def _compute_reward(self, state: State, action: Action, 
                         done: bool, truncated: bool, collision: bool) -> float:
        """Compute reward for current state and action.
        
        Args:
            state: Current vehicle state.
            action: Action taken.
            done: Whether episode done.
            truncated: Whether episode truncated.
            collision: Whether episode collided with obstacles.
            
        Returns:
            Reward value.
        """
        # Compute Reeds-Shepp path
        path_length = self.expert.get_path_length(state, self.goal)
        length_reduction = self.path_length - path_length
        self.path_length = path_length

        # Path length penalty
        reward = length_reduction * self.config['reward']['length_reduction']

        # Success bonus
        if done:
            reward += self.config['reward']['success']

        # Time limit penalty
        if truncated:
            reward += self.config['reward']['timeout']
            
        # Collision penalty
        if collision:
            reward += self.config['reward']['collision']
        
        return reward

    def _get_observation(self, state: State) -> Observation:
        """Get current observation.
        
        Returns:
            Observation: {vehicle_state, occupancy_grid}.
        """        
        # Update occupancy grid and history buffer
        self._update_occupancy_grid(state)
        
        # Get stacked occupancy grid
        obs = Observation(
            occupancy_grid=self.occupancy_buffer.get_stacked(), 
            state=self.state_buffer.get_stacked()
        )
        
        return obs
    
    def _check_goal_reached(self, state: State) -> bool:
        """Check if vehicle has reached the goal.
        
        Args:
            state: Current vehicle state.
            
        Returns:
            True if goal is reached.
        """
        if self.level == -1:
            distance_threshold = self.config['env']['distance_threshold']
            heading_threshold = deg2rad(self.config['env']['heading_threshold'])
            zero_speed = self.config['env']['zero_speed']
        else:
            distance_threshold = self.config['curriculum']['distance_threshold'][self.level]
            heading_threshold = deg2rad(self.config['curriculum']['heading_threshold'][self.level])
            zero_speed = self.config['curriculum']['zero_speed'][self.level]

        # Check position
        distance_error = np.sqrt(state.x**2 + state.y**2)
        if distance_error > distance_threshold:
            return False
        
        # Check heading
        heading_error = np.abs(mod2pi(state.yaw))
        if heading_error > heading_threshold:
            return False
        
        # Check velocity
        if np.abs(state.v) > zero_speed:
            return False
        
        return True
    
    def _update_occupancy_grid(self, state: State):
        """Update occupancy grid observation from vehicle perspective.
        
        Args:
            state: Current vehicle state.
        """
        origin = jnp.array([
            -self.grid_size[0] * self.grid_resolution / 2,
            -self.grid_size[1] * self.grid_resolution / 2
        ])  
        
        #TODO: Create empty occupancy grid first, later fill it with obstacles
        # obs_center = jnp.array([1.4, 3.0, 0])
        # obs_polygon = self._create_polygon(
        #     center=obs_center,
        #     cg_to_front=2.5,
        #     cg_to_rear=2.5,
        #     width=2.0
        # )
        # obs_grid = create_gaussian_buffered_mask(
        #     polygons=obs_polygon,
        #     sigma=self.config['observation']['sigma'],
        #     buffer=self.config['observation']['buffer'],
        #     resolution=self.config['observation']['grid_resolution'],
        #     origin=origin
        # )
        
        # Create state grid
        ego_polygon = self._create_polygon(
            center=jnp.array([state.x, state.y, state.yaw]),
            cg_to_front=self.cg_to_front,
            cg_to_rear=self.cg_to_rear,
            width=self.width
        )
        ego_grid = create_gaussian_buffered_mask(
            polygons=ego_polygon, 
            sigma=self.config['observation']['sigma'],
            buffer=self.config['observation']['buffer'],
            resolution=self.config['observation']['grid_resolution'],
            origin=origin
        )
        obs_grid = jnp.zeros_like(ego_grid)
        
        # Update occupancy grid
        self.occupancy_grid = jnp.clip(
            jnp.stack([obs_grid, ego_grid], axis=-1), 0.0, 1.0)  # (H, W, 2)
        # self.occupancy_grid = (255.0 * self.occupancy_grid).astype(jnp.uint8)

        # Update history buffer
        self.occupancy_buffer.append(self.occupancy_grid)        
        self.state_buffer.append(jnp.array([state.x, state.y, state.yaw, state.v, state.dir]))
        
    def _create_polygon(self, 
                        center: jnp.ndarray, 
                        cg_to_front: float, 
                        cg_to_rear: float, 
                        width: float) -> jnp.ndarray:
        """Create a polygon from vehicle parameters.

        Args:
            center: (N, 3) float32, vehicle center pose (x, y, yaw)
            cg_to_front: (N,) float32, distance from center to front
            cg_to_rear: (N,) float32, distance from center to rear
            width: (N,) float32, vehicle width
            
        Returns:
            polygon: (N, 4, 2) float32, each polygon has 4 (x, y) corners
        """
        # center: (N, 3) - [x, y, yaw]
        if center.ndim == 1:
            center = center[None, :]
            
        N = center.shape[0]
        
        # Local corners
        corners = jnp.array([
            [cg_to_front, width / 2],
            [cg_to_front, -width / 2],
            [-cg_to_rear, -width / 2],
            [-cg_to_rear, width / 2]
        ])  # shape: (4, 2)
        
        # Repeat corners N times
        corners = jnp.broadcast_to(corners, (N, 4, 2))  # shape: (N, 4, 2)
        
        # Rotation matrix
        yaw = center[:, 2]
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        rot_matrix = jnp.stack([
            jnp.stack([c, -s], axis=-1),  # (N, 2)
            jnp.stack([s, c], axis=-1),  # (N, 2)
        ], axis=1)  # (N, 2, 2)
        
        # Apply rotation: (N, 4, 2)
        rotated_corners = jnp.einsum('nij,nkj->nki', rot_matrix, corners)
        
        # Translation
        translated_corners = rotated_corners + center[:, None, :2]
        
        return translated_corners
    
@jax.jit
def check_collision(occupancy_grid: jnp.ndarray) -> bool:
    """Check if vehicle collides with obstacles.
        
    Returns:
        True if collision detected.
    """ 
    collision = jnp.any(jnp.sum(occupancy_grid[-1, ...], axis=-1) > 1.01)
    return collision

@jax.jit
def create_gaussian_buffered_mask(
    polygons: jnp.ndarray,              # (N, 4, 2) in world coords
    sigma=3.0,
    buffer=1.0,
    resolution=0.1,                     # meters per pixel
    origin=jnp.array([0.0, 0.0])        # world coord of grid[0, 0]
) -> jnp.ndarray:
    H, W = 256, 256  # Fixed for jax.jit
    ys = jnp.arange(H - 1, -1, -1)
    xs = jnp.arange(W)
    grid_x, grid_y = jnp.meshgrid(xs, ys, indexing='ij')  # (H, W)
    grid = jnp.stack([grid_x, grid_y], axis=-1)  # (H, W, 2), in pixel coord

    # Grid coordinates
    grid_polygons = world2grid(polygons, origin=origin, resolution=resolution)  # (N, 4, 2)
    
    def covered_by(point: jnp.ndarray, poly: jnp.ndarray) -> jnp.ndarray:
        """Check whether each pixel is covered by the polygon (grid coords)
        
        Args:
            point: (2,) float32, pixel coord
            poly: (4, 2) float32, polygon vertices in pixel coord
        
        Returns:
            mask: (H, W) float32, 1 if in polygon, 0 otherwise
        """
        x, y = point
        x0, y0 = poly[:, 0], poly[:, 1]
        x1 = jnp.roll(x0, -1)
        y1 = jnp.roll(y0, -1)
        cond1 = ((y0 > y) != (y1 > y))
        slope = (x1 - x0) / (y1 - y0 + 1e-6)
        x_int = x0 + slope * (y - y0)
        cond2 = x < x_int
        return jnp.sum(jnp.logical_and(cond1, cond2)) % 2 == 1
    
    def distance_to_polygon(p: jnp.ndarray, poly: jnp.ndarray) -> float:
        """Minimum distance from point p to polygon edges
        
        Args:
            p: (2,) float32, pixel coord
            poly: (4, 2) float32, polygon vertices in pixel coord
        
        Returns:
            dist: (H, W) float32, minimum distance to polygon edges
        """
        def distance_to_segment(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> float:
            """Distance from point p to segment (a, b)"""
            ap = p - a
            ab = b - a
            t = jnp.clip(jnp.dot(ap, ab) / (jnp.dot(ab, ab) + 1e-6), 0.0, 1.0)
            proj = a + t * ab
            return jnp.linalg.norm(p - proj)

        edges = jnp.stack([poly, jnp.roll(poly, -1, axis=0)], axis=1)  # (4, 2, 2)
        dists = jax.vmap(lambda a, b: distance_to_segment(p, a, b))(edges[:, 0], edges[:, 1])  # (4,)
        return jnp.min(dists)

    def single_buffered_mask(poly: jnp.ndarray) -> jnp.ndarray:
        """Check whether each pixel is in the polygon (grid coords)
        
        Args:
            poly: (4, 2) float32, polygon vertices in pixel coord
        
        Returns:
            mask: (H, W) float32, 1 if in polygon, 0 otherwise
        """
        def compute_value(pt: jnp.ndarray) -> float:
            inside = covered_by(pt, poly)
            dist = distance_to_polygon(pt, poly)
            return jnp.where(
                inside,
                1.0,
                jnp.where(
                    dist <= buffer, 
                    jnp.clip(jnp.exp(- (dist ** 2) / (2 * sigma ** 2)), 0.0, 1.0), 
                    0.0
                )
            )

        value_fn = jax.vmap(jax.vmap(compute_value, in_axes=0), in_axes=0)
        return value_fn(grid)

    masks = jax.vmap(single_buffered_mask)(grid_polygons)
    total_mask = jnp.clip(jnp.sum(masks, axis=0), 0.0, 1.0)

    return total_mask  # shape: (H, W)

@jax.jit
def world2grid(xy: jnp.ndarray, origin: jnp.ndarray, resolution: float) -> jnp.ndarray:
    """
    Convert world coordinates (meters) to grid indices (pixels).
    - xy: (..., 2)
    - origin: (2,) world coord of grid[0, 0]
    - resolution: meters per pixel
    """
    return ((xy - origin) / resolution).astype(jnp.int32)