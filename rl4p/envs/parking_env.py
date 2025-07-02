"""Parking environment for reinforcement learning.

This module implements a parking environment that inherits from gym.Env
for SAC-based reinforcement learning with continuous actions.
"""

import gym
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import reeds_shepp as rs
import pygame

from envs.kinematic_model import simulate
from envs.datatypes import State, Action
from envs.expert import Expert
from envs.render import ParkingRenderer, RenderConfig
from utils.unit import kph2mps, mps2kph, deg2rad, rad2deg, mod2pi


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
        - steering_angle: [-max_steering_angle, max_steering_angle]
        - acceleration: [-max_accel, max_accel]
    
    Reward:
        - Based on distance to goal, heading alignment, and collision penalty
    """
    
    def __init__(self, config: Dict[str, Any], render: bool = False):
        """Initialize the parking environment.
        
        Args:
            config: Environment configuration. If None, uses default config.
            render: Whether to render the environment.
        """
        super().__init__()
        
        self.config = config

        # Expert
        self.expert = Expert(config)
        
        # Define action and observation spaces
        max_steering_angle = deg2rad(self.config['vehicle_config']['max_steering_angle'])
        max_accel = self.config['vehicle_config']['max_accel']
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steering_angle, -max_accel]),
            high=np.array([max_steering_angle, max_accel]),
            dtype=np.float32
        )
        
        # Observation space: {vehicle state (5) + occupancy grid (256*256)}
        self.observation_space = gym.spaces.Dict({
          'vehicle_state': gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0., -1.]),
            high=np.array([np.inf, np.inf, np.pi, np.inf, 1.]),
            dtype=np.float32
          ),
          'occupancy_grid': gym.spaces.Box(
            low=0., high=1., shape=(256, 256), dtype=np.float32
          )
        })
        
        # Environment state
        self.state: Optional[State] = None
        self.goal = State(x=0., y=0., yaw=0., v=0., dir=1.)
        self.occupancy_grid: Optional[np.ndarray] = None
        self.step_count: int = 0
        
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
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates.
        
        Args:
            x: World x coordinate in meters.
            y: World y coordinate in meters.
            
        Returns:
            Grid coordinates (row, col).
        """
        grid_size = self.config['observation']['grid_size']
        resolution = self.config['observation']['grid_resolution']
        
        # Convert to grid coordinates (origin at center)
        grid_x = int((x + grid_size / 2) / resolution)
        grid_y = int((y + grid_size / 2) / resolution)
        
        # Return none if out of bounds
        if grid_x < 0 or grid_x >= grid_size or grid_y < 0 or grid_y >= grid_size:
            return None
        
        return grid_y, grid_x  # Return (row, col)
    
    def _get_occupancy_grid(self) -> np.ndarray:
        """Get occupancy grid observation from vehicle perspective.
                    
        Returns:
            Occupancy grid as array.
        """
        if self.occupancy_grid is not None:
            return self.occupancy_grid
        
        grid_size = self.config['observation']['grid_size']
        resolution = self.config['observation']['grid_resolution']
        
        # Initialize empty grid
        self.occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
                  
        return self.occupancy_grid
    
    def _check_collision(self, state: State) -> bool:
        """Check if vehicle collides with obstacles.
        
        Args:
            state: Vehicle state to check.
            
        Returns:
            True if collision detected.
        """        
        return False
    
    def _check_goal_reached(self, state: State) -> bool:
        """Check if vehicle has reached the goal.
        
        Args:
            state: Current vehicle state.
            
        Returns:
            True if goal is reached.
        """
        # Check position
        distance_threshold = self.config['env']['distance_threshold']
        distance_error = np.sqrt(state.x**2 + state.y**2)
        if distance_error > distance_threshold:
            return False
        
        # Check heading
        heading_threshold = self.config['env']['heading_threshold']
        heading_error = np.abs(mod2pi(state.yaw))
        if heading_error > heading_threshold:
            return False
        
        # Check velocity
        velocity_threshold = self.config['env']['velocity_threshold']
        velocity_error = np.abs(state.v)
        if velocity_error > velocity_threshold:
            return False
        
        return True
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation.
        
        Returns:
            Observation dictionary: {vehicle_state, occupancy_grid}.
        """
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Vehicle state: [x, y, yaw, v, dir]
        vehicle_state = np.array([
            self.state.x, self.state.y, self.state.yaw, self.state.v, self.state.dir
        ], dtype=np.float32)
        
        # Occupancy grid
        occupancy_grid = self._get_occupancy_grid()
        
        # Create observation dictionary
        observation = {
            'vehicle_state': vehicle_state,
            'occupancy_grid': occupancy_grid
        }
                
        return observation
    
    def _calculate_reward(self, state: State, action: Action, 
                         done: bool, truncated: bool) -> float:
        """Calculate reward for current state and action.
        
        Args:
            state: Current vehicle state.
            action: Action taken.
            done: Whether episode done.
            truncated: Whether episode truncated.
            
        Returns:
            Reward value.
        """
        # Compute Reeds-Shepp path
        radius = self.config['rs_path']['turning_radius']
        start = (state.x, state.y, state.yaw)
        goal = (0., 0., 0.)
        length = rs.path_length(start, goal, radius)

        # Path length penalty
        reward = length * self.config['reward']['path_length']

        # Success bonus
        if done:
            reward += self.config['reward']['success']

        # Truncated penalty
        if truncated:
            reward += self.config['reward']['truncated']
        
        return reward

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Clear trajectory if renderer exists
        if self.renderer is not None:
            self.renderer.clear_trajectory()
        
        # Generate random initial state without collision
        while True:
            # Random state
            val = self.config['env']['max_initial_position']
            x = np.random.uniform(-val, val)
            y = np.random.uniform(-val, val)
            
            val = deg2rad(self.config['env']['max_initial_heading'])
            yaw = np.random.uniform(-val, val)

            val = kph2mps(self.config['env']['max_initial_velocity'])
            v = np.random.uniform(-val, val)
            
            dir = np.random.choice([-1, 1])
            
            # Create state
            state = State(x=x, y=y, yaw=yaw, v=v, dir=dir)
            
            # Check if initial state is valid (no collision)
            if not self._check_collision(state):
                self.state = state
                break
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'vehicle_state': self.state,
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action array [steering_angle, acceleration].
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Get expert trajectory
        rs_path = self.expert.get_rs_path(self.state, self.goal)
        
        # Parse action
        max_steering_angle = deg2rad(self.config['vehicle_config']['max_steering_angle'])
        steering_angle = np.clip(action[0], -max_steering_angle, max_steering_angle)
        
        max_accel = self.config['vehicle_config']['max_accel']
        acceleration = np.clip(action[1], -max_accel, max_accel)
        
        # Create action object
        action = Action(delta=steering_angle, accel=acceleration)
        
        # Simulate next state
        next_state = simulate(
          state=self.state,
          action=action,
          dt=self.config['env']['dt'],
          wheelbase=self.config['vehicle_config']['wheelbase']
        )
        
        # Update current state
        self.state = next_state
        
        # Increment step counter
        self.step_count += 1
        
        # Check terminal condition
        done = False
        if self._check_goal_reached(self.state):
            done = True
                
        # Check collision or time limit
        truncated = False
        if self._check_collision(self.state) or self.step_count >= self.config['env']['max_steps']:
            truncated = True
        
        # Calculate reward
        reward = self._calculate_reward(self.state, action, done, truncated)
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info
        info = {
            'vehicle_state': self.state,
            'rs_path': rs_path,
            # 'expert_actions': expert_actions,
            'step_count': self.step_count,
        }
        
        return observation, reward, done, truncated, info
    
    def render(self, **kwargs):
        """Render the environment.
        
        Args:
            mode: Rendering mode ('human' for pygame window, 'rgb_array' for array).
            **kwargs: Additional arguments. (reward, done, truncated, expert trajectory)

        Returns:
            If mode is 'rgb_array', returns numpy array of the rendered frame.
            Otherwise returns None.
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
        reward, done, truncated, expert = kwargs.values()
        
        # Render
        self.renderer.render(
            state=self.state,
            step_count=self.step_count,
            reward=reward,
            done=done,
            truncated=truncated,
            expert=expert,
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
