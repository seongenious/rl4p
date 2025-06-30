"""Simulator module for RL4P project.

This module provides pygame-based simulator for parking simulation with
rollout functionality and interactive controls.
"""

import os
import argparse 
import jax 
import jax.numpy as jnp 
import numpy as np
import pygame 
from typing import Tuple, List, Optional
import time

from flax.training import checkpoints
from flax import linen as nn

from envs.freespace_env import FreespaceEnv
from envs.kinematic_model import simulate
from envs.datatypes import State, Action
from envs.reward import reward_fn
from models.sac_networks import PolicyNetwork
from utils.config import load_yaml_config
from utils.unit import mod2pi


def policy_inference(actor_def: PolicyNetwork, 
                    actor_params: dict, 
                    obs: jnp.ndarray) -> jnp.ndarray:
    """Perform policy inference to get deterministic actions.
    
    Args:
        actor_def: Actor network definition.
        actor_params: Actor network parameters.
        obs: Observations, shape: (batch_size, obs_dim).
        
    Returns:
        Deterministic actions, shape: (batch_size, action_dim).
    """
    mu, _ = actor_def.apply(actor_params, obs)
    return jnp.tanh(mu)


class ParkingSimulator:
    """Pygame-based parking simulator with rollout functionality."""
    
    def __init__(self, env: FreespaceEnv, actor_def: PolicyNetwork, actor_params: dict):
        """Initialize the visualizer.
        
        Args:
            env: Environment instance.
            actor_def: Actor network definition.
            actor_params: Actor network parameters.
        """
        self.env = env
        self.actor_def = actor_def
        self.actor_params = actor_params
        
        # Pygame setup
        pygame.init()
        self.screen_width = 1000
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Parking Simulator')
        self.clock = pygame.time.Clock()
        
        # Colors - Fancy color scheme
        self.DARK_BG = (44, 44, 44)  # Dark gray background
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (100, 150, 255)  # Bright blue
        self.RED = (255, 100, 100)   # Bright red
        self.GREEN = (100, 255, 100) # Bright green
        self.DARK_GREEN = (50, 200, 50)
        self.GRAY = (180, 180, 180, 100)  # Light gray for vehicle
        self.LIGHT_GRAY = (220, 220, 220)  # Very light gray
        self.DARK_GRAY = (80, 80, 80)
        self.BUTTON_HOVER = (120, 120, 120)
        self.YELLOW = (255, 255, 100)  # Bright yellow
        self.ORANGE = (255, 180, 100)  # Orange
        self.PURPLE = (200, 100, 255)  # Purple
        
        # Display parameters
        self.scale = 20  # pixels per meter
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        
        # Vehicle parameters
        self.wheelbase = self.env.config['vehicle_config']['wheelbase']
        self.cg_to_front = self.env.config['vehicle_config']['cg_to_front']
        self.cg_to_rear = self.env.config['vehicle_config']['cg_to_rear']
        
        # Parking slot parameters
        self.slot_cg_to_front = 3.5  # meters
        self.slot_cg_to_rear = 1.5  # meters
        self.slot_width = 2.4  # meters
        
        # Simulation state
        self.current_state = None
        self.current_action = None
        self.goal_state = None
        self.history = []
        self.rollout_trajectory = None
        self.simulation_time = 0
        self.max_simulation_time = 30  # seconds
        self.simulation_running = False
        self.last_update_time = time.time()
        self.simulation_dt = 0.2  # seconds per simulation step
        self.time_since_last_step = 0
        
        # Parking success state
        self.parking_success = False
        
        # Performance tracking
        self.rollout_generation_time = 0.0
                
        # Initialize environment
        self.reset_simulation()
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.
        
        Args:
            x: World x coordinate in meters.
            y: World y coordinate in meters.
            
        Returns:
            Screen coordinates (x, y).
        """
        screen_x = int(self.center_x + x * self.scale)
        screen_y = int(self.center_y - y * self.scale)  # Flip y for display
        return screen_x, screen_y

    def draw_polygon(self, center, cg_to_front, cg_to_rear, width, color, thickness):
        """Draw a polygon representing the parking slot or vehicle.
        
        Args:
            center: Center position (x, y, yaw) in world coordinates.
            cg_to_front: Distance from center to front in meters.
            cg_to_rear: Distance from center to rear in meters.
            width: Width of the polygon in meters.
            color: RGB color tuple.
            thickness: Line thickness (0 for filled polygon).
        """
        cx, cy, cyaw = center
        
        # Define corners relative to center (before rotation)
        corners = [
            (cg_to_front, width / 2),
            (cg_to_front, -width / 2),
            (-cg_to_rear, -width / 2),
            (-cg_to_rear, width / 2),
        ]
        
        # Apply rotation and translation
        rotated_corners = []
        for x, y in corners:
            # Apply rotation
            rotated_x = x * np.cos(cyaw) - y * np.sin(cyaw)
            rotated_y = x * np.sin(cyaw) + y * np.cos(cyaw)
            
            # Add center offset
            world_x = cx + rotated_x
            world_y = cy + rotated_y
            
            # Convert to screen coordinates
            screen_x, screen_y = self.world_to_screen(world_x, world_y)
            rotated_corners.append((screen_x, screen_y))
        
        # Draw the polygon
        pygame.draw.polygon(self.screen, color, rotated_corners, thickness)
    
    def check_parking_success(self, state: State) -> bool:
        """Check if vehicle has successfully parked.
        
        Args:
            state: Current vehicle state.
            
        Returns:
            True if parking is successful.
        """        
        # Check if rear wheel center is at goal position
        position_error = np.sqrt(state.x**2 + state.y**2)
        
        return position_error < self.env.config['reward']['path_resolution'] and \
            np.abs(state.v) < self.env.config['reward']['velocity_threshold'] and \
            mod2pi(state.yaw) < self.env.config['reward']['heading_threshold']
    
    def draw_parking_slot(self):
        """Draw the parking slot around the goal."""
        slot_color = self.GREEN if self.parking_success else self.WHITE
        slot_thickness = 2
        
        self.draw_polygon(
            (0, 0, 0),  # Goal position in world coordinates
            self.slot_cg_to_front, 
            self.slot_cg_to_rear, 
            self.slot_width, 
            slot_color, 
            slot_thickness
        )
    
    def draw_vehicle(self, state: State, alpha: int = 200):
        """Draw vehicle as a 3D bounding box.
        
        Args:
            state: Vehicle state.
            alpha: Transparency (0-255).
        """
        # Create vehicle surface with transparency
        self.draw_polygon(
            (state.x, state.y, state.yaw), 
            self.env.config['vehicle_config']['cg_to_front'], 
            self.env.config['vehicle_config']['cg_to_rear'], 
            self.env.config['vehicle_config']['width'], 
            self.GRAY, 
            0  # 0 for filled polygon,
        )
        
        # Draw rear wheel center indicator
        rear_screen_x, rear_screen_y = self.world_to_screen(state.x, state.y)
        pygame.draw.circle(self.screen, self.BLACK, (rear_screen_x, rear_screen_y), 5)
    
    def draw_history(self):
        """Draw history points and lines."""        
        if self.history is None:
            return
        
        points = []
        for state in self.history:
            x, y = self.world_to_screen(state.x, state.y)
            points.append((x, y))
            pygame.draw.circle(self.screen, self.RED, (x, y), 5)
        
        # Draw lines connecting points
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.RED, False, points, 3)
    
    def draw_rollout_trajectory(self):
        """Draw all rollout trajectories."""
        if self.rollout_trajectory is None:
            return
        
        points = []
        for state in self.rollout_trajectory:
            x, y = self.world_to_screen(state.x, state.y)
            points.append((x, y))
            pygame.draw.circle(self.screen, self.BLUE, (x, y), 5)
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.BLUE, False, points, 3)
        
    def draw_info(self):
        """Draw simulation information."""
        font = pygame.font.Font(None, 24)
        
        # Simulation time
        time_text = f"Simulation time: {self.simulation_time:.1f}s"
        time_surface = font.render(time_text, True, self.WHITE)
        self.screen.blit(time_surface, (10, 10))
        
        # Rollout generation time
        rollout_time_text = f"Rollout time: {self.rollout_generation_time:.3f}s"
        rollout_time_surface = font.render(rollout_time_text, True, self.WHITE)
        self.screen.blit(rollout_time_surface, (10, 35))

        # Vehicle state
        if self.current_state:
            state_text = "[State]"
            state_surface = font.render(state_text, True, self.WHITE)
            self.screen.blit(state_surface, (10, 70))

            pos_text = f"    Position: ({self.current_state.x:.1f}, {self.current_state.y:.1f})"
            pos_surface = font.render(pos_text, True, self.WHITE)
            self.screen.blit(pos_surface, (10, 95))
            
            yaw_text = f"    Yaw: {np.degrees(self.current_state.yaw):.1f}°"
            yaw_surface = font.render(yaw_text, True, self.WHITE)
            self.screen.blit(yaw_surface, (10, 120))

            vel_text = f"    Velocity: {self.current_state.v:.1f} m/s"
            vel_surface = font.render(vel_text, True, self.WHITE)
            self.screen.blit(vel_surface, (10, 145))
            
            dir_text = f"    Direction: {self.current_state.dir:.1f}"
            dir_surface = font.render(dir_text, True, self.WHITE)
            self.screen.blit(dir_surface, (10, 170))

        # Action
        if self.current_action:
            action_text = "[Action]"
            action_surface = font.render(action_text, True, self.WHITE)
            self.screen.blit(action_surface, (10, 195))

            delta_text = f"    Steering angle: {self.env.max_steering_angle * self.current_action.delta:.1f}°"
            delta_surface = font.render(delta_text, True, self.WHITE)
            self.screen.blit(delta_surface, (10, 220))

            accel_text = f"    Acceleration: {self.env.max_accel * self.current_action.accel:.1f} m/s^2"
            accel_surface = font.render(accel_text, True, self.WHITE)
            self.screen.blit(accel_surface, (10, 245))
        
        # Instructions
        instructions = [
            "[Control]",
            "   Start/Stop | SPACE",
            "   Reset | R",
        ]
        
        for i, instruction in enumerate(instructions):
            inst_surface = font.render(instruction, True, self.WHITE)
            self.screen.blit(inst_surface, (10, 280 + i * 25))
                
        # Parking status
        status_text = "Success" if self.parking_success else "Parking ..."
        status_color = self.GREEN if self.parking_success else self.YELLOW
        status_surface = font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (10, 365))
        
        
    
    def perform_rollout(self, initial_state: State, steps: int = 20) -> List[State]:
        """Perform rollout from initial state.
        
        Args:
            initial_state: Starting state for rollout.
            steps: Number of steps to simulate.
            
        Returns:
            List of states representing the trajectory.
        """
        start_time = time.time()

        trajectory = [initial_state]
        current_state = initial_state
        self.current_action = None
        
        for _ in range(steps):
            # Create observation
            obs = jnp.array([[
                current_state.x, current_state.y, current_state.yaw, 
                current_state.v, current_state.dir
            ]], dtype=jnp.float32)
            
            # Get action from policy
            action = policy_inference(self.actor_def, self.actor_params, obs)[0]
            
            # Convert to Action object
            action_obj = Action(delta=float(action[0]), accel=float(action[1]))

            if self.current_action is None:
                self.current_action = action_obj
            
            # Simulate next state
            next_state = simulate(
                current_state, action_obj, self.env.dt, self.env.wheelbase,
                self.env.max_speed, self.env.max_accel, self.env.max_steering_angle
            )
            
            trajectory.append(next_state)
            current_state = next_state
        
        self.rollout_generation_time = time.time() - start_time
        self.rollout_trajectory = trajectory
    
    def reset_simulation(self):
        """Reset the simulation with new random initial state."""
        # Clear screen immediately
        self.screen.fill(self.WHITE)
        pygame.display.flip()
        
        # Reset simulation state
        self.current_state, self.goal_state = self.env.reset()
        self.history = [self.current_state]
        self.simulation_time = 0
        self.simulation_running = False
        self.last_update_time = time.time()
        self.time_since_last_step = 0
        self.parking_success = False
        self.rollout_generation_time = 0.0
        self.rollout_trajectory = None
        
        # Generate rollouts for new state
        self.perform_rollout(self.current_state)
    
    def update_simulation(self, dt: float):
        """Update simulation state.
        
        Args:
            dt: Time step in seconds.
        """
        if not self.simulation_running or self.simulation_time >= self.max_simulation_time:
            return
        
        # Accumulate time
        self.time_since_last_step += dt
        
        # Check if it's time for a simulation step
        if self.time_since_last_step >= self.simulation_dt:
            # Check parking success
            self.parking_success = self.check_parking_success(self.current_state)

            # Generate rollouts for new state
            self.perform_rollout(self.current_state)

            # Simulate next state
            next_state = simulate(
                self.current_state, self.current_action, self.env.dt, self.env.wheelbase,
                self.env.max_speed, self.env.max_accel, self.env.max_steering_angle
            )

            self.current_state = next_state
            self.history.append(next_state)

            # Update simulation time
            self.simulation_time += self.simulation_dt
            self.time_since_last_step = 0
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle simulation
                    self.simulation_running = not self.simulation_running
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.reset_simulation()
        
        return True
    
    def render(self):
        """Render the current state."""
        # Clear screen with dark background
        self.screen.fill(self.DARK_BG)
        
        # Draw parking slot
        self.draw_parking_slot()

        # Draw vehicle
        if self.current_state:
            self.draw_vehicle(self.current_state)

        # Draw rollout trajectories
        self.draw_rollout_trajectory()
        
        # Draw current trajectory
        self.draw_history()
                
        # Draw UI elements
        self.draw_info()
                
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main simulation loop."""
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update simulation
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            self.update_simulation(dt)
            
            # Render
            self.render()
            
            # Cap frame rate
            self.clock.tick(30)
        
        pygame.quit()


def main() -> None:
    """Main function for parking visualization."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                       help='Path to SAC checkpoint directory')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to YAML config file')
    args = parser.parse_args()

    # Load config
    config = load_yaml_config(args.config)
    obs_dim = 5
    act_dim = 2
    hidden_dims = [256, 256]

    # Prepare actor network
    actor_def = PolicyNetwork(action_dim=act_dim, hidden_dims=hidden_dims)
    dummy_input = jnp.zeros((1, obs_dim))
    actor_params = actor_def.init(jax.random.PRNGKey(0), dummy_input)['params']

    # Load checkpoint
    ckpt = checkpoints.restore_checkpoint(args.ckpt_dir, target=None)
    actor_params = ckpt['actor_state']['params']

    # Initialize environment
    env = FreespaceEnv(config)

    # Create and run simulator
    simulator = ParkingSimulator(env, actor_def, actor_params)
    simulator.run()


if __name__ == '__main__':
    main()