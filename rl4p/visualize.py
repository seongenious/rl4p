"""Visualization module for RL4P project.

This module provides pygame-based visualization for parking simulation with
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


class ParkingVisualizer:
    """Pygame-based parking visualization with rollout functionality."""
    
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
        pygame.display.set_caption('RL4P: Parking Visualization')
        self.clock = pygame.time.Clock()
        
        # Colors - Fancy color scheme
        self.DARK_BG = (30, 30, 35)  # Dark gray background
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (100, 150, 255)  # Bright blue
        self.RED = (255, 100, 100)   # Bright red
        self.GREEN = (100, 255, 100) # Bright green
        self.DARK_GREEN = (50, 200, 50)
        self.GRAY = (180, 180, 180)  # Light gray for vehicle
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
        self.wheelbase = 2.5  # meters
        self.vehicle_length = 4.5  # meters
        self.vehicle_width = 1.8   # meters
        
        # Parking slot parameters
        self.slot_width = 5.2  # meters
        self.slot_height = 2.4  # meters
        self.goal_distance_from_front = 3.5  # meters from front of slot
        
        # Simulation state
        self.current_state = None
        self.goal_state = None
        self.trajectory = []
        self.rollout_trajectories = []
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
        
        # Button parameters
        self.button_width = 120
        self.button_height = 40
        self.button_margin = 20
        self.reset_button_rect = pygame.Rect(
            self.screen_width - self.button_width - self.button_margin,
            self.button_margin,
            self.button_width,
            self.button_height
        )
        self.button_hover = False
        
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
    
    def get_goal_position(self) -> Tuple[float, float]:
        """Get the goal position (rear wheel center).
        
        Returns:
            Goal position (x, y) in world coordinates.
        """
        # Goal is 3.5m from front of parking slot
        goal_x = (self.slot_width / 2) - self.goal_distance_from_front
        goal_y = 0.0
        return goal_x, goal_y
    
    def get_vehicle_rear_center(self, state: State) -> Tuple[float, float]:
        """Get the rear wheel center position of the vehicle.
        
        Args:
            state: Vehicle state (center position).
            
        Returns:
            Rear wheel center position (x, y) in world coordinates.
        """
        # Vehicle center to rear wheel center offset
        rear_offset = self.wheelbase / 2
        
        # Calculate rear wheel center
        rear_x = state.x - rear_offset * np.cos(state.yaw)
        rear_y = state.y - rear_offset * np.sin(state.yaw)
        
        return rear_x, rear_y
    
    def check_parking_success(self, state: State) -> bool:
        """Check if vehicle has successfully parked.
        
        Args:
            state: Current vehicle state.
            
        Returns:
            True if parking is successful.
        """
        # Get rear wheel center position
        rear_x, rear_y = self.get_vehicle_rear_center(state)
        
        # Get goal position
        goal_x, goal_y = self.get_goal_position()
        
        # Check if rear wheel center is at goal position
        position_error = np.sqrt((rear_x - goal_x)**2 + (rear_y - goal_y)**2)
        position_threshold = 0.5  # meters
        
        # Check velocity (should be near zero)
        low_velocity = abs(state.v) < 0.5  # m/s
        
        # Check heading (should be near zero)
        heading_error = abs(state.yaw) % (2 * np.pi)
        heading_error = min(heading_error, 2 * np.pi - heading_error)
        good_heading = heading_error < np.pi / 6  # 30 degrees
        
        return position_error < position_threshold and low_velocity and good_heading
    
    def draw_parking_slot(self):
        """Draw the parking slot around the goal."""
        # Convert to screen coordinates
        left = self.center_x - (self.slot_width / 2) * self.scale
        top = self.center_y - (self.slot_height / 2) * self.scale
        width = self.slot_width * self.scale
        height = self.slot_height * self.scale
        
        # Draw parking slot with very thick white border
        slot_color = self.DARK_GREEN if self.parking_success else self.WHITE
        border_thickness = 8  # Much thicker border
        
        # Draw thick border
        pygame.draw.rect(self.screen, slot_color, (left, top, width, height), border_thickness)
        
        # Draw additional inner border for better visibility
        inner_border = 2
        pygame.draw.rect(self.screen, self.BLACK, (left + border_thickness//2, top + border_thickness//2, 
                                                  width - border_thickness, height - border_thickness), inner_border)
        
        # Draw goal point (rear wheel center position)
        goal_x, goal_y = self.get_goal_position()
        goal_screen_x, goal_screen_y = self.world_to_screen(goal_x, goal_y)
        
        goal_color = self.YELLOW if self.parking_success else self.YELLOW
        pygame.draw.circle(self.screen, goal_color, (goal_screen_x, goal_screen_y), 12)
        pygame.draw.circle(self.screen, self.BLACK, (goal_screen_x, goal_screen_y), 12, 3)
    
    def draw_vehicle(self, state: State, alpha: int = 200):
        """Draw vehicle as a 3D bounding box.
        
        Args:
            state: Vehicle state.
            alpha: Transparency (0-255).
        """
        # Create vehicle surface with transparency
        vehicle_surface = pygame.Surface((self.vehicle_length * self.scale, self.vehicle_width * self.scale))
        vehicle_surface.set_alpha(alpha)
        vehicle_surface.fill(self.GRAY)
        
        # Add vehicle details
        # Draw wheels
        wheel_radius = 0.3 * self.scale
        wheel_color = self.DARK_GRAY
        
        # Front wheels (at front of vehicle)
        front_wheel_offset = self.wheelbase / 2
        front_left_x = (self.vehicle_length / 2 - front_wheel_offset) * self.scale
        front_left_y = (self.vehicle_width / 2 - 0.3) * self.scale
        front_right_x = (self.vehicle_length / 2 - front_wheel_offset) * self.scale
        front_right_y = (self.vehicle_width / 2 + 0.3) * self.scale
        
        pygame.draw.circle(vehicle_surface, wheel_color, (int(front_left_x), int(front_left_y)), int(wheel_radius))
        pygame.draw.circle(vehicle_surface, wheel_color, (int(front_right_x), int(front_right_y)), int(wheel_radius))
        
        # Rear wheels (at rear of vehicle)
        rear_left_x = (self.vehicle_length / 2) * self.scale
        rear_left_y = (self.vehicle_width / 2 - 0.3) * self.scale
        rear_right_x = (self.vehicle_length / 2) * self.scale
        rear_right_y = (self.vehicle_width / 2 + 0.3) * self.scale
        
        pygame.draw.circle(vehicle_surface, wheel_color, (int(rear_left_x), int(rear_left_y)), int(wheel_radius))
        pygame.draw.circle(vehicle_surface, wheel_color, (int(rear_right_x), int(rear_right_y)), int(wheel_radius))
        
        # Add vehicle body outline
        pygame.draw.rect(vehicle_surface, self.BLACK, (0, 0, self.vehicle_length * self.scale, self.vehicle_width * self.scale), 2)
        
        # Position vehicle
        x, y = self.world_to_screen(state.x, state.y)
        vehicle_rect = vehicle_surface.get_rect(center=(x, y))
        
        # Rotate vehicle
        rotated_surface = pygame.transform.rotate(vehicle_surface, -np.degrees(state.yaw))
        rotated_rect = rotated_surface.get_rect(center=(x, y))
        
        self.screen.blit(rotated_surface, rotated_rect)
        
        # Draw rear wheel center indicator
        rear_x, rear_y = self.get_vehicle_rear_center(state)
        rear_screen_x, rear_screen_y = self.world_to_screen(rear_x, rear_y)
        pygame.draw.circle(self.screen, self.ORANGE, (rear_screen_x, rear_screen_y), 5)
        pygame.draw.circle(self.screen, self.BLACK, (rear_screen_x, rear_screen_y), 5, 1)
    
    def draw_trajectory(self, trajectory: List[State], color: Tuple[int, int, int] = (100, 150, 255)):
        """Draw trajectory points and lines.
        
        Args:
            trajectory: List of state objects.
            color: RGB color tuple.
        """
        if len(trajectory) < 2:
            return
        
        # Draw trajectory points and lines
        points = []
        for state in trajectory:
            x, y = self.world_to_screen(state.x, state.y)
            points.append((x, y))
            pygame.draw.circle(self.screen, color, (x, y), 3)
        
        # Draw lines connecting points
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 3)
    
    def draw_rollout_trajectories(self):
        """Draw all rollout trajectories."""
        for i, trajectory in enumerate(self.rollout_trajectories):
            # Use different colors for different rollouts
            colors = [self.BLUE, self.PURPLE, self.ORANGE, self.GREEN]
            color = colors[i % len(colors)]
            alpha = max(50, 255 - i * 10)  # Fade out later rollouts
            
            # Create a surface for this trajectory with alpha
            if len(trajectory) > 1:
                points = []
                for state in trajectory:
                    x, y = self.world_to_screen(state.x, state.y)
                    points.append((x, y))
                
                if len(points) > 1:
                    # Draw with alpha
                    line_surface = pygame.Surface((self.screen_width, self.screen_height))
                    line_surface.set_alpha(alpha)
                    line_surface.fill((0, 0, 0, 0))
                    pygame.draw.lines(line_surface, color, False, points, 2)
                    self.screen.blit(line_surface, (0, 0))
    
    def draw_buttons(self):
        """Draw UI buttons."""
        # Check if mouse is hovering over reset button
        mouse_pos = pygame.mouse.get_pos()
        self.button_hover = self.reset_button_rect.collidepoint(mouse_pos)
        
        # Reset button with gradient effect
        button_color = self.BUTTON_HOVER if self.button_hover else self.DARK_GRAY
        pygame.draw.rect(self.screen, button_color, self.reset_button_rect)
        pygame.draw.rect(self.screen, self.WHITE, self.reset_button_rect, 2)
        
        # Button text
        font = pygame.font.Font(None, 28)
        text = font.render("Reset", True, self.WHITE)
        text_rect = text.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(text, text_rect)
    
    def draw_info(self):
        """Draw simulation information."""
        font = pygame.font.Font(None, 26)
        
        # Simulation time
        time_text = f"Time: {self.simulation_time:.1f}s"
        time_surface = font.render(time_text, True, self.WHITE)
        self.screen.blit(time_surface, (10, 10))
        
        # Vehicle state
        if self.current_state:
            state_text = f"Pos: ({self.current_state.x:.1f}, {self.current_state.y:.1f})"
            state_surface = font.render(state_text, True, self.WHITE)
            self.screen.blit(state_surface, (10, 40))
            
            vel_text = f"Vel: {self.current_state.v:.1f} m/s"
            vel_surface = font.render(vel_text, True, self.WHITE)
            self.screen.blit(vel_surface, (10, 70))
            
            yaw_text = f"Yaw: {np.degrees(self.current_state.yaw):.1f}°"
            yaw_surface = font.render(yaw_text, True, self.WHITE)
            self.screen.blit(yaw_surface, (10, 100))
        
        # Rollout generation time
        rollout_time_text = f"Rollout Time: {self.rollout_generation_time:.3f}s"
        rollout_time_surface = font.render(rollout_time_text, True, self.WHITE)
        self.screen.blit(rollout_time_surface, (10, 130))
        
        # Rollout count
        rollout_text = f"Rollouts: {len(self.rollout_trajectories)}"
        rollout_surface = font.render(rollout_text, True, self.WHITE)
        self.screen.blit(rollout_surface, (10, 160))
        
        # Parking status
        status_text = "PARKED!" if self.parking_success else "Parking..."
        status_color = self.GREEN if self.parking_success else self.YELLOW
        status_surface = font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (10, 190))
        
        # Instructions
        instructions = [
            "Controls:",
            "SPACE - Start/Stop",
            "R - Reset",
            "Click Reset Button"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_surface = font.render(instruction, True, self.LIGHT_GRAY)
            self.screen.blit(inst_surface, (10, 230 + i * 25))
    
    def perform_rollout(self, initial_state: State, steps: int = 20) -> List[State]:
        """Perform rollout from initial state.
        
        Args:
            initial_state: Starting state for rollout.
            steps: Number of steps to simulate.
            
        Returns:
            List of states representing the trajectory.
        """
        trajectory = [initial_state]
        current_state = initial_state
        
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
            
            # Simulate next state
            next_state = simulate(
                current_state, action_obj, self.env.dt, self.env.wheelbase,
                self.env.max_speed, self.env.max_accel, self.env.max_steering_angle
            )
            
            trajectory.append(next_state)
            current_state = next_state
        
        return trajectory
    
    def generate_rollouts(self):
        """Generate multiple rollouts from current state."""
        start_time = time.time()
        
        self.rollout_trajectories = []
        
        for _ in range(20):  # Generate 20 rollouts
            trajectory = self.perform_rollout(self.current_state)
            self.rollout_trajectories.append(trajectory)
        
        self.rollout_generation_time = time.time() - start_time
    
    def reset_simulation(self):
        """Reset the simulation with new random initial state."""
        # Clear screen immediately
        self.screen.fill(self.DARK_BG)
        pygame.display.flip()
        
        # Reset simulation state
        self.current_state, self.goal_state = self.env.reset()
        self.trajectory = [self.current_state]
        self.rollout_trajectories = []
        self.simulation_time = 0
        self.simulation_running = False
        self.last_update_time = time.time()
        self.time_since_last_step = 0
        self.parking_success = False
        self.rollout_generation_time = 0.0
        
        # Generate rollouts for new state
        self.generate_rollouts()
    
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
            # Get the first action from rollouts
            if self.rollout_trajectories and len(self.rollout_trajectories[0]) > 1:
                # Use the first rollout's first action
                rollout_idx = 0
                step_idx = 1  # First step after initial state
                
                if step_idx < len(self.rollout_trajectories[rollout_idx]):
                    # Get the action that led to this state
                    prev_state = self.rollout_trajectories[rollout_idx][step_idx - 1]
                    next_state = self.rollout_trajectories[rollout_idx][step_idx]
                    
                    # Create observation
                    obs = jnp.array([[
                        prev_state.x, prev_state.y, prev_state.yaw, 
                        prev_state.v, prev_state.dir
                    ]], dtype=jnp.float32)
                    
                    # Get action from policy
                    action = policy_inference(self.actor_def, self.actor_params, obs)[0]
                    
                    # Convert to Action object
                    action_obj = Action(delta=float(action[0]), accel=float(action[1]))
                    
                    # Simulate next state
                    next_state = simulate(
                        self.current_state, action_obj, self.env.dt, self.env.wheelbase,
                        self.env.max_speed, self.env.max_accel, self.env.max_steering_angle
                    )
                    
                    self.current_state = next_state
                    self.trajectory.append(next_state)
                    
                    # Check parking success
                    self.parking_success = self.check_parking_success(self.current_state)
                    
                    # Regenerate rollouts for new state (but don't block rendering)
                    # We'll do this in a separate thread or optimize later
                    if len(self.trajectory) % 5 == 0:  # Regenerate every 5 steps
                        self.generate_rollouts()
            
            # Update simulation time
            self.simulation_time += self.simulation_dt
            self.time_since_last_step = 0
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.reset_button_rect.collidepoint(event.pos):
                        self.reset_simulation()
            
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
        
        # Draw rollout trajectories
        self.draw_rollout_trajectories()
        
        # Draw current trajectory
        self.draw_trajectory(self.trajectory, self.RED)
        
        # Draw vehicle
        if self.current_state:
            self.draw_vehicle(self.current_state)
        
        # Draw UI elements
        self.draw_buttons()
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

    # Create and run visualizer
    visualizer = ParkingVisualizer(env, actor_def, actor_params)
    visualizer.run()


if __name__ == '__main__':
    main()