"""Renderer module for parking environment.

This module provides pygame-based visualization for the parking environment
with vehicle, parking slot, and trajectory visualization.
"""

import pygame
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from envs.datatypes import State, Action


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    
    # Screen parameters
    screen_width: int = 1000
    screen_height: int = 800
    
    # Display parameters
    scale: float = 20.0  # pixels per meter
    center_x: int = 500  # screen center x
    center_y: int = 400  # screen center y
    
    # Colors
    DARK_BG: Tuple[int, int, int] = (44, 44, 44)  # Dark gray background
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    BLUE: Tuple[int, int, int] = (100, 150, 255)  # Bright blue
    RED: Tuple[int, int, int] = (255, 100, 100)   # Bright red
    GREEN: Tuple[int, int, int] = (100, 255, 100) # Bright green
    DARK_GREEN: Tuple[int, int, int] = (50, 200, 50)
    GRAY: Tuple[int, int, int] = (180, 180, 180)  # Light gray for vehicle
    LIGHT_GRAY: Tuple[int, int, int] = (220, 220, 220)  # Very light gray
    DARK_GRAY: Tuple[int, int, int] = (80, 80, 80)
    YELLOW: Tuple[int, int, int] = (255, 255, 100)  # Bright yellow
    ORANGE: Tuple[int, int, int] = (255, 180, 100)  # Orange
    PURPLE: Tuple[int, int, int] = (200, 100, 255)  # Purple


class ParkingRenderer:
    """Pygame-based renderer for parking environment."""
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """Initialize the renderer.
        
        Args:
            config: Renderer configuration. If None, uses default config.
        """
        self.config = config or RenderConfig()
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.screen_width, self.config.screen_height))
        pygame.display.set_caption('Parking Simulator')
        self.clock = pygame.time.Clock()
        
        # Vehicle parameters (default values, can be overridden)
        self.wheelbase = 2.7         # meters
        self.front_overhang = 0.8    # meters
        self.rear_overhang = 1.0     # meters
        self.vehicle_width = 1.8     # meters
                
        # Parking slot parameters
        self.slot_cg_to_front = 3.8  # meters
        self.slot_cg_to_rear = 1.2   # meters
        self.slot_width = 2.4        # meters
        
        # Trajectory history
        self.trajectory: List[State] = []
        
        # Font for text rendering
        self.font = pygame.font.Font(None, 24)
    
    def set_vehicle_parameters(self, wheelbase: float, 
            front_overhang: float, rear_overhang: float, width: float):
        """Set vehicle parameters for rendering.
        
        Args:
            wheelbase: Vehicle wheelbase in meters.
            front_overhang: Distance from center of gravity to front of vehicle in meters.
            rear_overhang: Distance from center of gravity to rear of vehicle in meters.
            width: Vehicle width in meters.
        """
        self.wheelbase = wheelbase
        self.front_overhang = front_overhang
        self.rear_overhang = rear_overhang
        self.vehicle_width = width
        
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.
        
        Args:
            x: World x coordinate in meters.
            y: World y coordinate in meters.
            
        Returns:
            Screen coordinates (x, y).
        """
        screen_x = int(self.config.center_x + x * self.config.scale)
        screen_y = int(self.config.center_y - y * self.config.scale)  # Flip y for display
        return screen_x, screen_y
    
    def draw_polygon(self, center: Tuple[float, float, float], 
                    cg_to_front: float, cg_to_rear: float, width: float, color: Tuple[int, int, int], 
                    thickness: int = 0):
        """Draw a polygon representing vehicle or parking slot.
        
        Args:
            center: Center position (x, y, yaw) in world coordinates.
            cg_to_front: Distance from center of gravity to front of vehicle in meters.
            cg_to_rear: Distance from center of gravity to rear of vehicle in meters.
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
        if thickness == 0:
            pygame.draw.polygon(self.screen, color, rotated_corners)
        else:
            pygame.draw.polygon(self.screen, color, rotated_corners, thickness)
    
    def draw_vehicle(self, state: State, color: Optional[Tuple[int, int, int]] = None):
        """Draw vehicle as a polygon.
        
        Args:
            state: Vehicle state.
            color: Vehicle color. If None, uses default gray.
        """
        if color is None:
            color = self.config.GRAY
        
        # Draw vehicle body
        self.draw_polygon(
            (state.x, state.y, state.yaw),
            self.wheelbase + self.front_overhang,
            self.rear_overhang,
            self.vehicle_width,
            color,
            0  # Filled polygon
        )
        
        # Draw vehicle outline
        self.draw_polygon(
            (state.x, state.y, state.yaw),
            self.wheelbase + self.front_overhang,
            self.rear_overhang,
            self.vehicle_width,
            self.config.BLACK,
            2  # Outline
        )
        
        # Draw rear wheel center indicator
        rear_screen_x, rear_screen_y = self.world_to_screen(state.x, state.y)
        pygame.draw.circle(self.screen, self.config.ORANGE, (rear_screen_x, rear_screen_y), 5)
        pygame.draw.circle(self.screen, self.config.BLACK, (rear_screen_x, rear_screen_y), 5, 1)
    
    def draw_parking_slot(self, success: bool = False):
        """Draw the parking slot around the goal.
        
        Args:
            success: Whether parking was successful.
        """
        color = self.config.GREEN if success else self.config.WHITE
        thickness = 2
        
        self.draw_polygon(
            (0, 0, 0),  # Goal position in world coordinates
            self.slot_cg_to_front,
            self.slot_cg_to_rear,
            self.slot_width,
            color,
            thickness
        )
        
        # Draw goal point
        goal_screen_x, goal_screen_y = self.world_to_screen(0, 0)
        pygame.draw.circle(self.screen, self.config.RED, (goal_screen_x, goal_screen_y), 8)
        pygame.draw.circle(self.screen, self.config.BLACK, (goal_screen_x, goal_screen_y), 8, 2)
    
    def draw_trajectory(self, trajectory: List[State], color: Optional[Tuple[int, int, int]] = None):
        """Draw trajectory points and lines.
        
        Args:
            trajectory: List of state objects.
            color: Trajectory color. If None, uses default blue.
        """
        if color is None:
            color = self.config.BLUE
        
        if len(trajectory) < 2:
            return
        
        # Draw trajectory points and lines
        points = []
        for state in trajectory:
            x, y = self.world_to_screen(state.x, state.y)
            points.append((x, y))
            pygame.draw.circle(self.screen, color, (x, y), 2)
        
        # Draw lines connecting points
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 1)
    
    def draw_info(self, state: State, step_count: int, reward: float, 
                  done: bool, truncated: bool, rs_path: List[State]):
        """Draw simulation information.
        
        Args:
            state: Current vehicle state.
            step_count: Current step count.
            reward: Current reward.
            done: Whether episode is done.
            truncated: Whether episode was truncated.
            rs_path: Reeds-Shepp path.
        """
        # Background for info panel
        info_rect = pygame.Rect(10, 10, 300, 200)
        pygame.draw.rect(self.screen, self.config.DARK_GRAY, info_rect)
        pygame.draw.rect(self.screen, self.config.WHITE, info_rect, 2)
        
        # Vehicle state
        state_text = f"Position: ({state.x:.2f}, {state.y:.2f})"
        state_surface = self.font.render(state_text, True, self.config.WHITE)
        self.screen.blit(state_surface, (20, 20))
        
        yaw_text = f"Yaw: {np.degrees(state.yaw):.1f}°"
        yaw_surface = self.font.render(yaw_text, True, self.config.WHITE)
        self.screen.blit(yaw_surface, (20, 45))
        
        vel_text = f"Velocity: {state.v:.2f} m/s"
        vel_surface = self.font.render(vel_text, True, self.config.WHITE)
        self.screen.blit(vel_surface, (20, 70))
        
        dir_text = f"Direction: {state.dir}"
        dir_surface = self.font.render(dir_text, True, self.config.WHITE)
        self.screen.blit(dir_surface, (20, 95))

        # Reeds-Shepp path
        self.draw_trajectory(rs_path, self.config.PURPLE)
        
        # Episode info
        step_text = f"Step: {step_count}"
        step_surface = self.font.render(step_text, True, self.config.WHITE)
        self.screen.blit(step_surface, (20, 120))
        
        reward_text = f"Reward: {reward:.2f}"
        reward_surface = self.font.render(reward_text, True, self.config.WHITE)
        self.screen.blit(reward_surface, (20, 145))
        
        # Status
        if done:
            status_text = "SUCCESS!"
            status_color = self.config.GREEN
        elif truncated:
            status_text = "TIMEOUT!"
            status_color = self.config.YELLOW
        else:
            status_text = "PARKING..."
            status_color = self.config.WHITE
        
        status_surface = self.font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (20, 170))
    
    def draw_grid(self, grid_size: int = 20):
        """Draw reference grid.
        
        Args:
            grid_size: Grid spacing in meters.
        """
        grid_color = (60, 60, 60)
        
        # Draw vertical lines
        for x in range(-grid_size * 5, grid_size * 6, grid_size):
            start_x, start_y = self.world_to_screen(x, -grid_size * 5)
            end_x, end_y = self.world_to_screen(x, grid_size * 5)
            pygame.draw.line(self.screen, grid_color, (start_x, start_y), (end_x, end_y), 1)
        
        # Draw horizontal lines
        for y in range(-grid_size * 5, grid_size * 6, grid_size):
            start_x, start_y = self.world_to_screen(-grid_size * 5, y)
            end_x, end_y = self.world_to_screen(grid_size * 5, y)
            pygame.draw.line(self.screen, grid_color, (start_x, start_y), (end_x, end_y), 1)
    
    def update_trajectory(self, state: State):
        """Update trajectory with new state.
        
        Args:
            state: New state to add to trajectory.
        """
        self.trajectory.append(state)
        
        # Limit trajectory length to prevent memory issues
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-500:]
    
    def clear_trajectory(self):
        """Clear the trajectory."""
        self.trajectory = []
    
    def render(self, state: State, step_count: int, reward: float, 
               done: bool, truncated: bool, rs_path: List[State]):
        """Render the current state.
        
        Args:
            state: Current vehicle state.
            step_count: Current step count.
            reward: Current reward.
            done: Whether episode is done.
            truncated: Whether episode was truncated.
            rs_path: Reeds-Shepp path.
        """
        # Clear screen
        self.screen.fill(self.config.DARK_BG)
        
        # Draw reference grid
        self.draw_grid()
        
        # Draw parking slot
        self.draw_parking_slot(done)
        
        # Draw trajectory
        self.draw_trajectory(self.trajectory)
        
        # Draw vehicle
        self.draw_vehicle(state)
        
        # Draw info panel
        self.draw_info(state, step_count, reward, done, truncated, rs_path)
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events.
        
        Returns:
            True if should continue, False if should quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    
    def close(self):
        """Close the renderer."""
        pygame.quit() 