import numpy as np
import pygame

from shapely.affinity import affine_transform
from shapely.geometry import LineString

from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from env.base_env import BaseEnv

from configs import EnvConfig


class EnvViewer:
    """Viewer for parking environment."""

    def __init__(
        self, env: 'BaseEnv', 
        render_mode: str = None, 
        config: Optional[EnvConfig] = None
    ) -> None:
        self.env = env
        self.render_mode = "human" if render_mode is None else render_mode
        self.config = config or EnvConfig()

        self.fps = self.config.fps
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.tf_matrix = None

        display_flags = pygame.SHOWN if self.render_mode == "human" else pygame.HIDDEN
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.config.width, self.config.height), flags = display_flags)

        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
    
    def reset(self) -> None:
        k = self.config.render_scale
        tx = 0.5 * (self.config.width - k * (self.env.map.xmax + self.env.map.xmin))
        ty = 0.5 * (self.config.height - k * (self.env.map.ymax + self.env.map.ymin))
        self.tf_matrix = [k, 0, 0, k, tx, ty]
        return None

    def _coord_transform(self, object: Any) -> list:
        transformed = affine_transform(object, self.tf_matrix)
        return list(transformed.coords)

    def render(self) -> None:
        assert self.screen is not None
        assert self.clock is not None

        self.screen.fill(self.config.color.background)

        for obstacle in self.env.map.obstacles:
            pygame.draw.polygon(
                self.screen, obstacle.color, self._coord_transform(obstacle.shape))

        pygame.draw.polygon(
            self.screen, self.config.color.start, self._coord_transform(self.env.map.start_bbox), width=1)

        pygame.draw.polygon(
            self.screen, self.config.color.destination, self._coord_transform(self.env.map.dest_bbox))

        pygame.draw.polygon(
            self.screen, self.config.color.vehicle, self._coord_transform(self.env.vehicle.bbox))

        if self.config.render_traj and len(self.env.vehicle.trajectory) > 1:
            traj_len = min(len(self.env.vehicle.trajectory), self.config.color.trajectory_render_len)
            traj_colors = self.config.color.trajectory_colors
            for i in range(traj_len):
                bbox = self.env.vehicle.trajectory[-(traj_len - i)].bbox
                pygame.draw.polygon(
                    self.screen, traj_colors[-(traj_len - i)], self._coord_transform(bbox))

        if self.config.render_rs_path and self.env.rs_path is not None:
            rs_path = self.env.rs_path
            path = [[rs_path.x[k], rs_path.y[k], rs_path.yaw[k]] for k in range(len(rs_path.x))]
            linestring = LineString(point[:2] for point in path)
            pygame.draw.lines(
                self.screen, self.config.color.rs_path, False, self._coord_transform(linestring), width=1)

        if self.config.render_info:
            self._render_info_text()

        pygame.display.update()
        self.clock.tick(self.fps)
    
    def _render_info_text(self) -> None:
        info_text = f"Reward: {self.env.reward:.3f} ({self.env.sim_time} steps)"
        info_surface = self.font.render(info_text, True, self.config.color.text)
        self.screen.blit(info_surface, (10, 10))
    
    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
