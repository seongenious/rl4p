"""Parking environment for reinforcement learning."""
import sys
sys.path.append("../")
sys.path.append(".")

from typing import Dict, Any, Tuple, OrderedDict

import numpy as np
import gym
from gym import spaces
from shapely.geometry import Polygon 
from heapdict import heapdict
import pygame
import matplotlib.pyplot as plt

from reeds_shepp import reeds_shepp as rs
from map import Map
from viewer import EnvViewer
from vehicle import Vehicle
from image_processor import ImageProcessor
from configs import EnvStatus, EnvConfig


class BaseEnv(gym.Env):
    metadata = {"render_mode": ["human", "rgb_array",]}

    def __init__(
        self, 
        render_mode: str = None, 
        config = EnvConfig(), 
        verbose: bool = True
    ) -> None:
        super().__init__()

        self.render_mode = "human" if render_mode is None else render_mode
        self.config = config or EnvConfig()
        self.verbose = verbose

        self.viewer = EnvViewer(self, render_mode=self.render_mode, config=self.config)
        self.sim_time = 0
        self.tf_matrix = None

        self.vehicle = Vehicle(vehicle_config=config.vehicle, action_config=config.action)
        self.map = Map(data_dir='../data/dlp.data', config=config.observation, color_config=config.color)
        self.reward = 0.0
        self.img_processor = ImageProcessor(config=self.config)

        self.define_spaces(
            action_config=config.action,
            observation_config=config.observation,
        )


    def define_spaces(self, action_config, observation_config) -> None:
        self.action_space = spaces.Box(
            np.array([action_config.steer[0], action_config.speed[0]]).astype(np.float32),
            np.array([action_config.steer[1], action_config.speed[1]]).astype(np.float32),
        )

        self.observation_space = {}
        self.observation_space['img'] = spaces.Box(
            low=0, high=255, 
            shape=(
                observation_config.img_size[0]//observation_config.img_downsample_rate, 
                observation_config.img_size[1]//observation_config.img_downsample_rate, 
                observation_config.img_channels), 
            dtype=np.uint8,
        )
        self.observation_space['target'] = spaces.Box(
            low=np.array([0,-1,-1,-1,-1]),
            high=np.array([observation_config.max_dist_to_dest,1,1,1,1]),
            shape=(observation_config.target_dim,), 
            dtype=np.float64,
        )


    def reset(self, case_id: int = None, data_dir: str = None):
        self.sim_time = 0
        self.reward = 0.0

        initial_state = self.map.reset(case_id=case_id, data_dir=data_dir)
        self.vehicle.reset(initial_state)
        self.viewer.reset()

        return self.observe()


    def step(self, action: np.ndarray = None) -> Tuple[Any, float, bool, Dict[str, Any]]:
        prev_state = self.vehicle.state
        
        status = EnvStatus.RUNNING
        if action is not None:
            prev_info = self.vehicle.step(action)
            if self._is_arrived():
                status = EnvStatus.SUCCESS
            if self._has_collision():
                status = EnvStatus.COLLISION
            if self._is_outbound():
                status = EnvStatus.OUTBOUND
        
        self.sim_time += 1
        if self.sim_time >= self.config.max_time_step:
            status = EnvStatus.TIME_EXCEEDED

        # Render the environment
        self.viewer.render()

        # Get observation
        obs = self.observe()

        # Get reward
        reward = self.compute_reward()

        return obs, self.reward, status, None
    

    def _is_arrived(self) -> bool:
        ego_bbox = Polygon(self.vehicle.bbox)
        dst_bbox = Polygon(self.map.dest_bbox)
        union_area = ego_bbox.intersection(dst_bbox).area
        return union_area / dst_bbox.area > self.config.success_threshold


    def _has_collision(self) -> bool:
        for obstacle in self.map.obstacles:
            if self.vehicle.bbox.intersects(obstacle.shape):
                obstacle.color = self.config.color.collision
                return True
        return False
    

    def _is_outbound(self) -> bool:
        x, y, _ = self.vehicle.state.pose
        return x > self.map.xmax or x < self.map.xmin or y > self.map.ymax or y < self.map.ymin
    

    def observe(self) -> Dict[str, Any]:
        img = self._get_image(self.viewer.screen)
        target = self._get_target()
        return {'img': img, 'target': target}
    

    def _get_image(self, surface: pygame.Surface) -> np.ndarray:
        angle = self.vehicle.state.heading
        old_center = surface.get_rect().center

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(surface, np.rad2deg(angle))
        rotate = pygame.Surface((self.config.width, self.config.height))
        rotate.blit(capture, capture.get_rect(center=old_center))
        
        vehicle_center = np.array(self.viewer._coord_transform(self.vehicle.bbox.centroid)[0])
        dx = (vehicle_center[0] - old_center[0]) * np.cos(angle) + (vehicle_center[1] - old_center[1]) * np.sin(angle)
        dy = -(vehicle_center[0] - old_center[0]) * np.sin(angle) + (vehicle_center[1] - old_center[1]) * np.cos(angle)
        
        # Align the center of the observation with the center of the vehicle
        surface = pygame.Surface((self.config.width, self.config.height))
    
        surface.fill(self.config.color.background)
        surface.blit(rotate, (int(-dx), int(-dy)))
        surface = surface.subsurface((
            (self.config.width - self.config.observation.img_size[0]) / 2, 
            (self.config.height - self.config.observation.img_size[1]) / 2), 
            (self.config.observation.img_size[0], self.config.observation.img_size[1])
        )
    
        obs_str = pygame.image.tostring(surface, "RGB")
        img = np.frombuffer(obs_str, dtype=np.uint8)
        img = img.reshape((
            self.config.observation.img_size[0], 
            self.config.observation.img_size[1], 
            self.config.observation.img_channels)
        )

        processed_img = self.img_processor.process_img(img)

        # plt.figure(figsize=(10, 8))
        # plt.imshow(processed_img)
        # plt.title("생성된 관찰 이미지")
        # plt.axis('off')
        # plt.show()

        return processed_img


    def _get_target(self) -> np.ndarray:
        dest_pose = self.map.dest.pose
        ego_pose = self.vehicle.state.pose

        dx = dest_pose[0] - ego_pose[0]
        dy = dest_pose[1] - ego_pose[1]

        rel_distance = np.sqrt((dx) ** 2 + (dy) ** 2)
        rel_direction = np.arctan2(dy, dx) - ego_pose[2]
        rel_heading = dest_pose[2] - ego_pose[2]

        return np.array([
            rel_distance,
            np.cos(rel_direction),
            np.sin(rel_direction),
            np.cos(rel_heading),
        ])


    def compute_reward(self) -> OrderedDict:
        # rs_path = rs.get_optimal_path(self.vehicle.state.pose, self.map.dest.pose)

        return OrderedDict({
            'time_cost': 0.0,
            'rs_dist_reward': 0.0,
            'dist_reward': 0.0,
            'angle_reward': 0.0,
            'box_union_reward': 0.0,
        })


    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = BaseEnv(render_mode="human")

    for i in range(10):
        obs = env.reset(i + 1)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, status, info = env.step(action=action)
            done = status != EnvStatus.RUNNING
    
    env.close()