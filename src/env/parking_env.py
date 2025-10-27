"""Parking environment for reinforcement learning."""
from typing import Dict, Any, Tuple, OrderedDict, List

import numpy as np
import gym
from gym import spaces
from shapely.geometry import Polygon, LineString
from heapdict import heapdict
import pygame
import matplotlib.pyplot as plt

import env.reeds_shepp as rs
from env.map import Map, Area
from env.viewer import EnvViewer
from env.vehicle import Vehicle
from env.image_processor import ImageProcessor
from configs import EnvStatus, EnvConfig


class ParkingEnv(gym.Env):
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
        
        self.rs_path = None

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

        # Compute reward
        reward_info = self.compute_reward()

        # Create info
        self.rs_path = self._find_rs_path()
        info = {
            'rs_path': self.rs_path,
        }

        return obs, self.reward, status, info
    
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

    def _find_rs_path(self) -> List:
        start_x, start_y, start_heading = self.vehicle.state.pose
        goal_x, goal_y, goal_heading = self.map.dest.pose
        radius = np.tan(self.config.action.steer[-1])/self.config.vehicle.wheel_base

        all_rs_paths = rs.calc_all_paths(
            start_x, start_y, start_heading, goal_x, goal_y, goal_heading, radius, 0.2)
        
        if not all_rs_paths:
            return None

        cost_queue = heapdict()
        for rs_path in all_rs_paths:
            cost_queue[rs_path] = rs_path.L

        for idx in range(len(all_rs_paths)):
            rs_path = cost_queue.popitem()[0]
            if idx > self.config.rs_max_iter:
                break

            path = [[rs_path.x[k], rs_path.y[k], rs_path.yaw[k]] for k in range(len(rs_path.x))]
            if self._is_path_valid(path):
                return rs_path

        return None
    
    def _is_path_valid(self, path: List) -> bool:
        car_coords1 = np.array(self.config.vehicle.bbox.coords)[:4] # (4,2)
        car_coords2 = np.array(self.config.vehicle.bbox.coords)[1:] # (4,2)
        car_coords_x1 = car_coords1[:,0].reshape(1,-1)
        car_coords_y1 = car_coords1[:,1].reshape(1,-1) # (1,4)
        car_coords_x2 = car_coords2[:,0].reshape(1,-1)
        car_coords_y2 = car_coords2[:,1].reshape(1,-1) # (1,4)
        vxs = np.array([t[0] for t in path])
        vys = np.array([t[1] for t in path])
        # check outbound
        if np.min(vxs) < self.map.xmin or np.max(vxs) > self.map.xmax \
        or np.min(vys) < self.map.ymin or np.max(vys) > self.map.ymax:
            return False
        vthetas = np.array([t[2] for t in path])
        cos_theta = np.cos(vthetas).reshape(-1,1) # (T,1)
        sin_theta = np.sin(vthetas).reshape(-1,1)
        vehicle_coords_x1 = cos_theta*car_coords_x1 - sin_theta*car_coords_y1 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y1 = sin_theta*car_coords_x1 + cos_theta*car_coords_y1 + vys.reshape(-1,1)
        vehicle_coords_x2 = cos_theta*car_coords_x2 - sin_theta*car_coords_y2 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y2 = sin_theta*car_coords_x2 + cos_theta*car_coords_y2 + vys.reshape(-1,1)
        vx1s = vehicle_coords_x1.reshape(-1,1)
        vx2s = vehicle_coords_x2.reshape(-1,1)
        vy1s = vehicle_coords_y1.reshape(-1,1)
        vy2s = vehicle_coords_y2.reshape(-1,1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (vy2s - vy1s).reshape(-1,1) # (4*t,1)
        b = (vx1s - vx2s).reshape(-1,1)
        c = (vy1s*vx2s - vx1s*vy2s).reshape(-1,1)
        
        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x_max = np.max(vx1s) + 5
        x_min = np.min(vx1s) - 5
        y_max = np.max(vy1s) + 5
        y_min = np.min(vy1s) - 5

        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in self.map.obstacles:
            if isinstance(obst, Area):
                obst = obst.shape
            obst_coords = np.array(obst.coords) # (n+1,2)
            if (obst_coords[:,0] > x_max).all() or (obst_coords[:,0] < x_min).all()\
                or (obst_coords[:,1] > y_max).all() or (obst_coords[:,1] < y_min).all():
                continue
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return True
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # calculate the intersections
        det = a*e - b*d # (4, E)
        parallel_line_pos = (det==0) # (4, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (4, E)
        raw_y = (c*d - a*f)/det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        collide_map_x[raw_x>np.maximum(x1s, x2s)] = 0
        collide_map_x[raw_x<np.minimum(x1s, x2s)] = 0
        collide_map_y[raw_y>np.maximum(y1s, y2s)] = 0
        collide_map_y[raw_y<np.minimum(y1s, y2s)] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x>np.maximum(vx1s, vx2s)] = 0
        collide_map_x[raw_x<np.minimum(vx1s, vx2s)] = 0
        collide_map_y[raw_y>np.maximum(vy1s, vy2s)] = 0
        collide_map_y[raw_y<np.minimum(vy1s, vy2s)] = 0

        collide_map = collide_map_x*collide_map_y
        collide_map[parallel_line_pos] = 0
        collide = np.sum(collide_map) > 0

        if collide:
            return False
        return True
    
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