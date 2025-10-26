import pickle

import numpy as np
from numpy.random import randn, random
from shapely.geometry import LinearRing
from shapely.geometry.base import BaseGeometry
from typing import List

from env.vehicle import VehicleState
from configs import EnvConfig, ObservationConfig, ColorConfig


class Area(object):
    def __init__(self, shape: BaseGeometry = None, subtype: str = None, color: float = None):
        self.shape = shape
        self.subtype = subtype
        self.color = color

    def get_shape(self):
        return np.array(self.shape.coords)


class Map(object):
    def __init__(
        self, 
        data_dir: str = '../data/dlp.data',
        config: ObservationConfig = ObservationConfig(),
        color_config: ColorConfig = ColorConfig()
    ) -> None:
        self.config = config
        self.color = color_config
        self.case_id: int = None
        self.start: VehicleState = None
        self.dest: VehicleState = None
        self.start_bbox: LinearRing = None
        self.dest_bbox: LinearRing = None
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.n_obstacle = 0
        self.obstacles: List[Area] = []
        
        try:
            with open(data_dir, 'rb') as f:
                self.map_data = pickle.load(f)
            self.multi_start = isinstance(self.map_data[0][0], list)
        except Exception as e:
            print(f"Error loading map data: {e}")
            self.map_data = None
            self.multi_start = False

    def reset(self, case_id: int = None, data_dir: str = None) -> VehicleState:
        if data_dir is not None:
            with open(data_dir, 'rb') as f:
                self.map_data = pickle.load(f)
            self.multi_start = isinstance(self.map_data[0][0], list)
          
        if case_id is None:
            self.case_id = np.random.randint(0, len(self.map_data))
        else:
            if case_id >= len(self.map_data):
                case_id = case_id%len(self.map_data)
            self.case_id = case_id

        # Get poses from map data
        start, dest, obstacles = self.map_data[self.case_id][:3]

        if isinstance(start, tuple):
            start = list(start)
        if isinstance(dest, tuple):
            dest = list(dest)

        if self.multi_start:
            random_id = np.random.randint(0, len(start))
            start = start[random_id]
            start = (start[0] + randn()*0.05, start[1] + randn()*0.05, start[2] + randn()*0.02)
        
        # Create start and dest states
        self.start = VehicleState(start)
        self.start_bbox = self.start.bbox
        self.dest = VehicleState(dest)
        self.dest_bbox = self.dest.bbox

        max_dist_to_dest = self.config.max_dist_to_dest
        self.xmin = np.floor(min(self.start.position.x, self.dest.position.x) - max_dist_to_dest)
        self.xmax = np.ceil(max(self.start.position.x, self.dest.position.x) + max_dist_to_dest)
        self.ymin = np.floor(min(self.start.position.y, self.dest.position.y) - max_dist_to_dest)
        self.ymax = np.ceil(max(self.start.position.y, self.dest.position.y) + max_dist_to_dest)
        
        if not isinstance(obstacles[0], LinearRing):
            raise UserWarning('Obstcale shape should be shapely.LinearRing !')

        self.obstacles = []
        self.n_obstacle = 0
        if self.config.use_obstacle:
            self.filter_obstacles(obstacles)

        if random() > 0.5:
            self.flip_dest_orientation()
        if random() > 0.5:
            self.flip_start_orientation()

        return self.start
    
    def filter_obstacles(self, obstacles: List[LinearRing]):
        all_obstacles = list(
            [Area(shape=obs, subtype="obstacle", color=self.color.obstacle) for obs in obstacles]
        )

        filtered_obstacles = []
        for obs in all_obstacles:
            obs_coords = np.array(obs.shape.coords)
            x_max = np.max(obs_coords[:,0])
            x_min = np.min(obs_coords[:,0])
            y_max = np.max(obs_coords[:,1])
            y_min = np.min(obs_coords[:,1])
            if not (x_max <= self.xmin or x_min >= self.xmax or y_max <= self.ymin or y_min >= self.ymax):
                filtered_obstacles.append(obs)
        
        self.obstacles = filtered_obstacles
        self.n_obstacle = len(self.obstacles)
    
    def get_boundary(self):
        obs_coords = []
        for obs in self.obstacles:
            obs_coords.extend(list(obs.shape.coords))
        obs_coords = np.array(obs_coords)
        self.xmin = np.floor(min(np.min(obs_coords[:,0]), self.xmin))
        self.xmax = np.floor(max(np.max(obs_coords[:,0]), self.xmax))
        self.ymin = np.floor(min(np.min(obs_coords[:,1]), self.ymin))
        self.ymax = np.floor(max(np.max(obs_coords[:,1]), self.ymax))
    
    def change_start_dest(self):
        self.start, self.dest = self.dest, self.start
        self.start_bbox, self.dest_bbox = self.dest_bbox, self.start_bbox

    def _flip_box_orientation(self, target_state:VehicleState):
        x, y, heading = target_state.pose
        center = np.mean(target_state.bbox.coords[:-1], axis=0)
        new_x = 2 * center[0] - x
        new_y = 2 * center[1] - y
        heading = heading + np.pi
        return VehicleState([new_x, new_y, heading])

    def flip_dest_orientation(self):
        self.dest = self._flip_box_orientation(self.dest)
        self.dest_bbox = self.dest.bbox

    def flip_start_orientation(self):
        self.start = self._flip_box_orientation(self.start)
        self.start_bbox = self.start.bbox