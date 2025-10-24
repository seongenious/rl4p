from typing import Callable, List, Tuple, Union
import copy

import numpy as np
from shapely.geometry import Point, LinearRing
from shapely.affinity import affine_transform

from configs import VehicleConfig, ActionConfig, ColorConfig


class VehicleState:
    def __init__(self, state: List[float]) -> None:
        self.position: Point = Point(state[:2])
        self.heading: float = state[2]
        if len(state) == 3:
            self.speed: float = 0
            self.steer: float = 0
        else:
            self.speed: float = state[3]
            self.steer: float = state[4]

    @property
    def pose(self) -> Tuple[float, float, float]:
        return (self.position.x, self.position.y, self.heading)
    
    @property
    def bbox(self) -> LinearRing:
        cos_h, sin_h = np.cos(self.heading), np.sin(self.heading)
        mat = [cos_h, -sin_h, sin_h, cos_h, self.position.x, self.position.y]
        return affine_transform(VehicleConfig.bbox, mat)


class KinematicModel:
    def __init__(self, vehicle_config: VehicleConfig, action_config: ActionConfig) -> None:
        self.wheel_base = vehicle_config.wheel_base
        self.step_time = vehicle_config.step_time
        self.num_step = vehicle_config.num_step
        self.speed_range = action_config.speed
        self.steer_range = action_config.steer

    def step(self, state: VehicleState, action: Union[List, np.ndarray]) -> VehicleState:
        x, y, heading = state.position.x, state.position.y, state.heading
        steer, speed = np.clip(action[0], *self.steer_range), np.clip(action[1], *self.speed_range)

        for _ in range(self.num_step):
            x += speed * np.cos(heading) * self.step_time
            y += speed * np.sin(heading) * self.step_time
            heading += speed * np.tan(steer) / self.wheel_base * self.step_time

        return VehicleState([x, y, heading, speed, steer])


class Vehicle:
    def __init__(
        self, 
        vehicle_config: VehicleConfig = VehicleConfig(), 
        action_config: ActionConfig = ActionConfig(), 
        color_config: ColorConfig = ColorConfig()
    ) -> None:
        self.initial_state: VehicleState = None
        self.state: VehicleState = None
        self.bbox: LinearRing = None
        self.trajectory: List[VehicleState] = []
        self.model: Callable = KinematicModel(vehicle_config, action_config)
        self.color = color_config.vehicle
        self.v_max = None
        self.v_min = None

    def reset(self, initial_state: VehicleState) -> None:
        self.initial_state = initial_state
        self.state = self.initial_state
        self.v_max = self.initial_state.speed
        self.v_min = self.initial_state.speed
        self.bbox = self.state.bbox
        self.trajectory.clear()
        self.trajectory.append(self.state)

    def step(self, action: Union[List, np.ndarray]) -> Tuple:
        prev_info = copy.deepcopy((self.state, self.bbox, self.v_max, self.v_min))
        self.state = self.model.step(self.state, action)
        self.bbox = self.state.bbox
        self.trajectory.append(self.state)
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min
        return prev_info

    def retreat(self, prev_info: Tuple) -> None:
        self.state, self.bbox, self.v_max, self.v_min = prev_info
        self.trajectory.pop(-1)