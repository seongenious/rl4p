import numpy as np
import torch
from enum import Enum

from typing import List, Tuple
from shapely.geometry import LinearRing


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42


class VehicleConfig:
    wheel_base: float = 2.845  # [m]
    front_overhang: float = 0.8  # [m]
    rear_overhang: float = 1.0  # [m]
    length: float = 4.855  # [m]
    width: float = 1.88  # [m]
    bbox: LinearRing = LinearRing([
        (-rear_overhang, -width/2), 
        (front_overhang + wheel_base, -width/2), 
        (front_overhang + wheel_base,  width/2), 
        (-rear_overhang,  width/2)])
    num_step: int = 100
    step_time: float = 0.005  # [s] total step time is step_time * num_step


class ActionConfig:
    speed: List[float] = [-2.5, 2.5]  # [m/s]
    steer: List[float] = [-0.7, 0.7]  # [rad]
    accel: List[float] = [-1.0, 1.0]  # [m/s^2]
    angular_speed: List[float] = [-0.5, 0.5]  # [rad/s]
    precision: int = 10
    discrete_actions: List[List[float]] = []
    for i in np.arange(speed[0], speed[1], (speed[1] - speed[0])/precision):
        discrete_actions.append([i, 0])
    for i in np.arange(steer[0], steer[1], (steer[1] - steer[0])/precision):
        discrete_actions.append([0, i])
    
    @property
    def num_discrete_actions(self) -> int:
        return len(self.discrete_actions)


class ObservationConfig:
    img_size: Tuple[int, int] = (256, 256)
    img_channels: int = 3
    img_downsample_rate: int = 4
    target_dim: int = 5
    use_obstacle: bool = False
    max_dist_to_dest: float = 20


class ColorConfig:
    background: Tuple[int, int, int, int] = (255, 255, 255, 255)
    start: Tuple[int, int, int, int] = (100, 149, 237, 255)
    destination: Tuple[int, int, int, int] = (69, 139, 0, 255)
    obstacle: Tuple[int, int, int, int] = (150, 150, 150, 255)
    collision: Tuple[int, int, int, int] = (255, 0, 0, 255)
    rs_path: Tuple[int, int, int, int] = (0, 0, 255, 255)
    trajectory_high: Tuple[int, int, int, int] = (10, 10, 200, 255)
    trajectory_low: Tuple[int, int, int, int] = (10, 10, 10, 255)
    trajectory_render_len: int = 20
    trajectory_colors: List[Tuple[int, int, int, int]] = list(
        map(tuple, np.linspace(np.array(trajectory_low), np.array(trajectory_high), trajectory_render_len, endpoint=True, dtype=np.uint8)))
    vehicle: Tuple[int, int, int, int] = (30, 144, 255, 255)
    text: Tuple[int, int, int, int] = (0, 0, 0, 255)


class EnvStatus(Enum):
    RUNNING = 0
    SUCCESS = 1
    COLLISION = 2
    OUTBOUND = 3
    TIME_EXCEEDED = 4


class EnvConfig:
    width: int = 600
    height: int = 600
    fps: int = 100
    max_time_step: int = 200
    render_scale: int = 12
    rs_max_dist: float = 10
    rs_max_iter: int = 4
    render_traj: bool = True
    success_threshold: float = 0.95

    vehicle: VehicleConfig = VehicleConfig()
    action: ActionConfig = ActionConfig()
    observation: ObservationConfig = ObservationConfig()
    color: ColorConfig = ColorConfig()
    
    
