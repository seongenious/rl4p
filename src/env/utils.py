import numpy as np
from typing import Tuple

def mod2pi(angle: float) -> float:
    """ Make angle between -pi and pi
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def world2local(ref: np.ndarray, world: np.ndarray) -> np.ndarray:
    ref_x, ref_y, ref_theta, _ = ref
    
    dx = world[0] - ref_x
    dy = world[1] - ref_y
    
    cos_neg_theta = np.cos(-ref_theta)
    sin_neg_theta = np.sin(-ref_theta)
    
    x = dx * cos_neg_theta - dy * sin_neg_theta
    y = dx * sin_neg_theta + dy * cos_neg_theta
    yaw = world[2] - ref_theta
    
    return np.array([x, y, yaw])