from typing import List
import numpy as np

from env.reeds_shepp import PATH


ACTION_TYPE = {'L': 1, 'S': 0, 'R': -1}

class RsPlanner(object):
    def __init__(self, step_distance: float) -> None:
        self.rs_path = None
        self.step_distance = step_distance
        self.actions = []

    def reset(self) -> None:
        self.rs_path = None
        self.actions = []
    
    def set_path(self, rs_path: PATH) -> None:
        self.rs_path = rs_path
        action_list = []
        for i in range(len(self.rs_path.ctypes)):
            steer = ACTION_TYPE[self.rs_path.ctypes[i]]
            num_steps = self.rs_path.lengths[i] / self.step_distance
            action_list.append([steer, num_steps])

        actual_actions = []
        for steer, num_steps in action_list:
            if abs(num_steps) < 1 and abs(num_steps) > 1e-3:
                actual_actions.append([steer, num_steps])
            elif num_steps > 1:
                while num_steps > 1:
                    actual_actions.append([steer, 1])
                    num_steps -= 1
                if abs(num_steps) > 1e-3:
                    actual_actions.append([steer, num_steps])
            elif num_steps < -1:
                while num_steps < -1:
                    actual_actions.append([steer, -1])
                    num_steps += 1
                if abs(num_steps) > 1e-3:
                    actual_actions.append([steer, num_steps])
        
        self.actions = actual_actions

    def get_action(self) -> List:
        if len(self.actions) == 0:
            return None
            
        action = self.actions.pop(0)
        if len(self.actions) == 0 and self.rs_path is not None:
            self.reset()
        return action