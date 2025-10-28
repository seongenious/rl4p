from typing import Dict, List, Tuple, Any
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size: int) -> None:
        self.items = ["state", "action", "reward", "done", "next_obs", "log_prob"]
        self.buffer = {}
        for item in self.items:
            self.buffer[item] = deque([], maxlen=buffer_size)
    
    def push(self, experience: Tuple) -> None:
        """Save a transition"""
        for i, item in enumerate(self.items):
            self.buffer[item].append(experience[i])

    def get_items(self, indices: np.ndarray) -> Dict[str, List[Any]]:
        batches = {}
        for item in self.items:
            batches[item] = []
        batches["next_state"] = []
        for idx in indices:
            for item in self.items:
                batches[item].append(self.buffer[item][idx])
            if idx == self.__len__() - 1 or self.buffer["done"][idx]:
                batches["next_state"].append(None)
            else:
                batches["next_state"].append(self.buffer["state"][idx + 1])
        for idx in batches.keys():
            if isinstance(batches[idx][0], np.ndarray):
                batches[idx] = np.array(batches[idx])
        return batches

    def sample(self, batch_size: int) -> Dict[str, List[Any]]:
        indices = np.random.randint(self.__len__(), size=batch_size)
        return self.get_items(indices)

    def clear(self) -> None:
        for item in self.items:
            self.buffer[item].clear()

    def __len__(self) -> int:
        return len(self.buffer["state"])