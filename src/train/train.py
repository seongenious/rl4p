import sys
sys.path.append("..")
sys.path.append(".")
import argparse

import torch

import env.parking_env as env
from configs import EnvStatus, EnvConfig


if __name__ == "__main__":
    env = env.ParkingEnv(render_mode="human")

    for i in range(10):
        obs = env.reset(i + 1)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, status, info = env.step(action=action)
            done = status != EnvStatus.RUNNING
    
    env.close()