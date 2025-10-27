import sys
sys.path.append("..")
sys.path.append(".")
import argparse

import torch

import env.parking_env as env
from model.planner import RsPlanner
from configs import EnvStatus, EnvConfig


USE_PLANNER = True

if __name__ == "__main__":
    env = env.ParkingEnv(render_mode="human")

    step_distance = env.config.vehicle.step_time * env.config.vehicle.num_step * env.config.action.speed[1]
    planner = RsPlanner(step_distance)

    for i in range(10):
        obs = env.reset(i + 1)
        done = False
        while not done:
            if USE_PLANNER:
                action = planner.get_action()
            else:
                action = env.action_space.sample()

            obs, reward, status, info = env.step(action=action)
            done = status != EnvStatus.RUNNING

            planner.set_path(info['rs_path'])
    
    env.close()