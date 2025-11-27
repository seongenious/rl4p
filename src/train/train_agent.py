import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
from shutil import copyfile
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent import Agent
from model.planner import RsPlanner
from model.sac import SAC
from model.guardian import Guardian
from env.parking_env import ParkingEnv
from configs import *


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sac_ckpt', type=str, default=None) # './model/ckpt/SAC.pt'
    parser.add_argument('--encoder_ckpt', type=str, default='../ckpt/autoencoder/autoencoder.pt')
    parser.add_argument('--guardian_ckpt', type=str, default='../ckpt/guardian/guardian.pt')
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=2000)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    verbose = args.verbose

    if args.visualize:
        env = ParkingEnv(config=EnvConfig(), verbose=verbose)
    else:
        env = ParkingEnv(config=EnvConfig(), verbose=verbose, render_mode='rgb_array')
    
    # Set random seed
    env.action_space.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Set path to log and save model
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path + '/log/exp/sac_%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    
    # Save configs
    copyfile('./configs.py', save_path + 'configs.txt')
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    # Load SAC model
    model_config = ModelConfig()
    sac = SAC(config=model_config)
    if args.sac_ckpt is not None:
        sac.load(args.sac_ckpt, params_only=True)
        print('load pre-trained model!')
    if args.encoder_ckpt is not None and os.path.exists(args.encoder_ckpt):
        sac.load_img_encoder(args.encoder_ckpt, require_grad=True)
        print('load pre-trained image encoder!')

    # Load guardian
    guardian = Guardian(config=model_config.guardian)
    if args.guardian_ckpt is not None and os.path.exists(args.guardian_ckpt):
        guardian.load(args.guardian_ckpt, require_grad=True)
        print('load pre-trained guardian!')

    # Set planner
    step_distance = env.config.vehicle.step_time * env.config.vehicle.num_step * env.config.action.speed[1]
    planner = RsPlanner(step_distance)
    
    # Agent initialization
    agent = Agent(rl_agent=sac, guardian=guardian, planner=planner)

    for i in range(args.train_episode):
        obs = env.reset(i + 1)
        done = False
        total_reward = 0
        step_num = 0

        while not done:
            step_num += 1
            action = env.action_space.sample()

            obs, reward, status, info = env.step(action=action)
            done = status != EnvStatus.RUNNING

            total_reward += reward
    
    env.close()