# RL4P: Reinforcement Learning for Parking

A reinforcement learning project for autonomous parking using SAC (Soft Actor Critic) model.

## Project Overview

This project implements a reinforcement learning system for autonomous parking using the SAC algorithm. The system learns to control a vehicle's steering and acceleration inputs to perform parking maneuvers.

## Todo

[x] ~~Multi-head policy network (steering angle, accel)~~
[v] Change observation
[x] ~~Auxiliary task for Gear change~~
[v] Reward normalization [-1, 1]
  - Progress = 1 - current RS path length / initial RS path length
  - Kinematic condition = [0, 1]
  - Off road = -1 if progress < 0
[v] Add ~~dropout~~ and L2 normalization to critic network
[] Reward shaping (not tend to stop hold)
