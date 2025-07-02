# RL4P: Reinforcement Learning for Parking

A reinforcement learning project for autonomous parking using BC-SAC (Behavior Cloned Soft Actor Critic) model.

## Project Overview

This project implements a reinforcement learning system for autonomous parking using the BC-SAC algorithm. The system learns to control a vehicle's steering and acceleration inputs to perform parking maneuvers.

## Project Structure

```
rl4p/
├── rl4p/                    # Main package
│   ├── envs/               # Environment definitions
│   │   ├── base_env.py     # Base environment class
│   │   ├── freespace_env.py # Free space environment
│   │   ├── kinematic_model.py # Vehicle kinematic model
│   │   ├── reward.py       # Reward functions
│   │   └── datatypes.py    # Data type definitions
│   ├── models/             # Model definitions
│   │   ├── sac_networks.py # SAC networks (basic)
│   │   ├── bc_sac_networks.py # BC-SAC networks (project spec)
│   │   └── sac_trainer.py  # SAC training utilities
│   ├── utils/              # Utilities
│   │   ├── io.py          # I/O functions
│   │   ├── config.py      # Configuration loading
│   │   ├── logger.py      # Logging
│   │   ├── plot.py        # Plotting
│   │   ├── replay_buffer.py # Replay buffer
│   │   └── unit.py        # Unit conversions
│   ├── run_sac.py         # SAC training script
│   ├── visualize.py       # Visualization script
│   └── example.py         # Example script
├── tests/                 # Test code
│   └── test_reeds_shepp.py # Reeds-Shepp path test
├── config/                # Configuration files
│   └── sac.yaml          # SAC configuration
├── data/                  # Data storage
├── checkpoints/           # Model checkpoints
└── scripts/               # Scripts
```

## Key Features

### Model Architecture
- **Input**: Initial state (x, y, yaw, v), goal state (x, y, yaw, v), occupancy grid map
- **Output**: Control inputs [delta, accel] (steering angle, acceleration)
- **Goal State**: Always (0, 0, 0, v) - normalized to local coordinate system center

### Environment
- Virtual dataset generation
- Driver logs and off-policy data for SAC
- Stored in `/data/` directory

### Validation
- Pygame-based simulation environment
- Parking slot (5m x 2m) around goal state
- Trajectory generation through rollout (20 iterations)
- 30-second simulation duration

## Usage

### Data Generation
```bash
python rl4p/example.py
```

### SAC Training
```bash
python rl4p/run_sac.py
```

### Visualization
```bash
python rl4p/visualize.py --ckpt_dir ./checkpoints --config ./config/sac.yaml
```

## Development Guidelines

- Follow Google Coding Style
- All function comments in English
- Include detailed explanatory comments in code
- Clearly specify model shapes

## Current Status

- Building SAC-only training and validation pipeline for BC-SAC
- Organizing test code and modularization in progress

tensorboard --logdir=runs