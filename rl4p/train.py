"""SAC training script for RL4P project.

This module implements a simplified Soft Actor-Critic (SAC) algorithm
for testing purposes. The full BC-SAC pipeline will be implemented separately.
"""

import os
import time
import argparse
from datetime import datetime
from collections import deque

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from env.parking_env import ParkingEnv
from model.trainer import (
    create_networks, create_train_state, train_step, soft_update, sample_action
)
from model.replay_buffer import (
    ReplayBuffer, create_buffer, add_buffer, sample_batch_transitions
)
from util.io import load_yaml_config, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC model.')
    parser.add_argument('--output', type=str, default=None, help='Output directory name')
    return parser.parse_args()


def main():
    """Run main training loop for SAC (simplified, critic only).
    
    Note: This is a simplified SAC implementation for testing purposes.
    The full BC-SAC pipeline will be implemented separately.
    """
    print('Start to train SAC model...')
    
    args = parse_args()
    if args.output is None:
      raise ValueError('Output directory is required.')

    # Get configuration in yaml
    config = load_yaml_config('./config/sac.yaml')
    
    # Network configurations
    learning_rate = config['train']['learning_rate']
    weight_decay = config['train']['weight_decay']
    gamma = config['train']['gamma']
    tau = config['train']['tau']
    alpha = config['train']['alpha']
    
    # Training configurations
    num_episodes = config['train']['num_episodes']
    random_steps = config['train']['random_steps']
    update_after = config['train']['update_after']
    update_every = config['train']['update_every']
    update_repeat = config['train']['update_repeat']

    # Logger configurations
    log_dir = os.path.join("./runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(log_dir)
    
    # Other configurations
    ckpt_dir = config['train']['dir']
    ckpt_dir = os.path.join(ckpt_dir, args.output)
    
    # Create networks
    print('Initialize SAC model...')
    rng = jax.random.PRNGKey(0)
    networks, network_params = create_networks(rng)
    states = create_train_state(networks, network_params, learning_rate, weight_decay)
    target_critic1_params = states['critic1'].params
    target_critic2_params = states['critic2'].params
    
    # Create replay buffer
    replay_buffer = create_buffer()
    
    # Setup environment 
    env = ParkingEnv(config, render=config['train']['render'])

    # Training loop
    start = time.time()
    success_history = deque(maxlen=int(num_episodes / 10))
    pbar = tqdm(range(num_episodes))
    for step in pbar:
        pbar.set_description(f"Episode {step + 1}")

        # Reset episode 
        obs, info = env.reset()
        done, truncated = False, False
        episode_reward = 0
            
        # Run episode
        while True:
            rng, sub_rng = jax.random.split(rng)
            
            # 0. Expand dimension of observation
            occupancy_grid = jnp.expand_dims(obs.occupancy_grid, axis=0)
            state = jnp.expand_dims(obs.state, axis=0)
            
            # 1. Encode feature
            feat = networks['encoder'].apply(
                {'params': states['encoder'].params}, occupancy_grid, state)
            
            # 2. Sample action
            action = env.action_space.sample() if step < random_steps else sample_action(
                states['actor'], states['actor'].params, feat, sub_rng)[0]
            if action.ndim == 2: 
                action = action[0]

            # 3. Step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 4. Update replay buffer
            add_buffer(replay_buffer, obs, action, reward, done or truncated, next_obs)
            success_history.append(int(done))
                        
            # 5. Render environment
            episode_reward += reward
            kwargs = {
                'action': action,
                'reward': episode_reward,
                'done': done,
                'truncated': truncated,
                'rs_path': info['rs_path'],
            }
            env.render(obs, **kwargs)
            obs = next_obs
            
            if done or truncated: 
                break
            
            # Draw activation map
            # activation = feat[0].mean(axis=-1)  # (H, W)
    
            # plt.figure(figsize=(4, 4))
            # plt.imshow(activation, cmap='viridis')
            # plt.colorbar()
            # plt.title(f"Feature Activation Map at step {step}")
    
            # if logger:
            #     logger.add_figure("activation_map", plt.gcf(), global_step=step)
            # else:
            #     plt.show()
            # plt.close()
                    
        # Network update
        if step >= update_after and (step - update_after) % update_every == 0:
            for _ in range(update_repeat):
                rng, sub_rng = jax.random.split(rng)
                
                batch = sample_batch_transitions(replay_buffer, rng)
                
                states, logs = train_step(
                    sub_rng, states, batch, alpha, target_critic1_params, target_critic2_params, gamma)
                target_critic1_params = soft_update(
                    target_critic1_params, states['critic1'].params, tau)
                target_critic2_params = soft_update(
                    target_critic2_params, states['critic2'].params, tau)

            # Log loss per update
            logger.add_scalar(f"Loss/Actor", float(logs['actor_loss']), step)
            logger.add_scalar(f"Loss/Critic", float(logs['critic_loss']), step)
            logger.add_scalar(f"SuccessRate", sum(success_history) / len(success_history), step)

            # Save checkpoint
            save_checkpoint(
                step=step,
                encoder_state=states['encoder'],
                actor_state=states['actor'], 
                critic1_state=states['critic1'],
                critic2_state=states['critic2'],
                ckpt_dir=ckpt_dir
            )
        
    # Close environment
    env.close()
    logger.close()
    
    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}.")


if __name__ == '__main__':
    main()
