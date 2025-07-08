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
    states = create_train_state(networks, network_params, learning_rate)
    target_critic_params = states['critic'].params
    
    # Create replay buffer
    replay_buffer = create_buffer()
    
    # Curriculum learning variables
    window_size = config['curriculum']['window_size']
    success_threshold = config['curriculum']['success_rate']
    curriculum_length = len(config['curriculum']['initial_position'])
    success_history = deque(maxlen=window_size)
    level = 0
    episodes = 0

    # Training loop
    start = time.time()
    while level < curriculum_length:
        print(f"\n=== Starting Level {level} ===")
        print(f"Initial position: {config['curriculum']['initial_position'][level]} (m)")
        print(f"Initial heading: {config['curriculum']['initial_heading'][level]} (deg)")
        print(f"Distance threshold: {config['curriculum']['distance_threshold'][level]} (m)")
        print(f"Heading threshold: {config['curriculum']['heading_threshold'][level]} (deg)")
        print(f"Zero speed: {config['curriculum']['zero_speed'][level]} (km/h)")
        
        # Setup environment 
        env = ParkingEnv(config, level, config['train']['render'])

        pbar = tqdm(range(num_episodes))
        for step in pbar: 
            pbar.set_description(f"Level {level} - Episode {step + 1}")

            # Reset episode 
            obs, info = env.reset()
            done, truncated = False, False
            episode_reward = 0
                
            # Run episode
            while True:
                rng, sub_rng = jax.random.split(rng)
                
                # 1. Encode feature
                feat = networks['encoder'].apply(
                    {'params': states['encoder'].params}, obs.occupancy_grid, obs.state)
                
                # 2. Sample action
                action = env.action_space.sample() if step < random_steps else sample_action(
                    states['actor'], states['actor'].params, feat, sub_rng)[0]
                if action.ndim == 2: 
                    action = action[0]

                # 3. Step
                next_obs, reward, done, truncated, info = env.step(action)
                
                # 4. Update replay buffer
                add_buffer(replay_buffer, obs, action, reward, done or truncated, next_obs)
                            
                # 5. Render environment
                episode_reward += reward
                kwargs = {
                    'action': action,
                    'reward': episode_reward,
                    'done': done,
                    'truncated': truncated,
                    'initial_path': info['initial_path'],
                    'current_path': info['current_path'],
                }
                env.render(obs, **kwargs)
                obs = next_obs
                
                if done or truncated: 
                    break

            # Update curriculum learning variables
            episodes += 1
            success_history.append(int(done))  # 1 for success, 0 for failure
            
            # Calculate current success rate
            success_rate = sum(success_history) / window_size

            # Log reward and success per episode
            logger.add_scalar(f"Level_{level}/Reward", episode_reward, step)
            logger.add_scalar(f"Level_{level}/SuccessRate", success_rate, step)
                        
            # Network update
            if step >= update_after and (step - update_after) % update_every == 0:
                for _ in range(update_repeat):
                    batch = sample_batch_transitions(replay_buffer, rng)
                    rng, sub_rng = jax.random.split(rng)
                    
                    states, logs = train_step(
                        sub_rng, states, batch, alpha, target_critic_params, gamma)
                    target_critic_params = soft_update(
                        target_critic_params, states['critic'].params, tau)

                # Log loss per update
                logger.add_scalar(f"Level_{level}/Loss/Actor", float(logs['actor_loss']), step)
                logger.add_scalar(f"Level_{level}/Loss/Critic", float(logs['critic_loss']), step)

                # Save checkpoint
                save_checkpoint(
                    step=episodes,
                    encoder_state=states['encoder'],
                    actor_state=states['actor'], 
                    critic_state=states['critic'], 
                    ckpt_dir=ckpt_dir
                )
            
            # Check if we should advance to next level
            if success_rate >= success_threshold:
                success_history.clear()
                level += 1
                break
        
        # If we didn't advance to next level, continue with more episodes
        if level >= curriculum_length:
            break
    
    # Close environment
    env.close()
    logger.close()
    
    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}.")
    print(f"Total episodes: {episodes}")
    print(f"Final level reached: {level}")


if __name__ == '__main__':
    main()
