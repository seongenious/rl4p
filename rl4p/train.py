"""SAC training script for RL4P project.

This module implements a simplified Soft Actor-Critic (SAC) algorithm
for testing purposes. The full BC-SAC pipeline will be implemented separately.
"""

import os
import time
import argparse
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
# import jax
# jax.xla = lambda: None  # dummy
# setattr(jax.xla, "Device", jax.Device)
# from acme.jax import utils
# from acme.jax import types
# from acme import specs
# from acme.jax import replay as acme_replay
# from torch.utils.tensorboard import SummaryWriter


from env.parking_env import ParkingEnv
from model.trainer import (
    create_networks, create_train_state, train_step, soft_update, sample_action
)
# from util.replay_buffer import ReplayBuffer
from util.io import (
    load_yaml_config, save_checkpoint
)


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
    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']
    gamma = config['train']['gamma']
    tau = config['train']['tau']
    alpha = config['train']['alpha']
    
    # Training configurations
    num_episodes = config['train']['num_episodes']
    random_steps = config['train']['random_steps']
    update_after = config['train']['update_after']
    update_every = config['train']['update_every']
    
    # Logger configurations
    log_interval = config['train']['log_interval']
    log_dir = os.path.join("./runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # logger = SummaryWriter(log_dir)
    
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
    print('Initialize replay buffer...')
    # replay_buffer = ReplayBuffer(max_size=config['train']['buffer_size'])

    # Setup environment 
    print('Initialize parking env...')
    env = ParkingEnv(config, config['train']['render'])
    
    print('Complete to initialize.')
    start = time.time()
    
    for step in tqdm(range(num_episodes), desc="Training..."): 
        # Reset episode 
        obs, info = env.reset()
        done = False
        episode_reward = 0
                
        # Run episode
        while not done:
            rng, sub_rng = jax.random.split(rng)
            
            # 1. Encode feature
            feat = networks['encoder'].apply(
                {'params': states['encoder'].params}, obs.occupancy_grid, obs.state)
            
            # 2. Sample action
            # if step < random_steps:
            # action = env.action_space.sample() 
            action, log_prob = sample_action(
                networks['policy'], states['policy'].params, feat, sub_rng)
            
            # 3. Step
            next_obs, reward, done, truncated, info = env.step(action[0])
            
            # 4. Update replay buffer
            # replay_buffer.append(obs, action, reward, next_obs, done)
            obs = next_obs
            
            # 5. Render environment
            kwargs = {'reward': reward, 'done': done, 'truncated': truncated, 'rs_path': info['rs_path']}
            env.render(obs, **kwargs)
            
            if done or truncated: 
                episode_reward = reward
                break
        
        # Network update
        if step >= update_after and step % update_every == 0:
            batch = replay_buffer.sample(batch_size)
            rng, sub_rng = jax.random.split(rng)
            
            states, logs = train_step(sub_rng, states, batch, alpha, target_critic_params, gamma)
            target_critic_params = soft_update(target_critic_params, states['critic'].params, tau)

            if step % log_interval == 0:
                print(f"Episode {step}, Actor Loss: {logs['actor_loss']:.3f}, Critic Loss: {logs['critic_loss']:.3f}")
                save_checkpoint(
                    step=step, 
                    actor_state=states['policy'], 
                    critic1_state=states['critic'], 
                    critic2_state=states['critic'],  
                    ckpt_dir=ckpt_dir
                )

    print(f'Training completed in {time.time() - start:.2f}s.')


if __name__ == '__main__':
    main()
