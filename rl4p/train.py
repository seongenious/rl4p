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
# from torch.utils.tensorboard import SummaryWriter

from envs.parking_env import ParkingEnv
from models.networks import PolicyNetwork, QNetwork, sample_action
from models.trainer import (
    create_train_state, create_networks, initialize_train_states,
    compute_target_q, update_critic, update_actor, update_alpha, soft_update
)
from utils.replay_buffer import ReplayBuffer
from utils.io import (
    load_yaml_config, load_transitions_into_buffer, 
    save_checkpoint, load_checkpoint
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
        
    # Training configurations
    num_episodes = config['train']['num_episodes']
    random_steps = config['train']['random_steps']
    update_after = config['train']['update_after']
    update_every = config['train']['update_every']
    updates_per_step = config['train']['updates_per_step']
    
    # Logger configurations
    log_interval = config['train']['log_interval']
    log_dir = os.path.join("./runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # logger = SummaryWriter(log_dir)
    
    # Other configurations
    ckpt_dir = config['train']['dir']
    ckpt_dir = os.path.join(ckpt_dir, args.output)
    
    # Create networks
    print('Initialize SAC model...')
    rng = jax.random.PRNGKey(42)
    actor, critic = create_networks()
    actor_state, critic1_state, critic2_state = initialize_train_states(
        rng, actor, critic, learning_rate=learning_rate)
    target_critic1_params = critic1_state.params
    target_critic2_params = critic2_state.params
    
    # Alpha-related variables for entropy regularization
    log_alpha = jnp.array(0.0)  # log(alpha)
    target_entropy = -2  # common default (-action_dim)
    alpha_optimizer = optax.adam(3e-4)
    alpha_opt_state = alpha_optimizer.init(log_alpha)
    
    # Create replay buffer
    print('Initialize replay buffer...')
    replay_buffer = ReplayBuffer(max_size=config['train']['buffer_size'])

    # Setup environment 
    print('Initialize parking env...')
    env = ParkingEnv(config, config['train']['render'])
    
    print('Complete to initialize.')
    start = time.time()
    
    for step in tqdm(range(num_episodes), desc="Training..."):
        # Reset episode 
        obs, info = env.reset()
        episode_reward = 0
                
        # Run episode
        while True:
            # Policy
            rng, action_rng = jax.random.split(rng)
            action = env.action_space.sample() \
                if step < random_steps else sample_action(actor, actor_state.params, obs, action_rng)[0]
            
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Stack buffer
            replay_buffer.add(
                obs_vehicle_state=obs['vehicle_state'],
                obs_occupancy_grid=obs['occupancy_grid'],
                action=action,
                next_obs_vehicle_state=next_obs['vehicle_state'],
                next_obs_occupancy_grid=next_obs['occupancy_grid'],
                reward=reward,
                done=done,
                truncated=truncated,
                # expert_action=action,    
            )

            # Render environment
            kwargs = {
                'reward': reward, 
                'done': done, 
                'truncated': truncated, 
                'rs_path': info['rs_path']
            }
            
            env.render(**kwargs)
            
            if done or truncated: 
                episode_reward = reward
                break
        
        # Network update
        if step >= update_after and step % update_every == 0:
            for _ in range(updates_per_step):
                rng, sample_rng = jax.random.split(rng)
                batch = replay_buffer.sample(batch_size=batch_size, rng=sample_rng)
                
                # 1. compute target Q using target critic and actor
                alpha = jnp.exp(log_alpha)
                rng, target_rng = jax.random.split(rng)
                target_q, target_rng = compute_target_q(
                    actor_params=actor_state.params,
                    target_critic1_params=target_critic1_params,
                    target_critic2_params=target_critic2_params,
                    critic_apply_fn=critic1_state.apply_fn,  # both critics share same fn
                    actor_apply_fn=actor_state.apply_fn,
                    rng=target_rng,
                    next_obs=batch['next_obs'],
                    reward=batch['reward'],
                    done=batch['done'],
                    gamma=gamma,
                    alpha=alpha
                )
                
                # 2. critic update
                critic1_state, critic1_loss = update_critic(
                    critic1_state, target_q, batch['obs'], batch['action']
                )
                critic2_state, critic2_loss = update_critic(
                    critic2_state, target_q, batch['obs'], batch['action']
                )
                
                # 3. actor update
                rng, actor_rng = jax.random.split(rng)
                actor_state, actor_loss, log_prob = update_actor(
                    actor_state=actor_state,
                    critic_params=critic1_state.params,
                    critic_apply_fn=critic1_state.apply_fn,
                    rng=actor_rng,
                    obs=batch['obs'],
                    alpha=alpha
                )

                # 4. alpha update
                log_alpha, alpha_opt_state, alpha_loss = update_alpha(
                    log_alpha=log_alpha,
                    alpha_opt_state=alpha_opt_state,
                    log_prob=log_prob,
                    target_entropy=target_entropy,
                    alpha_optimizer=alpha_optimizer
                )

                # 5. soft update
                target_critic1_params = soft_update(
                    target_critic1_params, critic1_state.params, tau
                )
                target_critic2_params = soft_update(
                    target_critic2_params, critic2_state.params, tau
                )

        # Log and save checkpoint
        if step >= update_after and step % log_interval == 0:
            # logger.add_scalar('actor_loss', actor_loss, step)
            # logger.add_scalar('critic/loss1', critic1_loss, step)
            # logger.add_scalar('critic/loss2', critic2_loss, step)
            # logger.add_scalar('alpha_loss', alpha_loss, step)
            # logger.add_scalar('reward', episode_reward, step)
            
            save_checkpoint(
                step=step, 
                actor_state=actor_state, 
                critic1_state=critic1_state, 
                critic2_state=critic2_state, 
                log_alpha=log_alpha, 
                alpha_opt_state=alpha_opt_state, 
                ckpt_dir=ckpt_dir
            )

    print(f'Training completed in {time.time() - start:.2f}s.')


if __name__ == '__main__':
    main()
