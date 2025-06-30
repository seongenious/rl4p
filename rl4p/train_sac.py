"""SAC training script for RL4P project.

This module implements a simplified Soft Actor-Critic (SAC) algorithm
for testing purposes. The full BC-SAC pipeline will be implemented separately.
"""

import os
import time

import jax
import jax.numpy as jnp
import optax

from utils.io import load_transitions_into_buffer
from models.sac_networks import PolicyNetwork, QNetwork
from models.sac_trainer import (
    save_checkpoint, create_train_state, compute_target_q,
    update_critic, update_actor, soft_update, update_alpha
)


def main():
    """Run main training loop for SAC (simplified, critic only).
    
    Note: This is a simplified SAC implementation for testing purposes.
    The full BC-SAC pipeline will be implemented separately.
    """
    # Network configuration
    obs_dim = 5  # [x, y, yaw, v, dir] - simplified for testing
    act_dim = 2  # [delta, accel]
    hidden_dims = [256, 256]
    batch_size = 256
    learning_rate = 3e-4
    num_steps = 1000
    gamma = 0.99
    tau = 0.005

    # Initialize random key
    rng = jax.random.PRNGKey(0)

    # Initialize actor and critic networks using the shared network definitions
    actor = PolicyNetwork(action_dim=act_dim, hidden_dims=hidden_dims)
    critic = QNetwork(hidden_dims=hidden_dims)

    rng, actor_key, critic1_key, critic2_key = jax.random.split(rng, 4)
    
    # Create training states
    actor_state = create_train_state(
        actor_key, actor, (jnp.zeros((1, obs_dim)),), learning_rate
    )
    critic1_state = create_train_state(
        critic1_key, critic, 
        (jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim))), learning_rate
    )
    critic2_state = create_train_state(
        critic2_key, critic, 
        (jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim))), learning_rate
    )
    target_critic1_params = critic1_state.params
    target_critic2_params = critic2_state.params
    
    # Alpha-related variables for entropy regularization
    log_alpha = jnp.array(0.0)  # log(alpha)
    target_entropy = -act_dim  # common default
    
    alpha_optimizer = optax.adam(3e-4)
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    print('Training started...')
    start = time.time()
    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    print('Load transitions into buffer')
    replay_buffer = load_transitions_into_buffer(
        folder_path='./data/', max_size=50000, obs_dim=obs_dim, act_dim=act_dim
    )

    for step in range(1, num_steps + 1):
        # Sample data
        batch = replay_buffer.sample(batch_size=batch_size, rng=rng)

        # 1. compute target Q using target critic and actor
        alpha = jnp.exp(log_alpha)
        target_q, rng = compute_target_q(
            actor_params=actor_state.params,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            critic_apply_fn=critic1_state.apply_fn,  # both critics share same fn
            actor_apply_fn=actor_state.apply_fn,
            rng=rng,
            next_obs=batch['next_obs'],
            reward=batch['rewards'],
            done=batch['dones'],
            gamma=gamma,
            alpha=alpha
        )
        
        # 2. critic update
        critic1_state, loss1 = update_critic(
            critic1_state, target_q, batch['obs'], batch['actions']
        )
        critic2_state, loss2 = update_critic(
            critic2_state, target_q, batch['obs'], batch['actions']
        )
        
        # 3. actor update
        actor_state, actor_loss, log_prob = update_actor(
            actor_state=actor_state,
            critic_params=critic1_state.params,
            rng=rng,
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

        if step % 100 == 0 or step == 1:
            print(
                f'[{step}] actor_loss={actor_loss:.4f}, critic1={loss1:.4f}, '
                f'critic2={loss2:.4f}, alpha={alpha:.4f}, alpha_loss={alpha_loss:.4f}'
            )
            save_checkpoint(
                step, actor_state, critic1_state, critic2_state, 
                log_alpha, alpha_opt_state, ckpt_dir
            )

    print(f'Training completed in {time.time() - start:.2f}s.')


if __name__ == '__main__':
    main()
