"""SAC training utilities for RL4P project.

This module contains training functions for Soft Actor-Critic (SAC) algorithm.
"""

import time
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import optax
import distrax
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import unfreeze
import jax.tree_util as tree

from model.encoder import SacEncoder
from model.networks import PolicyNetwork, QNetwork, TwinQNetwork, sample_action
from env.datatypes import Observation


def create_networks(rng: jax.random.PRNGKey) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create SAC networks with proper initialization.
            
    Returns:
        Tuple of (networks, network_params).
    """
    encoder = SacEncoder()
    policy = PolicyNetwork()
    critic = TwinQNetwork()
    
    networks = {
        'encoder': encoder,
        'policy': policy,
        'critic': critic
    }
    
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    encoder_params = encoder.init(rng1, jnp.zeros([1, 3, 256, 256, 2]), jnp.zeros([1, 3, 5]))['params']
    policy_params = policy.init(rng2, jnp.zeros([1, 256]))['params']
    critic_params = critic.init(rng3, jnp.zeros([1, 256]), jnp.zeros([1, 2]))['params']
    
    network_params = {
        'encoder': encoder_params,
        'policy': policy_params,
        'critic': critic_params
    }

    return networks, network_params

def create_train_state(networks: Dict[str, Any],
                       network_params: Dict[str, Any],
                       learning_rate: float) -> Dict[str, Any]:
    """Initialize the Flax TrainState with model parameters and optimizer.

    Args:
        networks: Dictionary of networks.
        network_params: Dictionary of network parameters.
        learning_rate: Learning rate for optimizer.

    Returns:
        Initialized TrainState.
    """
    tx = optax.adam(learning_rate)
    states = {
        'encoder': train_state.TrainState.create(
            apply_fn=networks['encoder'].apply, params=network_params['encoder'], tx=tx),
        'policy': train_state.TrainState.create(
            apply_fn=networks['policy'].apply, params=network_params['policy'], tx=tx),
        'critic': train_state.TrainState.create(
            apply_fn=networks['critic'].apply, params=network_params['critic'], tx=tx),
    }
    
    return states

@jax.jit
def train_step(rng: jax.random.PRNGKey,
               states: Dict[str, train_state.TrainState],
               batch: Dict[str, jnp.ndarray],
               alpha: float,
               target_critic_params: Dict[str, jnp.ndarray],
               tau: float,
               gamma: float) -> Tuple[Dict[str, train_state.TrainState], Dict[str, jnp.ndarray]]:
    """Perform a training step.
    
    Args:
        rng: JAX random key.
        states: Dictionary of train states.
        batch: Dictionary of batch data.
        alpha: Entropy coefficient.
        target_critic_params: Target critic parameters.
        tau: Soft update coefficient.
        gamma: Discount factor.
        
    Returns:
        Tuple of (updated states, logs).
    """
    def loss_fn(params):
        # 1. Encode feature
        feat = states['encoder'].apply_fn(
            {'params': params['encoder']}, batch['obs_occupancy_grid'], batch['obs_vehicle_state'])

        # 2. Sample action
        mu, log_std = states['policy'].apply_fn({'params': params['policy']}, feat)

        # 3. Q-values from critic
        q1, q2 = states['critic'].apply_fn({'params': params['critic']}, feat, batch['action'])

        # 4. Update target Q
        next_feat = states['encoder'].apply_fn(
            {'params': params['encoder']}, batch['next_obs_occupancy_grid'], batch['next_obs_vehicle_state'])
        next_mu, next_log_std = states['policy'].apply_fn({'params': params['policy']}, next_feat)
        next_std = jnp.exp(next_log_std)
        next_dist = distrax.Normal(next_mu, next_std)
        next_action = next_dist.sample(seed=rng)
        next_log_prob = next_dist.log_prob(next_action).sum(axis=-1)

        target_q1, target_q2 = states['critic'].apply_fn(
            {'params': target_critic_params}, next_feat, jnp.tanh(next_action))
        min_q = jnp.minimum(target_q1, target_q2)
        target_q = batch['reward'] + gamma * (1.0 - batch['done']) * (min_q - alpha * next_log_prob)
        
        # 5. Critic loss
        critic_loss = jnp.mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2)

        # 6. Actor loss
        q1_pi, _ = states['critic'].apply_fn({'params': params['critic']}, feat, jnp.tanh(next_action))
        actor_loss = jnp.mean(alpha * log_prob - q1_pi)

        total_loss = critic_loss + actor_loss
        return total_loss, {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'log_prob': log_prob
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logs), grads = grad_fn({
        'encoder': states['encoder'].params,
        'policy': states['policy'].params,
        'critic': states['critic'].params
    })

    new_states = {k: states[k].apply_gradients(grads=grads[k]) for k in states}

    return new_states, logs

@jax.jit
def soft_update(target_params: Dict[str, Any], 
                source_params: Dict[str, Any], 
                tau: float) -> Dict[str, Any]:
    """Perform Polyak averaging (soft update) of target parameters.
    
    Args:
        target_params: Target network parameters.
        source_params: Source network parameters.
        tau: Soft update coefficient.
        
    Returns:
        Updated target parameters.
    """
    return jax.tree_util.tree_map(
        lambda t, s: (1 - tau) * t + tau * s, target_params, source_params
    )

@jax.jit
def update_alpha(
    log_alpha: jnp.ndarray,
    alpha_opt_state: optax.OptState,
    log_prob: jnp.ndarray,
    target_entropy: float,
    alpha_optimizer: optax.GradientTransformation
) -> Tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
    """Perform update for entropy coefficient α.

    Args:
        log_alpha: Current log α.
        alpha_opt_state: Optimizer state.
        log_prob: log π(a|s)
        target_entropy: Target entropy value.
        alpha_optimizer: Optimizer for alpha.

    Returns:
        Tuple of updated log_alpha, opt_state, and alpha_loss.
    """
    def loss_fn(log_alpha):
        alpha = jnp.exp(log_alpha)
        loss = -jnp.mean(alpha * (log_prob + target_entropy))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(log_alpha)
    updates, alpha_opt_state = alpha_optimizer.update(grads, alpha_opt_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    return log_alpha, alpha_opt_state, loss

