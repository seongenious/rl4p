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

from models.networks import PolicyNetwork, QNetwork, sample_action


def create_train_state(rng: jax.random.PRNGKey,
                       model: Any,
                       inputs: Tuple,
                       learning_rate: float) -> train_state.TrainState:
    """Initialize the Flax TrainState with model parameters and optimizer.

    Args:
        rng: JAX random key.
        model: Flax Module (Actor or Critic).
        inputs: Dummy inputs for model initialization.
        learning_rate: Learning rate for optimizer.

    Returns:
        Initialized TrainState.
    """
    params = model.init(rng, *inputs)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def create_networks() -> Tuple[PolicyNetwork, QNetwork]:
    """Create SAC networks with proper initialization.
            
    Returns:
        Tuple of (policy_network, q_network).
    """
    policy_net = PolicyNetwork()
    q_net = QNetwork()
    
    return policy_net, q_net


def initialize_train_states(
    rng: jax.random.PRNGKey,
    policy_net: PolicyNetwork,
    q_net: QNetwork,
    learning_rate: float = 3e-4
) -> Tuple[train_state.TrainState, train_state.TrainState, train_state.TrainState]:
    """Initialize SAC training states.
    
    Args:
        rng: JAX random key.
        policy_net: Policy network.
        q_net: Q network.
        learning_rate: Learning rate for optimizers.
        
    Returns:
        Tuple of (actor_state, critic1_state, critic2_state).
    """
    # Create dummy inputs for initialization
    dummy_obs = {
        'vehicle_state': jnp.zeros((1, 5)),
        'occupancy_grid': jnp.zeros((1, 256, 256))
    }
    dummy_action = jnp.zeros((1, 2))
    
    # Split RNG for different initializations
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    
    # Initialize states
    actor_state = create_train_state(rng1, policy_net, (dummy_obs,), learning_rate)
    critic1_state = create_train_state(rng2, q_net, (dummy_obs, dummy_action), learning_rate)
    critic2_state = create_train_state(rng3, q_net, (dummy_obs, dummy_action), learning_rate)
    
    return actor_state, critic1_state, critic2_state 


def compute_target_q(
    actor_params: Dict[str, Any],
    target_critic1_params: Dict[str, Any],
    target_critic2_params: Dict[str, Any],
    critic_apply_fn: Any,
    actor_apply_fn: Any,
    rng: jax.random.PRNGKey,
    next_obs: Dict[str, jnp.ndarray],
    reward: jnp.ndarray,
    done: jnp.ndarray,
    gamma: float,
    alpha: float,
) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
    """Compute SAC target Q using target critics and policy at next state.
    
    Args:
        actor_params: Actor network parameters.
        target_critic1_params: First target critic parameters.
        target_critic2_params: Second target critic parameters.
        critic_apply_fn: Critic apply function.
        actor_apply_fn: Actor apply function.
        rng: JAX random key.
        next_obs: Next observations dictionary with 'vehicle_state' and 'occupancy_grid'.
        reward: Rewards, shape: (batch_size,).
        done: Done flags, shape: (batch_size,).
        gamma: Discount factor.
        alpha: Entropy coefficient.
        
    Returns:
        Target Q-values and updated random key.
    """
    # Get action distribution from actor
    mu, log_std = actor_apply_fn(actor_params, next_obs)
    
    # Sample action and compute log probability
    rng, subkey = jax.random.split(rng)
    std = jnp.exp(log_std)
    dist = distrax.Normal(mu, std)
    action = dist.sample(seed=subkey)
    log_prob = dist.log_prob(action).sum(axis=-1)
    tanh_action = jnp.tanh(action)

    q1 = critic_apply_fn(target_critic1_params, next_obs, tanh_action)
    q2 = critic_apply_fn(target_critic2_params, next_obs, tanh_action)
    min_q = jnp.minimum(q1, q2)
    
    # Compute target Q
    target_q = reward + gamma * (1.0 - done) * (min_q - alpha * log_prob)

    return target_q, rng


def critic_loss_fn(critic_apply: Any, 
                   critic_params: Dict[str, Any], 
                   obs: Dict[str, jnp.ndarray], 
                   act: jnp.ndarray, 
                   target_q: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute the critic loss.

    Args:
        critic_apply: Critic apply function.
        critic_params: Parameters of critic.
        obs: Batch of observations dictionary.
        act: Batch of actions, shape: (batch_size, act_dim).
        target_q: Target Q-values, shape: (batch_size,).

    Returns:
        Scalar critic loss and auxiliary data.
    """
    q = critic_apply(critic_params, obs, act)
    loss = jnp.mean((q - target_q) ** 2)
    return loss, {'q': q}


@jax.jit
def update_critic(critic_state: train_state.TrainState,
                  target_q: jnp.ndarray,
                  obs: Dict[str, jnp.ndarray],
                  act: jnp.ndarray) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """Perform a critic update step.

    Args:
        critic_state: Current TrainState for critic.
        target_q: Target Q-values, shape: (batch_size,).
        obs: Observations dictionary.
        act: Actions, shape: (batch_size, act_dim).

    Returns:
        Updated critic TrainState and critic loss.
    """
    def loss_fn(params):
        q = critic_state.apply_fn(params, obs, act)
        loss = jnp.mean((q - target_q) ** 2)
        return loss, {'q': q}
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=grads)
    return new_critic_state, loss


@jax.jit
def update_actor(actor_state: train_state.TrainState,
                 critic_params: Dict[str, Any],
                 rng: jax.random.PRNGKey,
                 obs: Dict[str, jnp.ndarray],
                 alpha: float) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
    """Perform actor update step using reparameterization trick.

    Args:
        actor_state: Current TrainState for actor.
        critic_params: Critic network parameters.
        rng: JAX random key.
        obs: Observations dictionary.
        alpha: Entropy coefficient.

    Returns:
        Tuple of (updated actor state, scalar loss, log_prob of actions).
    """
    def actor_loss_fn(params, rng):
        mu, log_std = actor_state.apply_fn(params, obs)
        key, subkey = jax.random.split(rng)
        std = jnp.exp(log_std)
        normal = distrax.Normal(loc=mu, scale=std)
        action = normal.sample(seed=subkey)
        log_prob = normal.log_prob(action).sum(axis=-1)
        tanh_action = jnp.tanh(action)

        q = QNetwork().apply(critic_params, obs, tanh_action)
        loss = (alpha * log_prob - q).mean()
        return loss, log_prob

    (loss, log_prob), grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_state.params, rng)
    actor_state = actor_state.apply_gradients(grads=grads)
    return actor_state, loss, log_prob


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
        lambda tp, sp: (1 - tau) * tp + tau * sp, 
        target_params, source_params
    )


@jax.jit
def update_alpha(log_alpha: jnp.ndarray,
                alpha_opt_state: optax.OptState,
                log_prob: jnp.ndarray,
                target_entropy: float,
                alpha_optimizer: optax.GradientTransformation) -> Tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
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
    def alpha_loss_fn(log_alpha):
        alpha = jnp.exp(log_alpha)
        loss = -jnp.mean(alpha * (log_prob + target_entropy))
        return loss

    loss, grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    updates, alpha_opt_state = alpha_optimizer.update(grads, alpha_opt_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    return log_alpha, alpha_opt_state, loss