from typing import Sequence, Tuple, Dict

import os
import time
import numpy as np

import jax
import jax.numpy as jnp
import optax
import distrax
import flax.linen as nn
from flax.training import train_state, checkpoints
from functools import partial

from utils.io import load_transitions_into_buffer
from utils.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """Stochastic Gaussian policy network."""

    hidden_dims: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
        mu = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20, 2)
        std = jnp.exp(log_std)
        return mu, std


class Critic(nn.Module):
    """Q-value network."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, -1)


def save_checkpoint(step: int,
                    actor_state,
                    critic1_state,
                    critic2_state,
                    log_alpha,
                    alpha_opt_state,
                    ckpt_dir: str):
    """Saves SAC training checkpoint."""
    checkpoints.save_checkpoint(
        ckpt_dir,
        target={
            'step': step,
            'actor_state': actor_state,
            'critic1_state': critic1_state,
            'critic2_state': critic2_state,
            'log_alpha': log_alpha,
            'alpha_opt_state': alpha_opt_state
        },
        step=step,
        overwrite=False
    )


def load_checkpoint(ckpt_dir: str):
    """Loads checkpoint if available, else returns None."""
    return checkpoints.restore_checkpoint(ckpt_dir, target=None)


def create_train_state(rng: jax.random.PRNGKey,
                       model: nn.Module,
                       inputs: Tuple,
                       learning_rate: float) -> train_state.TrainState:
    """Initializes the Flax TrainState with model parameters and optimizer.

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
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_target_q(
    actor_params,
    target_critic1_params,
    target_critic2_params,
    critic_apply_fn,
    actor_apply_fn,
    rng,
    next_obs,
    reward,
    done,
    gamma,
    alpha,
):
    """Computes SAC target Q using target critics and policy at next state."""
    mu, std = actor_apply_fn(actor_params, next_obs)
    rng, subkey = jax.random.split(rng)
    dist = distrax.Normal(mu, std)
    action = dist.sample(seed=subkey)
    log_prob = dist.log_prob(action).sum(axis=-1)
    tanh_action = jnp.tanh(action)

    q1 = critic_apply_fn(target_critic1_params, next_obs, tanh_action)
    q2 = critic_apply_fn(target_critic2_params, next_obs, tanh_action)
    min_q = jnp.minimum(q1, q2)
    target_q = reward + gamma * (1.0 - done) * (min_q - alpha * log_prob)

    return target_q, rng


def critic_loss_fn(critic_apply, critic_params, obs, act, target_q) -> Tuple[jnp.ndarray, Dict]:
    """Computes the critic loss.

    Args:
        critic_apply: Critic apply function.
        critic_params: Parameters of critic.
        obs: Batch of observations.
        act: Batch of actions.
        target_q: Target Q-values.

    Returns:
        Scalar critic loss and auxiliary data.
    """
    q = critic_apply(critic_params, obs, act)
    loss = jnp.mean((q - target_q) ** 2)
    return loss, {'q': q}


@jax.jit
def update_critic(critic_state: train_state.TrainState,
                  target_q: jnp.ndarray,
                  obs: jnp.ndarray,
                  act: jnp.ndarray) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """Performs a critic update step.

    Args:
        critic_state: Current TrainState for critic.
        target_q: Target Q-values.
        obs: Observations.
        act: Actions.

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
                 critic_params: Dict,
                 rng: jax.random.PRNGKey,
                 obs: jnp.ndarray,
                 alpha: float) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
    """Performs actor update step using reparameterization trick.

    Returns:
        Tuple of (updated actor state, scalar loss, log_prob of actions).
    """
    def actor_loss_fn(params, rng):
        mu, std = actor_state.apply_fn(params, obs)
        key, subkey = jax.random.split(rng)
        normal = distrax.Normal(loc=mu, scale=std)
        action = normal.sample(seed=subkey)
        log_prob = normal.log_prob(action).sum(axis=-1)
        tanh_action = jnp.tanh(action)

        q = Critic(hidden_dims=[256, 256]).apply(critic_params, obs, tanh_action)
        loss = (alpha * log_prob - q).mean()
        return loss, log_prob

    (loss, log_prob), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, rng)
    actor_state = actor_state.apply_gradients(grads=grads)
    return actor_state, loss, log_prob


@jax.jit
def soft_update(target_params, source_params, tau: float) -> dict:
    """Performs Polyak averaging (soft update) of target parameters."""
    return jax.tree_util.tree_map(lambda tp, sp: (1 - tau) * tp + tau * sp, target_params, source_params)


def main():
    """Main training loop for SAC (simplified, critic only)."""
    obs_dim = 5
    act_dim = 2
    hidden_dims = [256, 256]
    batch_size = 256
    learning_rate = 3e-4
    num_steps = 1000
    gamma = 0.99
    tau = 0.005

    # Initialize random key
    rng = jax.random.PRNGKey(0)

    # Initialize actor and critic networks
    actor = Actor(hidden_dims=hidden_dims, action_dim=act_dim)
    critic = Critic(hidden_dims=hidden_dims)

    rng, actor_key, critic1_key, critic2_key = jax.random.split(rng, 4)
    
    actor_state = create_train_state(actor_key, actor, (jnp.zeros((1, obs_dim)),), learning_rate)
    critic1_state = create_train_state(critic1_key, critic, (jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim))), learning_rate)
    critic2_state = create_train_state(critic2_key, critic, (jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim))), learning_rate)
    target_critic1_params = critic1_state.params
    target_critic2_params = critic2_state.params
    
    # alpha-related variables
    log_alpha = jnp.array(0.0)  # log(alpha)
    alpha_opt = optax.adam(learning_rate)
    alpha_opt_state = alpha_opt.init(log_alpha)
    target_entropy = -act_dim  # common default
    
    alpha_optimizer = optax.adam(3e-4)
    alpha_opt_state = alpha_optimizer.init(log_alpha)
        
    @jax.jit
    def update_alpha(log_alpha: jnp.ndarray,
                    alpha_opt_state: optax.OptState,
                    log_prob: jnp.ndarray,
                    target_entropy: float) -> Tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
        """Performs update for entropy coefficient α.

        Args:
            log_alpha: Current log α.
            alpha_opt_state: Optimizer state.
            log_prob: log π(a|s)
            target_entropy: Target entropy value.
            alpha_optimizer: optax optimizer for log α.

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

    print("Training started...")
    start = time.time()
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Load transitions into buffer")
    replay_buffer = load_transitions_into_buffer(
        folder_path='./data/', max_size=50000, obs_dim=obs_dim, act_dim=act_dim)

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
            critic1_state, target_q, batch['obs'], batch['actions'])
        critic2_state, loss2 = update_critic(
            critic2_state, target_q, batch['obs'], batch['actions'])
        
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
            target_entropy=target_entropy
        )

        # 5. soft update
        target_critic1_params = soft_update(target_critic1_params, critic1_state.params, tau)
        target_critic2_params = soft_update(target_critic2_params, critic2_state.params, tau)

        if step % 100 == 0 or step == 1:
            print(f"[{step}] actor_loss={actor_loss:.4f}, critic1={loss1:.4f}, critic2={loss2:.4f}, alpha={alpha:.4f}, alpha_loss={alpha_loss:.4f}")
            save_checkpoint(step, actor_state, critic1_state, critic2_state, log_alpha, alpha_opt_state, ckpt_dir)

    print(f"Training completed in {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
