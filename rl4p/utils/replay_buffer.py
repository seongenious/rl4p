from typing import Optional, Dict
import jax
import jax.numpy as jnp


class ReplayBuffer:
    """JAX-compatible replay buffer with state-only observation.

    Stores transition tuples of (obs, action, next_obs, reward, done).
    Supports single transition addition, batch addition, and random sampling.
    """

    def __init__(self, max_size: int, obs_dim: int, act_dim: int):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of transitions to store.
            obs_dim: Dimension of the observation vector.
            act_dim: Dimension of the action vector.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = jnp.zeros((max_size, obs_dim))
        self.actions = jnp.zeros((max_size, act_dim))
        self.next_obs = jnp.zeros((max_size, obs_dim))
        self.rewards = jnp.zeros((max_size,))
        self.dones = jnp.zeros((max_size,), dtype=jnp.bool_)


    def add_batch(self, data: Dict[str, jnp.ndarray]):
        """Add a batch of transitions to the buffer.

        Args:
            data: A dictionary with keys: 'obs', 'actions', 'next_obs', 'rewards', 'dones'.
                Each value should be a JAX array of shape [N, ...].
        """
        n = data['obs'].shape[0]
        for i in range(n):
            self.add(
                data['obs'][i],
                data['actions'][i],
                data['next_obs'][i],
                data['rewards'][i],
                data['dones'][i]
            )


    def add(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        reward: float,
        done: bool
    ):
        """Add a single transition to the buffer.

        Args:
            obs: Observation vector at time t.
            action: Action vector taken at time t.
            next_obs: Observation vector at time t+1.
            reward: Reward received after taking the action.
            done: Whether the episode terminated after this transition.
        """
        self.obs = self.obs.at[self.ptr].set(obs)
        self.actions = self.actions.at[self.ptr].set(action)
        self.next_obs = self.next_obs.at[self.ptr].set(next_obs)
        self.rewards = self.rewards.at[self.ptr].set(reward)
        self.dones = self.dones.at[self.ptr].set(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size: int, rng: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.
            rng: A JAX PRNGKey used for random sampling.

        Returns:
            A dictionary with keys: 'obs', 'actions', 'next_obs', 'rewards', 'dones'.
            Each value is a JAX array of shape [batch_size, ...].
        """
        idx = jax.random.randint(rng, (batch_size,), minval=0, maxval=self.size)
        return {
            'obs': self.obs[idx],
            'actions': self.actions[idx],
            'next_obs': self.next_obs[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx]
        }
