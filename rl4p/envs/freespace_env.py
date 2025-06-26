import jax
import jax.numpy as jnp
from typing import Tuple, List

from envs.base_env import BaseEnv
from envs.datatypes import State, Action, Transition
from envs.reward import reward_fn
from envs.kinematic_model import simulate
from utils.unit import kph2mps, deg2rad, mod2pi


class FreespaceEnv(BaseEnv):
    """A simple kinematic freespace environment for RL-based parking using JAX."""

    def __init__(self, config: dict, seed: int = 0):
        """Initialize the environment.

        Args:
            config: Configuration dictionary containing environment and vehicle parameters.
            seed: Random seed for JAX PRNG key.
        """
        super().__init__(config, seed)


    def reset(self) -> Tuple[State, State]:
        """Reset the environment and sample a new goal.

        Returns:
            A tuple of (initial_state, goal_state)
        """
        self.rng_key, *subkeys = jax.random.split(self.rng_key, 6)

        x = jax.random.uniform(subkeys[0], (), minval=-self.max_range, maxval=self.max_range)
        y = jax.random.uniform(subkeys[1], (), minval=-self.max_range, maxval=self.max_range)
        yaw = jax.random.uniform(subkeys[2], (), minval=-jnp.pi, maxval=jnp.pi)
        v = jax.random.uniform(subkeys[3], (), minval=-self.max_speed, maxval=self.max_speed)
        dir = jnp.where(v >= 0, 1, -1)
        
        self.state = State(x=x, y=y, yaw=yaw, v=v, dir=dir)
        self.goal = State(x=0., y=0., yaw=0., v=0., dir=1)

        return self.state, self.goal


    def step(self, action: Action) -> Transition:
        """Step forward in the environment using the given action.

        Args:
            action: Action to apply (delta, accel)

        Returns:
            A Transition object (s, a, s', r, done)
        """
        prev_state = self.state 
        next_state = simulate(
            prev_state, action, self.dt, self.wheelbase, 
            self.max_speed, self.max_accel, self.max_steering_angle
        )
        reward, done = self.compute_reward(next_state)
        self.state = next_state
        return Transition(
            obs=prev_state, action=action, next_obs=next_state, reward=reward, done=done)


    def compute_reward(self, state: State) -> Tuple[float, bool]:
        """Use external reward function."""
        return reward_fn(state, self.goal, self.config)
    
    
    def get_observation(self) -> jnp.ndarray:
        """Return observation vector [x, y, yaw, v, dir]."""
        return jnp.array([
            self.state.x, self.state.y, self.state.yaw, self.state.v, self.state.dir
            ], dtype=jnp.float32
        )


    def render(self):
        """Render is not implemented yet."""
        raise NotImplementedError("Rendering is not implemented for FreespaceEnv.")