import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple, List

from envs.datatypes import State, Action, Transition
from envs.kinematic_model import simulate
from utils.unit import kph2mps, deg2rad, mod2pi

class BaseEnv(ABC):
    """Abstract base class for JAX-based reinforcement learning environments."""

    def __init__(self, config: dict, seed: int = 0):
        """Initialize base environment.

        Args:
            config: Configuration dictionary.
            seed: Random seed for JAX.
        """
        if config is None:
            raise ValueError("`config` must not be None.")
        
        self.config = config
        self.rng_key = jax.random.PRNGKey(seed)

        self.state: State = None
        self.goal: State = None
        
        self.dt = config['env']['dt']
        self.max_range = config['env']['max_range']
        
        self.wheelbase = config['vehicle_config']['wheelbase']
        self.max_speed = kph2mps(config['vehicle_config']['max_speed'])
        self.max_accel = config['vehicle_config']['max_accel']
        self.max_steering_angle = deg2rad(config['vehicle_config']['max_steering_angle'])
        
        self.reset()

    @abstractmethod
    def reset(self) -> Tuple[State, State]:
        """Reset environment state and goal."""
        raise NotImplementedError("reset() must be implemented by subclass.")

    @abstractmethod
    def step(self, action: Action) -> Transition:
        """Apply action and return transition."""
        raise NotImplementedError("step() must be implemented by subclass.")

    @abstractmethod
    def compute_reward(self, state: State) -> Tuple[float, bool]:
        """Compute reward given the next state."""
        raise NotImplementedError("compute_reward() must be implemented by subclass.")

    @abstractmethod
    def get_observation(self) -> jnp.ndarray:
        """Return current observation (for policy input)."""
        raise NotImplementedError("get_observation() must be implemented by subclass.")

    @abstractmethod
    def render(self):
        """Optional: visualize the environment."""
        raise NotImplementedError("render() must be implemented by subclass.")

    def sample_transitions(self, num_samples: int) -> List[Transition]:
        """Sample multiple transitions using random actions.

        Args:
            num_samples: Number of transitions to sample.

        Returns:
            A list of transitions.
        """
        transitions = []
        for _ in range(num_samples):
            self.rng_key, subkey1, subkey2 = jax.random.split(self.rng_key, 3)
            delta = jax.random.uniform(subkey1, (), minval=-1.0, maxval=1.0)
            accel = jax.random.uniform(subkey2, (), minval=-1.0, maxval=1.0)
            action = Action(delta=delta, accel=accel)
            transitions.append(self.step(action))
        return transitions
