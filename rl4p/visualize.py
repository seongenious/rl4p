import os
import argparse 
import jax 
import jax.numpy as jnp 
import numpy as np
import pygame 

from flax.training import checkpoints
from flax import linen as nn

from envs.freespace_env import FreespaceEnv
from envs.kinematic_model import simulate
from utils.config import load_yaml_config


class Actor(nn.Module):
    hidden_dims: list
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


def policy_inference(actor_def, actor_params, obs):
    mu, _ = actor_def.apply(actor_params, obs)
    return jnp.tanh(mu)


def draw_trajectory(screen, trajectory, color=(0, 0, 255)):
    for state in trajectory:
        x = int(state.x * 10 + 400)
        y = int(800 - (state.y * 10 + 400))  # Flip y for display
        pygame.draw.circle(screen, color, (x, y), 3)


def visualize_parking(env, actor_def, actor_params):
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("RL4P: Parking Visualization")
    clock = pygame.time.Clock()

    state = env.reset(randomize=True)
    trajectory = [state]

    for _ in range(100):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        obs = jnp.array([[state.x, state.y, state.yaw, state.v, state.dir]], dtype=jnp.float32)
        action = policy_inference(actor_def, actor_params, obs)[0]
        state = env.kinematic_step(state, action)
        trajectory.append(state)

        screen.fill((255, 255, 255))
        draw_trajectory(screen, trajectory)
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to SAC checkpoint directory')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # Load config
    config = load_yaml_config(args.config)
    obs_dim = 5
    act_dim = 2
    hidden_dims = [256, 256]

    # Prepare actor network
    actor_def = Actor(hidden_dims=hidden_dims, action_dim=act_dim)
    dummy_input = jnp.zeros((1, obs_dim))
    actor_params = actor_def.init(jax.random.PRNGKey(0), dummy_input)["params"]

    # Load checkpoint
    ckpt = checkpoints.restore_checkpoint(args.ckpt_dir, target=None)
    actor_params = ckpt['actor_state']['params']

    # Initialize environment
    env = FreespaceEnv(config)

    # Run visualization
    visualize_parking(env, actor_def, actor_params)


if __name__ == '__main__':
    main()