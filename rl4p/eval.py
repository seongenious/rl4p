import os 
import argparse
from tqdm import tqdm

import jax
import jax.numpy as jnp

from env.parking_env import ParkingEnv
from model.trainer import create_networks, create_train_state
from model.networks import sample_action
from util.io import load_yaml_config, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='SAC model path')
    parser.add_argument('--model', type=str, default=None, help='model path')
    return parser.parse_args()
  
def main() -> None:
    """Main function for data generation."""
    args = parse_args()
    
    # Get configuration in yaml
    config = load_yaml_config('./config/sac.yaml')
    num_episodes = config['eval']['num_episodes']
    
    # Load checkpoint
    networks, params, states = None, None, None
    if args.model is not None:
      rng = jax.random.PRNGKey(0)
      networks, _ = create_networks(rng)
      params = load_checkpoint(args.model) if args.model is not None else None
      states = create_train_state(networks, params)
    
    # Setup environment 
    env = ParkingEnv(config, level=-1, render=True)
        
    # Run episodes
    for _ in tqdm(range(num_episodes), desc='Run episodes'):
      # Run episode
      obs, info = env.reset()
      episode_reward = 0
      
      while True:
        if args.model is not None:
            # Expand dimension of observation
            occupancy_grid = jnp.expand_dims(obs.occupancy_grid, axis=0)
            state = jnp.expand_dims(obs.state, axis=0)
            
            # Encode feature
            feat = networks['encoder'].apply(
                {'params': states['encoder'].params}, occupancy_grid, state)
            
            # Sample action
            action = sample_action(states['actor'], states['actor'].params, feat)[0]
            action = action[0]
        else:
            # Sample action
            action = env.action_space.sample()
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
                
        # Render environment
        episode_reward += reward
        kwargs = {
          'action': action,
          'reward': episode_reward,
          'done': done,
          'truncated': truncated,
          'rs_path': info['rs_path'],
        }
        env.render(obs, **kwargs)
                      
        # Check terminal condition
        if done or truncated:
            break

    # Close environment
    env.close()


if __name__ == '__main__':
  main()  
      
