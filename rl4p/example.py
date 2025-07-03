import os 
from tqdm import tqdm

from env.parking_env import ParkingEnv
from util.io import load_yaml_config


def main() -> None:
    """Main function for data generation."""
    # Get configuration in yaml
    config = load_yaml_config('./config/sac.yaml')
    num_episodes = 100

    # Setup environment 
    env = ParkingEnv(config, True)
        
    # Run episodes
    for _ in tqdm(range(num_episodes), desc='Run episodes'):
      # Run episode
      obs, info = env.reset()

      while True:
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)

        # Render environment
        kwargs = {
          'reward': reward,
          'done': done,
          'truncated': truncated,
          'rs_path': info['rs_path']
        }
        env.render(obs, **kwargs)
      
        # Check terminal condition
        if done or truncated:
          break

    # Close environment
    env.close()


if __name__ == '__main__':
  main()  
      
