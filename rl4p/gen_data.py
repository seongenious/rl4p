import os 
import argparse
from tqdm import tqdm

from envs.parking_env import ParkingEnv
from utils.config import load_yaml_config
from utils.io import create_transition_dict, save_transitions_dict, batch_transitions_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for RL training.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name (without .npz extension)')
    return parser.parse_args()


def main() -> None:
    """Main function for data generation."""
    args = parse_args()
    if args.output is None:
      raise ValueError('File name is required.')

    print(f'Generating data... Output file: {args.output}.npz')

    # Get configuration in yaml
    config = load_yaml_config('./config/sac.yaml')
    
    num_episodes = config['data_gen']['num_episodes']
    render = config['data_gen']['render']
    file_dir = config['data_gen']['dir']
    file_path = os.path.join(file_dir, args.output + '.npz')

    # Setup environment 
    env = ParkingEnv(config, render)
    print('Environment setup complete.')
        
    # Run episodes
    transitions = []  
    for step in tqdm(range(num_episodes), desc='Run episodes'):
      # Run episode
      obs, info = env.reset()

      while True:
        action = env.action_space.sample()  # Random action
        next_obs, reward, done, truncated, info = env.step(action)

        # Save transition without expert action
        transition = create_transition_dict(
          obs_vehicle_state=obs['vehicle_state'],
          obs_occupancy_grid=obs['occupancy_grid'],
          action=action,
          next_obs_vehicle_state=next_obs['vehicle_state'],
          next_obs_occupancy_grid=next_obs['occupancy_grid'],
          reward=reward,
          done=done,
          truncated=truncated,
          # expert_action=action
        )
        transitions.append(transition)

        # Render environment
        kwargs = {
          'reward': reward,
          'done': done,
          'truncated': truncated,
          'rs_path': info['rs_path']
        }
        env.render(**kwargs)
      
        # Check terminal condition
        if done or truncated:
          break

    # Save transitions
    batched_transitions = batch_transitions_dict(transitions)
    save_transitions_dict(data=batched_transitions, path=file_path)
    print(f'Total {len(transitions)} transitions from {num_episodes} episodes saved to {file_path}')

    # Close environment
    env.close()


if __name__ == '__main__':
  main()  
      
