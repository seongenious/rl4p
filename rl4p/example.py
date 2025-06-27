import jax
import os

from envs.freespace_env import FreespaceEnv
from utils.io import save_transitions, load_transitions
from utils.replay_buffer import ReplayBuffer
from utils.config import load_yaml_config
from utils.plot import plot_transition_distribution


def main():
    # Get configuration in yaml
    config = load_yaml_config('./config/sac.yaml')
    num_samples = config['data_gen']['num_samples']
    seed = config['data_gen']['random_seed']
    file_dir = config['data_gen']['file_dir']
    file_name = config['data_gen']['file_name']
    file_path = os.path.join(file_dir, file_name + ".npz")
    fig_path = os.path.join(file_dir, file_name + ".jpg")
    
    # Setup environment and sample transitions
    env = FreespaceEnv(config, seed=seed)
    transitions = env.sample_transitions(num_samples)
    print(f"Sampled {len(transitions)} transitions")
    
    # Save sample transitions
    save_transitions(transitions, file_path)
    print(f"Saved {len(transitions)} transitions to `{file_path}`")
    
    # Load sample transitions
    data = load_transitions(file_path)
    print(f"Loaded transitions: {list(data.keys())}")
        
    # Display transition distribution
    plot_transition_distribution(data, fig_path)
    print(f"Saved distribution figure to `{fig_path}`")

    if False:
        # Initialize replay buffer and add batch
        obs_dim = data["obs"].shape[1]
        act_dim = data["actions"].shape[1]
        buffer = ReplayBuffer(max_size=10000, obs_dim=obs_dim, act_dim=act_dim)
        buffer.add_batch(data)
        print(f"Replay buffer size: {buffer.size}")

        # Sampling from replay buffer
        rng = jax.random.PRNGKey(0)
        batch = buffer.sample(batch_size=32, rng=rng)
        print("Sampled batch keys: ", list(batch.keys()))
        print("Sampled obs shape: ", batch["obs"].shape)


if __name__ == "__main__":
    main()
