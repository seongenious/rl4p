import jax
import jax.numpy as jnp

from envs.freespace_env import FreespaceEnv
from utils.io import save_transitions, load_transitions
from utils.replay_buffer import ReplayBuffer
from utils.config import load_yaml_config

CONFIG_PATH = './config/sac.yaml'
FILE_PATH = './data/sample_example.npz'


def main():
    # Get configuration in yaml
    config = load_yaml_config(CONFIG_PATH)
    
    # Setup environment and sample transitions
    env = FreespaceEnv(config, seed=42)
    transitions = env.sample_transitions(1000)

    # Save sample transitions
    save_transitions(transitions, FILE_PATH)
    print(f"Saved {len(transitions)} transitions to {FILE_PATH}")

    # Load sample transitions
    data = load_transitions(FILE_PATH)
    print(f"Loaded transitions: {list(data.keys())}")

    # Initialize replay buffer and add batch
    obs_dim = data["obs"].shape[1]
    act_dim = data["actions"].shape[1]
    buffer = ReplayBuffer(max_size=10000, obs_dim=obs_dim, act_dim=act_dim)
    buffer.add_batch(data)
    print(f"Replay buffer size: {buffer.size}")

    # Sampling from replay buffer
    rng = jax.random.PRNGKey(0)
    batch = buffer.sample(batch_size=32, rng=rng)
    print("Sampled batch keys:", list(batch.keys()))
    print("Sampled obs shape:", batch["obs"].shape)


if __name__ == "__main__":
    main()
