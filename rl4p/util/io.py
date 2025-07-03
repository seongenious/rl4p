from typing import List, Dict, Any, Optional
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state, checkpoints

from util.replay_buffer import ReplayBuffer


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary.

    Args:
        path: The file path to the YAML configuration file.

    Returns:
        A dictionary representing the parsed YAML content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the YAML file has invalid syntax.

    Example:
        config = load_yaml_config("config.yaml")
        print(config['vehicle_config']['max_speed'])
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def print_config(config: Dict[str, Any]):
    """Pretty-print a configuration dictionary to the console.

    Args:
        config: The configuration dictionary to print.

    Example:
        config = load_yaml_config("config.yaml")
        print_config(config)
    """
    import pprint
    pprint.pprint(config)
    
    
def save_transitions_dict(data: Dict[str, jnp.ndarray], path: str):
    """Save transition data as a .npz file.

    Args:
        data: Dictionary containing transition data with keys:
            - 'obs_vehicle_state': [N, 5] (x, y, yaw, v, dir)
            - 'obs_occupancy_grid': [N, 256, 256] (optional)
            - 'action': [N, 2] (delta, accel)
            - 'next_obs_vehicle_state': [N, 5]
            - 'next_obs_occupancy_grid': [N, 256, 256] (optional)
            - 'reward': [N]
            - 'done': [N]
            - 'truncated': [N]
            - 'expert_action': [N, 2] (optional)
        path: Destination file path for saving the .npz archive.
    """
    # Convert JAX arrays to numpy arrays
    np_data = {key: np.array(value) for key, value in data.items()}
    np.savez(path, **np_data)


def load_transitions_dict(path: str) -> Dict[str, jnp.ndarray]:
    """Load transition data from a .npz file.

    Args:
        path: File path to a .npz archive containing saved transitions.

    Returns:
        A dictionary mapping keys to JAX arrays.

    Example:
        data = load_transitions_dict("offline_data.npz")
        print(data['obs_vehicle_state'].shape)  # (N, 5)
        print(data['obs_occupancy_grid'].shape)  # (N, 256, 256) if exists
    """
    data = np.load(path)
    return {k: jnp.array(data[k]) for k in data}


def save_replay_buffer(buffer: ReplayBuffer, path: str):
    """Save a replay buffer to disk.

    Args:
        buffer: ReplayBuffer object to save.
        path: Destination file path.
    """
    data = buffer.get_all_data()
    save_transitions_dict(data, path)


def load_replay_buffer(path: str, max_size: int, include_occupancy_grid: bool = False, 
                      include_expert: bool = False) -> ReplayBuffer:
    """Load data into a replay buffer.

    Args:
        path: File path to a .npz archive.
        max_size: Maximum size of the replay buffer.
        include_occupancy_grid: Whether the data includes occupancy grid.
        include_expert: Whether the data includes expert actions.

    Returns:
        ReplayBuffer object filled with loaded data.
    """
    data = load_transitions_dict(path)
    
    # Create buffer with appropriate settings
    buffer = ReplayBuffer(
        max_size=max_size,
        obs_dim=5,  # vehicle state dimension
        act_dim=2,  # action dimension
        include_occupancy_grid=include_occupancy_grid,
        include_expert=include_expert
    )
    
    # Add data to buffer
    buffer.add_batch(data)
    
    return buffer


def load_transitions_into_buffer(
        folder_path: str, max_size: int, include_occupancy_grid: bool = False,
        include_expert: bool = False) -> ReplayBuffer:
    """Load all .npz files in a folder into a replay buffer.

    Args:
        folder_path: Path to the folder containing .npz transition files.
        max_size: Maximum size of the replay buffer.
        include_occupancy_grid: Whether to include occupancy grid data.
        include_expert: Whether to include expert action data.

    Returns:
        ReplayBuffer object filled with loaded transitions.
    """
    buffer = ReplayBuffer(
        max_size=max_size,
        obs_dim=5,
        act_dim=2,
        include_occupancy_grid=include_occupancy_grid,
        include_expert=include_expert
    )
    
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npz"))
    print(f"Found {len(files)} transition files.")
    
    for f in files:
        file_path = os.path.join(folder_path, f)
        data = load_transitions_dict(file_path)
        buffer.add_batch(data)

    print(f"Replay buffer populated with {buffer.size} transitions.")
    return buffer


def create_transition_dict(
        obs_vehicle_state: jnp.ndarray,
        action: jnp.ndarray,
        next_obs_vehicle_state: jnp.ndarray,
        reward: float,
        done: bool,
        truncated: bool = False,
        obs_occupancy_grid: Optional[jnp.ndarray] = None,
        next_obs_occupancy_grid: Optional[jnp.ndarray] = None,
        expert_action: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    """Create a transition dictionary from individual components.
    
    Args:
        obs_vehicle_state: Vehicle state array [5]
        action: Action array [2]
        next_obs_vehicle_state: Next vehicle state array [5]
        reward: Reward value
        done: Done flag
        truncated: Truncated flag
        obs_occupancy_grid: Occupancy grid array [256, 256] (optional)
        next_obs_occupancy_grid: Next occupancy grid array [256, 256] (optional)
        expert_action: Expert action array [2] (optional)
        
    Returns:
        Dictionary with transition data
    """
    transition = {
        'obs_vehicle_state': obs_vehicle_state,
        'action': action,
        'next_obs_vehicle_state': next_obs_vehicle_state,
        'reward': jnp.array(reward),
        'done': jnp.array(done),
        'truncated': jnp.array(truncated)
    }
    
    if obs_occupancy_grid is not None:
        transition['obs_occupancy_grid'] = obs_occupancy_grid
        transition['next_obs_occupancy_grid'] = next_obs_occupancy_grid
    
    if expert_action is not None:
        transition['expert_action'] = expert_action
    
    return transition


def batch_transitions_dict(transitions: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
    """Batch a list of transition dictionaries into a single dictionary.
    
    Args:
        transitions: List of transition dictionaries.
        
    Returns:
        Batched transition dictionary.
    """
    if not transitions:
        raise ValueError("Cannot batch empty list of transitions")
    
    # Get all unique keys
    all_keys = set()
    for t in transitions:
        all_keys.update(t.keys())
    
    # Batch each key
    batched = {}
    for key in all_keys:
        values = []
        for t in transitions:
            if key in t:
                values.append(t[key])
            else:
                # Handle missing keys by creating zero arrays
                if key in ['obs_occupancy_grid', 'next_obs_occupancy_grid']:
                    values.append(jnp.zeros((256, 256)))
                elif key == 'expert_action':
                    values.append(jnp.zeros(2))
                else:
                    raise ValueError(f"Missing required key: {key}")
        
        batched[key] = jnp.stack(values)
    
    return batched


def save_checkpoint(step: int,
                    actor_state: train_state.TrainState,
                    critic1_state: train_state.TrainState,
                    critic2_state: train_state.TrainState,
                    log_alpha: jnp.ndarray,
                    alpha_opt_state: optax.OptState,
                    ckpt_dir: str) -> None:
    """Save SAC training checkpoint.
    
    Args:
        step: Current training step.
        actor_state: Actor network training state.
        critic1_state: First critic network training state.
        critic2_state: Second critic network training state.
        log_alpha: Log of entropy coefficient.
        alpha_opt_state: Alpha optimizer state.
        ckpt_dir: Directory to save checkpoint.
    """
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
        overwrite=True
    )


def load_checkpoint(ckpt_dir: str) -> Any:
    """Load checkpoint if available, else return None.
    
    Args:
        ckpt_dir: Directory containing checkpoint.
        
    Returns:
        Checkpoint data or None if not found.
    """
    return checkpoints.restore_checkpoint(ckpt_dir, target=None)