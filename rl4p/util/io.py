from typing import List, Dict, Any, Optional
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state, checkpoints
from flax import serialization
import pickle


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

def save_checkpoint(step: int,
                    encoder_state: train_state.TrainState,
                    actor_state: train_state.TrainState,
                    critic_state: train_state.TrainState,
                    ckpt_dir: str) -> None:
    """Save SAC training checkpoint.
    
    Args:
        step: Current training step.
        encoder_state: Encoder network training state.
        actor_state: Actor network training state.
        critic_state: Critic network training state.
        ckpt_dir: Directory to save checkpoint.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pkl")

    with open(ckpt_path, "wb") as f:
        # Serialize each state and write with size information
        encoder_bytes = serialization.to_bytes(encoder_state)
        actor_bytes = serialization.to_bytes(actor_state)
        critic_bytes = serialization.to_bytes(critic_state)
        
        # Write size followed by data for each state
        f.write(len(encoder_bytes).to_bytes(8, 'big'))
        f.write(encoder_bytes)
        f.write(len(actor_bytes).to_bytes(8, 'big'))
        f.write(actor_bytes)
        f.write(len(critic_bytes).to_bytes(8, 'big'))
        f.write(critic_bytes)

def load_checkpoint(ckpt_path: str) -> Optional[Dict[str, train_state.TrainState]]:
    """Load checkpoint from pkl file if available, else return None.
    
    Args:
        ckpt_path: Path to checkpoint.
        
    Returns:
        Dictionary containing encoder, actor, critic TrainStates or None if not found.
    """
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return None
    
    try:
        with open(ckpt_path, "rb") as f:
            # Read size and data for each state
            encoder_size = int.from_bytes(f.read(8), 'big')
            encoder_bytes = f.read(encoder_size)
            encoder_state = serialization.from_bytes(train_state.TrainState, encoder_bytes)
            
            actor_size = int.from_bytes(f.read(8), 'big')
            actor_bytes = f.read(actor_size)
            actor_state = serialization.from_bytes(train_state.TrainState, actor_bytes)
            
            critic_size = int.from_bytes(f.read(8), 'big')
            critic_bytes = f.read(critic_size)
            critic_state = serialization.from_bytes(train_state.TrainState, critic_bytes)
            
            return {
                'encoder': encoder_state['params'],
                'actor': actor_state['params'],
                'critic': critic_state['params'],
            }
            
    except Exception as e:
        print(f"Error loading checkpoint from {ckpt_path}: {e}")
        return None