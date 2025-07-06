from typing import List, Dict, Any, Optional
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state, checkpoints
from flax import serialization


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
                    actor_state: train_state.TrainState,
                    critic1_state: train_state.TrainState,
                    critic2_state: train_state.TrainState,
                    ckpt_dir: str) -> None:
    """Save SAC training checkpoint.
    
    Args:
        step: Current training step.
        actor_state: Actor network training state.
        critic1_state: First critic network training state.
        critic2_state: Second critic network training state.
        ckpt_dir: Directory to save checkpoint.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(ckpt_dir + f"/checkpoint_{step}.pkl", "wb") as f:
        f.write(serialization.to_bytes(actor_state))
        f.write(serialization.to_bytes(critic1_state))
        f.write(serialization.to_bytes(critic2_state))
        
    # checkpoints.save_checkpoint(
    #     ckpt_dir=ckpt_dir,
    #     target={
    #         'step': step,
    #         'actor_state': actor_state,
    #         'critic1_state': critic1_state,
    #         'critic2_state': critic2_state,
    #     },
    #     step=step,
    #     overwrite=True,
    # )

def load_checkpoint(ckpt_dir: str) -> Any:
    """Load checkpoint if available, else return None.
    
    Args:
        ckpt_dir: Directory containing checkpoint.
        
    Returns:
        Checkpoint data or None if not found.
    """
    return checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)