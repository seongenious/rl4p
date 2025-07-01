"""Test script for new replay buffer and io functions."""

import jax
import jax.numpy as jnp
import numpy as np
from utils.replay_buffer import ReplayBuffer
from utils.io import save_transitions_dict, load_transitions_dict, create_transition_dict, batch_transitions_dict


def test_replay_buffer():
    """Test the new replay buffer functionality."""
    print("Testing ReplayBuffer...")
    
    # Create buffer with occupancy grid and expert actions
    buffer = ReplayBuffer(
        max_size=1000,
        obs_dim=5,
        act_dim=2,
        include_occupancy_grid=True,
        include_expert=True
    )
    
    # Add some transitions
    for i in range(10):
        obs_vehicle = jnp.array([i, i*0.5, 0.1*i, 2.0, 1])
        action = jnp.array([0.1, 0.2])
        next_obs_vehicle = jnp.array([i+1, (i+1)*0.5, 0.1*(i+1), 2.1, 1])
        obs_grid = jnp.zeros((256, 256))
        next_obs_grid = jnp.zeros((256, 256))
        expert_action = jnp.array([0.15, 0.25])
        
        buffer.add(
            obs_vehicle_state=obs_vehicle,
            action=action,
            next_obs_vehicle_state=next_obs_vehicle,
            reward=1.0,
            done=False,
            truncated=False,
            obs_occupancy_grid=obs_grid,
            next_obs_occupancy_grid=next_obs_grid,
            expert_action=expert_action
        )
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test sampling
    rng = jax.random.PRNGKey(0)
    batch = buffer.sample(5, rng)
    print(f"Sampled batch keys: {list(batch.keys())}")
    print(f"Vehicle state shape: {batch['obs_vehicle_state'].shape}")
    
    # Test vehicle-only sampling
    vehicle_batch = buffer.sample_vehicle_only(3, rng)
    print(f"Vehicle-only batch keys: {list(vehicle_batch.keys())}")
    
    return buffer


def test_io_functions():
    """Test the new io functions."""
    print("\nTesting IO functions...")
    
    # Create some transition data
    transitions = []
    for i in range(5):
        transition = create_transition_dict(
            obs_vehicle_state=jnp.array([i, i*0.5, 0.1*i, 2.0, 1]),
            action=jnp.array([0.1, 0.2]),
            next_obs_vehicle_state=jnp.array([i+1, (i+1)*0.5, 0.1*(i+1), 2.1, 1]),
            reward=1.0,
            done=False,
            truncated=False,
            obs_occupancy_grid=jnp.zeros((256, 256)),
            next_obs_occupancy_grid=jnp.zeros((256, 256)),
            expert_action=jnp.array([0.15, 0.25])
        )
        transitions.append(transition)
    
    # Test batching
    batched = batch_transitions_dict(transitions)
    print(f"Batched keys: {list(batched.keys())}")
    print(f"Batched vehicle state shape: {batched['obs_vehicle_state'].shape}")
    
    # Test save/load
    save_path = "test_transitions.npz"
    save_transitions_dict(batched, save_path)
    
    loaded = load_transitions_dict(save_path)
    print(f"Loaded keys: {list(loaded.keys())}")
    print(f"Loaded vehicle state shape: {loaded['obs_vehicle_state'].shape}")
    
    # Clean up
    import os
    os.remove(save_path)
    
    return batched


def test_buffer_io():
    """Test buffer save/load functionality."""
    print("\nTesting buffer IO...")
    
    # Create and populate buffer
    buffer = ReplayBuffer(
        max_size=100,
        include_occupancy_grid=True,
        include_expert=True
    )
    
    # Add data
    data = {
        'obs_vehicle_state': jnp.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]),
        'action': jnp.array([[0.1, 0.2], [0.2, 0.3]]),
        'next_obs_vehicle_state': jnp.array([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]),
        'reward': jnp.array([1.0, 2.0]),
        'done': jnp.array([False, True]),
        'truncated': jnp.array([False, False]),
        'obs_occupancy_grid': jnp.zeros((2, 256, 256)),
        'next_obs_occupancy_grid': jnp.zeros((2, 256, 256)),
        'expert_action': jnp.array([[0.15, 0.25], [0.25, 0.35]])
    }
    
    buffer.add_batch(data)
    print(f"Buffer size after adding batch: {len(buffer)}")
    
    # Save buffer
    save_path = "test_buffer.npz"
    from utils.io import save_replay_buffer, load_replay_buffer
    
    save_replay_buffer(buffer, save_path)
    
    # Load buffer
    loaded_buffer = load_replay_buffer(
        save_path, 
        max_size=100, 
        include_occupancy_grid=True, 
        include_expert=True
    )
    
    print(f"Loaded buffer size: {len(loaded_buffer)}")
    
    # Compare data
    original_data = buffer.get_all_data()
    loaded_data = loaded_buffer.get_all_data()
    
    for key in original_data.keys():
        if key in loaded_data:
            is_equal = jnp.allclose(original_data[key], loaded_data[key])
            print(f"{key}: {'✓' if is_equal else '✗'}")
    
    # Clean up
    import os
    os.remove(save_path)
    
    return buffer


if __name__ == '__main__':
    print("Testing new replay buffer and io functions...")
    
    # Test replay buffer
    buffer = test_replay_buffer()
    
    # Test io functions
    batched = test_io_functions()
    
    # Test buffer io
    test_buffer_io()
    
    print("\nAll tests completed!") 