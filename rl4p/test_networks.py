"""Test script for new SAC networks."""

import jax
import jax.numpy as jnp
import numpy as np
from models.networks import PolicyNetwork, QNetwork, sample_action, policy_inference


def test_networks():
    """Test the new SAC networks with dictionary observations."""
    print("Testing SAC Networks...")
    
    # Create networks
    policy_net = PolicyNetwork()
    q_net = QNetwork()
    
    # Create dummy observation
    batch_size = 4
    obs = {
        'vehicle_state': jnp.array([
            [1.0, 2.0, 0.5, 3.0, 1],
            [2.0, 3.0, 0.6, 3.1, 1],
            [3.0, 4.0, 0.7, 3.2, 1],
            [4.0, 5.0, 0.8, 3.3, 1]
        ]),
        'occupancy_grid': jnp.zeros((batch_size, 256, 256))
    }
    
    # Create dummy action
    action = jnp.array([
        [0.1, 0.2],
        [0.2, 0.3],
        [0.3, 0.4],
        [0.4, 0.5]
    ])
    
    # Initialize networks
    rng = jax.random.PRNGKey(0)
    policy_params = policy_net.init(rng, obs)
    q_params = q_net.init(rng, obs, action)
    
    print(f"Policy network initialized with {len(policy_params['params'])} layers")
    print(f"Q network initialized with {len(q_params['params'])} layers")
    
    # Test policy network (training mode)
    (mean, log_std), _ = policy_net.apply(policy_params, obs, training=True, mutable=['batch_stats'])
    print(f"Policy output - mean shape: {mean.shape}, log_std shape: {log_std.shape}")
    print(f"Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"Log_std range: [{log_std.min():.3f}, {log_std.max():.3f}]")
    
    # Test Q network (training mode)
    (q_value, _) = q_net.apply(q_params, obs, action, training=True, mutable=['batch_stats'])
    print(f"Q value shape: {q_value.shape}")
    print(f"Q value range: [{q_value.min():.3f}, {q_value.max():.3f}]")
    
    # Test action sampling
    rng, sample_rng = jax.random.split(rng)
    sampled_action = sample_action(policy_net, policy_params, obs, sample_rng, training=True)
    print(f"Sampled action shape: {sampled_action.shape}")
    print(f"Sampled action range: [{sampled_action.min():.3f}, {sampled_action.max():.3f}]")
    
    # Test deterministic action
    det_action = sample_action(policy_net, policy_params, obs, training=False)
    print(f"Deterministic action shape: {det_action.shape}")
    print(f"Deterministic action range: [{det_action.min():.3f}, {det_action.max():.3f}]")
    
    return policy_net, q_net, policy_params, q_params


def test_legacy_compatibility():
    """Test legacy compatibility function."""
    print("\nTesting Legacy Compatibility...")
    
    # Create network
    policy_net = PolicyNetwork()
    
    # Create legacy observation (vehicle state only)
    batch_size = 2
    legacy_obs = jnp.array([
        [1.0, 2.0, 0.5, 3.0, 1],
        [2.0, 3.0, 0.6, 3.1, 1]
    ])
    
    # Initialize network with dictionary observation
    rng = jax.random.PRNGKey(0)
    dict_obs = {
        'vehicle_state': legacy_obs,
        'occupancy_grid': jnp.zeros((batch_size, 256, 256))
    }
    policy_params = policy_net.init(rng, dict_obs)
    
    # Test legacy function
    legacy_action = policy_inference(policy_net, policy_params, legacy_obs, training=False)
    print(f"Legacy action shape: {legacy_action.shape}")
    print(f"Legacy action range: [{legacy_action.min():.3f}, {legacy_action.max():.3f}]")
    
    # Compare with dictionary version
    dict_action = sample_action(policy_net, policy_params, dict_obs, training=False)
    print(f"Dictionary action shape: {dict_action.shape}")
    
    # Check if they're the same
    is_same = jnp.allclose(legacy_action, dict_action)
    print(f"Legacy and dictionary actions are same: {is_same}")
    
    return policy_net, policy_params


def test_network_shapes():
    """Test network shapes with different batch sizes."""
    print("\nTesting Network Shapes...")
    
    policy_net = PolicyNetwork()
    q_net = QNetwork()
    
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create observations
        obs = {
            'vehicle_state': jnp.zeros((batch_size, 5)),
            'occupancy_grid': jnp.zeros((batch_size, 256, 256))
        }
        action = jnp.zeros((batch_size, 2))
        
        # Initialize networks
        rng = jax.random.PRNGKey(0)
        policy_params = policy_net.init(rng, obs)
        q_params = q_net.init(rng, obs, action)
        
        # Test forward pass (training mode)
        (mean, log_std), _ = policy_net.apply(policy_params, obs, training=True, mutable=['batch_stats'])
        (q_value, _) = q_net.apply(q_params, obs, action, training=True, mutable=['batch_stats'])
        
        print(f"  Policy mean: {mean.shape}")
        print(f"  Policy log_std: {log_std.shape}")
        print(f"  Q value: {q_value.shape}")
        
        # Test action sampling
        rng, sample_rng = jax.random.split(rng)
        sampled_action = sample_action(policy_net, policy_params, obs, sample_rng, training=True)
        print(f"  Sampled action: {sampled_action.shape}")


def test_trainer_compatibility():
    """Test compatibility with trainer functions."""
    print("\nTesting Trainer Compatibility...")
    
    from models.trainer import create_sac_networks, initialize_sac_states
    
    # Create networks using trainer function
    policy_net, q_net = create_sac_networks(obs_dim=5, act_dim=2)
    
    # Initialize states
    rng = jax.random.PRNGKey(0)
    actor_state, critic1_state, critic2_state = initialize_sac_states(rng, policy_net, q_net)
    
    print(f"Actor state created: {actor_state is not None}")
    print(f"Critic1 state created: {critic1_state is not None}")
    print(f"Critic2 state created: {critic2_state is not None}")
    
    # Test forward pass with states
    batch_size = 2
    obs = {
        'vehicle_state': jnp.zeros((batch_size, 5)),
        'occupancy_grid': jnp.zeros((batch_size, 256, 256))
    }
    action = jnp.zeros((batch_size, 2))
    
    # Test actor
    (mean, log_std), _ = actor_state.apply_fn(actor_state.params, obs, training=True, mutable=['batch_stats'])
    print(f"Actor output shapes - mean: {mean.shape}, log_std: {log_std.shape}")
    
    # Test critic
    (q_value, _) = critic1_state.apply_fn(critic1_state.params, obs, action, training=True, mutable=['batch_stats'])
    print(f"Critic output shape: {q_value.shape}")
    
    return actor_state, critic1_state, critic2_state


if __name__ == '__main__':
    print("Testing new SAC networks...")
    
    # Test basic functionality
    policy_net, q_net, policy_params, q_params = test_networks()
    
    # Test legacy compatibility
    test_legacy_compatibility()
    
    # Test different batch sizes
    test_network_shapes()
    
    # Test trainer compatibility
    test_trainer_compatibility()
    
    print("\nAll tests completed!") 