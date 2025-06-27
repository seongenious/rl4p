from flax.training import train_state
from flax.core import FrozenDict
from acme.agents.jax import sac
from acme import specs
from acme.jax import networks as networks_lib
import optax
import jax
import rlax
import jax.numpy as jnp

from models.sac_networks import PolicyNetwork, QNetwork


def make_sac_networks(env_spec: specs.EnvironmentSpec) -> sac.SACNetworks:
    """Initializes SACNetworks using Flax-defined policy and Q-networks.

    Args:
        env_spec (specs.EnvironmentSpec): Observation and action specs.

    Returns:
        sac.SACNetworks: Networks ready for SAC agent.
    """
    action_dim = env_spec.actions.shape[0]
    obs_dim = env_spec.observations.shape[0]

    rng = jax.random.PRNGKey(0)

    # Define dummy input
    dummy_obs = jnp.zeros(env_spec.observations.shape)
    dummy_action = jnp.zeros(env_spec.actions.shape)

    # Initialize networks
    policy_module = PolicyNetwork(action_dim)
    q1_module = QNetwork()
    q2_module = QNetwork()

    policy_params = policy_module.init(rng, dummy_obs)
    q1_params = q1_module.init(rng, dummy_obs, dummy_action)
    q2_params = q2_module.init(rng, dummy_obs, dummy_action)

    # Create Flax-compatible callable wrappers
    def apply_policy(params: FrozenDict, obs: jnp.ndarray):
        return policy_module.apply(params, obs)

    def apply_q1(params: FrozenDict, obs: jnp.ndarray, action: jnp.ndarray):
        return q1_module.apply(params, obs, action)

    def apply_q2(params: FrozenDict, obs: jnp.ndarray, action: jnp.ndarray):
        return q2_module.apply(params, obs, action)

    return sac.SACNetworks(
        policy_network=networks_lib.FeedForwardNetwork(policy_module.init, apply_policy),
        policy_network_params=policy_params,
        q_network=networks_lib.FeedForwardNetwork(q1_module.init, apply_q1),
        q_network_2=networks_lib.FeedForwardNetwork(q2_module.init, apply_q2),
        q_network_params=q1_params,
        q_network_2_params=q2_params,
        sample_network=lambda mean, log_std, key: rlax.gaussian_diagonal().sample(key, (mean, jnp.exp(log_std)))
    )
