from typing import Dict, Optional
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils.unit import rad2deg


def plot_transition_distribution(data: Dict[str, jnp.ndarray], path: Optional[str] = None):
    """Visualize and optionally save distribution plots of transitions.

    Args:
        data: Dictionary containing 'obs', 'actions', 'rewards', 'dones'.
        path: If provided, saves the figure to this file path (e.g., 'figs/distribution.png').
              If None, the figure is shown interactively.
    """
    obs = data['obs']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']

    x = obs[:, 0]
    y = obs[:, 1]
    yaw = obs[:, 2]
    v = obs[:, 3]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) x vs y
    axes[0, 0].scatter(x, y, alpha=0.5)
    axes[0, 0].set_title('Position Distribution')
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')

    # (2) yaw histogram
    axes[0, 1].hist(rad2deg(yaw), bins=50)
    axes[0, 1].set_title('Yaw Distribution')
    axes[0, 1].set_xlabel('Yaw [deg]')
    axes[0, 1].set_ylabel('Frequency')

    # (3) velocity histogram
    axes[0, 2].hist(v, bins=50)
    axes[0, 2].set_title('Velocity Distribution')
    axes[0, 2].set_xlabel('Veloccity [m/s]')
    axes[0, 2].set_ylabel('Frequency')

    # (4) action scatter
    axes[1, 0].scatter(actions[:, 0], actions[:, 1], alpha=0.5)
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].set_xlabel('Delta [-]')
    axes[1, 0].set_ylabel('Accel [-]')

    # (5) reward histogram
    axes[1, 1].hist(rewards, bins=50)
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')

    # (6) done flag count
    counts = [jnp.sum(~dones), jnp.sum(dones)]
    total = counts[0] + counts[1]
    bars = axes[1, 2].bar(["Not Done", "Done"], counts)
    axes[1, 2].set_title("Done Distribution")

    for bar, count in zip(bars, counts):
        percent = float(count) / float(total) * 100
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{int(count)} ({percent:.1f}%)",
                      ha="center", va="bottom")

    plt.tight_layout()

    if path:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()

