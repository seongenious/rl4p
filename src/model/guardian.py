import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Guardian(nn.Module):
    def __init__(self, bev_dim: int = 128, action_dim: int = 8, hidden_dim: int = 128, wheel_base: float = 2.8, dt: float = 0.5):
        super(Guardian, self).__init__()
        self.wheel_base = wheel_base
        self.dt = dt
        self.action_embed = nn.Sequential(
            nn.Linear(2, action_dim // 2),
            nn.GELU(),
            nn.Linear(action_dim // 2, action_dim)
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(bev_dim + action_dim),
            nn.Linear(bev_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _preprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        delta, v = action.split(1, dim=-1)
        kappa = torch.tan(delta) / self.wheel_base
        dist = v * self.dt
        feat = torch.cat([kappa, dist], dim=-1)
        return feat

    def forward(self, z_bev, action: torch.Tensor) -> torch.Tensor:
        if isinstance(z_bev, np.ndarray):
            z_bev = torch.from_numpy(z_bev).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        if len(z_bev.shape) == 3:
            z_bev = z_bev.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        action = self._preprocess_action(action)
        action_feat = self.action_embed(action)
        x = torch.cat([z_bev, action_feat], dim=-1)
        x = self.mlp(x)  # Keep shape [batch_size, 1]
        p_collide = torch.sigmoid(x)
        return p_collide