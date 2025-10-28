import torch
import torch.nn as nn
import torch.nn.functional as F


class Guardian(nn.Module):
    def __init__(self, bev_dim: int = 128, action_dim: int = 32, hidden_dim: int = 128, wheel_base: float = 2.8):
        super(Guardian, self).__init__()
        self.wheel_base = wheel_base
        self.action_embed = nn.Sequential(
            nn.Linear(2, action_dim // 2),
            nn.GELU(),
            nn.Linear(action_dim // 2, action_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(bev_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _preprocess_action(self, delta, v):
        kappa = torch.tan(delta) / self.wheel_base
        feat = torch.cat([kappa, v], dim=-1)
        return feat

    def forward(self, z_bev, delta, v):
        action = self._preprocess_action(delta, v)
        action_feat = self.action_embed(action)
        x = torch.cat([z_bev, action_feat], dim=-1)
        x = self.mlp(x).squeeze(-1)
        p_collide = torch.sigmoid(x)
        return p_collide