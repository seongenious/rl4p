import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configs import *


class Guardian(nn.Module):
    def __init__(self, config: GuardianConfig) -> None:
        super(Guardian, self).__init__()

        self.device = device

        self.wheel_base = config.wheel_base
        self.dt = config.dt
        self.action_embed = nn.Sequential(
            nn.Linear(2, config.action_feat_dim // 2),
            nn.GELU(),
            nn.Linear(config.action_feat_dim // 2, config.action_feat_dim)
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.bev_feat_dim + config.action_feat_dim),
            nn.Linear(config.bev_feat_dim + config.action_feat_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
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

    def load(self, path: str, require_grad: bool = False) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict)
        for param in self.parameters():
            param.requires_grad = require_grad