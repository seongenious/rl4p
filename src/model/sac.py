from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from model.agent import Agent
from model.replay_buffer import ReplayBuffer


class SAC(nn.Module):
    def __init__(self):
        super(SAC, self).__init__()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Actor network
        # self.actor = Transformer()
        self.log_std = nn.Parameter(-torch.zeros(1, 2)).to(self.device)
        self.log_std.requires_grad = True
        self.actor_optimizer = torch.optim.Adam(
          [{'params': self.actor.parameters()}, {'params': self.log_std}],
          lr=0.001,
        )

        # Critic network
        self.critic_1 = None
        self.critic_target_1 = deepcopy(self.critic_1)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=0.001)

        self.critic_2 = None
        self.critic_target_2 = deepcopy(self.critic_2)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=0.001)

        # Alpha
        self.log_alpha = torch.tensor(np.log(0.01)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.001)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size=10240)
    
    