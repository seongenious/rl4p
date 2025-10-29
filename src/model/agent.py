import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)