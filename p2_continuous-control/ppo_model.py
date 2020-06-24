import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.std = nn.Parameter(torch.ones(action_size))
        
    def forward(self, x, action):
         """Build a policy network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = F.tanh(self.fc3(x))
        dist = torch.distributions.Normal(mean, 0.6*self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        return mean, log_prob