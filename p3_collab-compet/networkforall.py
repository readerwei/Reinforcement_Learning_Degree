import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actions_size=0, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        self.nonlin = f.relu #leaky_relu
        self.actor = actor
        self.action_size = actions_size

        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        if self.actor:
            self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        else:
            self.fc2 = nn.Linear(hidden_in_dim + actions_size, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc1.weight.data.uniform_(-3e-2, 3e-2)
        # self.fc2.weight.data.uniform_(-3e-2, 3e-2)
        self.fc3.weight.data.uniform_(-3e-1, 3e-1)

    def forward(self, x):
        if self.actor:
            # return a vector of the force
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = self.fc3(h2)
            action = torch.tanh(h3)
            
            # we bound the norm of the vector to be between -1 and 1
            return action
        
        else:
            # critic first layer takes only the state space inputs and 2nd later will incorprate action space results
            actions=x[:,  -self.action_size:]
            state = x[:, :-self.action_size ]
            h1 = self.nonlin(self.fc1(state))
            h1_cat = torch.cat([h1, actions], dim=1)
            h2 = self.nonlin(self.fc2(h1_cat))
            h3 = self.fc3(h2)
            # critic network simply outputs a number
            return h3