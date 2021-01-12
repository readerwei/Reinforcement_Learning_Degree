# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise
NOISE_SIGMA = 0.2      # sigma for Ornstein-Uhlenbeck noise
NOISE_THETA = 0.15
WEIGHT_DECAY = 0 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0, theta=NOISE_THETA, sigma=NOISE_SIGMA)

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=lr_actor,  weight_decay = WEIGHT_DECAY)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay = WEIGHT_DECAY)


    def act(self, obs, noise=0.0, no_grad=False):
        obs = obs.to(device)

        if no_grad:
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(obs).detach()
            self.actor.train()
        else:
            action = self.actor(obs)
            
        ounoise = self.noise.noise().to(device)
        action += noise*ounoise
        return torch.clamp(action, -1, 1)

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        # ounoise = self.noise.noise().to(device)
        # action = self.target_actor(obs) + noise*ounoise
        action = self.target_actor(obs)
        return torch.clamp(action, -1, 1)
