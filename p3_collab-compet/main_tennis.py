from unityagents import UnityEnvironment
import numpy as np

from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor

def seeding(seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    

env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")
# env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents 
num_agents = len(env_info.agents)
# size of each action
action_size = brain.vector_action_space_size
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]


seeding()
# number of parallel agents
parallel_envs = 1
# number of training episodes.
# change this to higher number to experiment. say 30000.
number_of_episodes = 3000
batchsize = 1000
# how many episodes to save policy and gif
save_interval = 1000
t = 0

# amplitude of OU noise, this slowly decreases to 0
noise = 2
noise_reduction = 0.9999

# how many episodes before update
episode_per_update = 2 * parallel_envs
log_path = os.getcwd()+"/log"
model_dir= os.getcwd()+"/model_dir"
os.makedirs(model_dir, exist_ok=True)
# torch.set_num_threads(parallel_envs)

# keep 5000 episodes worth of replay
buffer = ReplayBuffer(int(1e2))

# initialize policy and critic
maddpg = MADDPG()
logger = SummaryWriter(log_dir=log_path)
agent0_reward = []
agent1_reward = []

while True:


    # explore = only explore for a certain number of episodes
    # action input needs to be transposed
    actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
    noise *= noise_reduction
    
    actions_array = torch.stack(actions).detach().numpy()

    # transpose the list of list
    # flip the first two indices
    # input to step requires the first index to correspond to number of parallel agents
    actions_for_env = np.rollaxis(actions_array,1)
    
    # step forward
    env_info = env.step(actions_for_env[0])[brain_name]
    next_obs_all = env_info.vector_observations         # get next state (for each agent)
    next_obs = [list(next_obs_all)]
    next_obs_full= [np.concatenate(next_obs_all)]
    rewards  = env_info.rewards                         # get reward (for each agent)
    dones    = env_info.local_done                        # see if episode finished
    
           
    # add data to buffer
    transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)
    
    buffer.push(transition)
    
    reward_this_episode += rewards

    obs, obs_full = next_obs, next_obs_full
    if np.any(dones):
        break
        
samples = buffer.sample(10)
maddpg.update(samples, 1, logger)



