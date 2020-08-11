from unityagents import UnityEnvironment
import numpy as np
import torch
import os
from datetime import datetime

from buffer import ReplayBuffer
from maddpg import MADDPG
from tensorboardX import SummaryWriter
from utilities import transpose_list, transpose_to_tensor

def seeding(seed=2325):
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
parallel_envs = 4
# number of training episodes.
# change this to higher number to experiment. say 30000.
number_of_episodes = 3000
batchsize = 1024
# how many episodes to save policy and gif
save_interval = 100
t = 0

# amplitude of OU noise, this slowly decreases to 0
noise = 0.5
noise_reduction = 0.9999

# how many episodes before update
episode_per_update = 2 * parallel_envs
log_path = os.getcwd()+"/log/" + str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
model_dir= os.getcwd()+"/model_dir"
os.makedirs(model_dir, exist_ok=True)
torch.set_num_threads(parallel_envs)

# keep 5000 episodes worth of replay
buffer = ReplayBuffer(int(1e5))

# initialize policy and critic
maddpg = MADDPG(discount_factor=0.995, tau=1e-3)
logger = SummaryWriter(log_dir=log_path)
agent0_reward = []
agent1_reward = []

obs_iter = 0 

for episode in range(0, number_of_episodes):

    reward_this_episode = np.zeros((parallel_envs, num_agents))
    env_info = env.reset(train_mode=True)[brain_name] 
    all_obs = env_info.vector_observations

    obs = [list(all_obs)]
    obs_full= [np.concatenate(all_obs)]

    save_info = ((episode) % save_interval < parallel_envs or episode==number_of_episodes-parallel_envs)

    while True:
        # explore = only explore for a certain number of episodes
        # action input needs to be transposed
        actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
        noise *= noise_reduction

        actions_array = torch.stack(actions).detach().cpu().numpy()

        # transpose the list of list
        # flip the first two indices
        # input to step requires the first index to correspond to number of parallel agents
        actions_for_env = np.rollaxis(actions_array,1)

        # step forward
        env_info = env.step(actions_for_env[0])[brain_name]
        next_obs_all = env_info.vector_observations         # get next state (for each agent)
        next_obs = [list(next_obs_all)]
        next_obs_full= [np.concatenate(next_obs_all)]
        rewards  = [env_info.rewards]                         # get reward (for each agent)
        dones    = [env_info.local_done]                        # see if episode finished

        # add data to buffer
        transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)

        buffer.push(transition)

        reward_this_episode += rewards

        obs, obs_full = next_obs, next_obs_full

        obs_iter += 1
        logger.add_scalars('agent0/actions',  {'action[0]': actions_for_env[0][0][0], 
                                               'action[1]': actions_for_env[0][0][1]},  obs_iter) 
        if np.any(dones):
            break

    # if len(buffer) > batchsize and episode % episode_per_update < parallel_envs:
    if len(buffer) > batchsize:
        for a_i in range(num_agents):
            samples = buffer.sample(batchsize)
            maddpg.update(samples, a_i, logger)
        maddpg.update_targets() #soft update the target network towards the actual networks

    for i in range(parallel_envs):
        agent0_reward.append(reward_this_episode[i,0])
        agent1_reward.append(reward_this_episode[i,1])

    if episode % 10 == 0 or episode == number_of_episodes-1:
        avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
        max_rewards = [np.max(agent0_reward), np.max(agent1_reward)]
        agent0_reward = []
        agent1_reward = []
        for a_i, avg_rew in enumerate(avg_rewards):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)
        for a_i, max_rew in enumerate(max_rewards):
            logger.add_scalar('agent%i/max_episode_rewards' % a_i, max_rew, episode)

    save_dict_list =[]
    if save_info:
        for i in range(num_agents):
            save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                         'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                         'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                         'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
            save_dict_list.append(save_dict)
            torch.save(save_dict_list, os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

env.close()
logger.close()