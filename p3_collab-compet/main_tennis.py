import os
import shutil
from datetime import datetime
from collections import deque
import numpy as np
import progressbar as pb

import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment

from buffer import ReplayBuffer
from maddpg import MADDPG
from utilities import transpose_list, transpose_to_tensor


def seeding(seed=6666):
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
number_of_episodes = 10000
episodes_before_training = 100
learn_per_episode = 1
batchsize = 512
# how many episodes to save policy
save_interval = 500
t = 0

# amplitude of OU noise, this slowly decreases to 0
noise = 1.0
EPSILON_DECAY = 0.998
noise_low_threshold = 0.05

# how many episodes before update
episode_per_update = 1 * parallel_envs
log_path = os.getcwd()+"/log/" + str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
# log_path = os.getcwd() + "/log/Test/"
# shutil.rmtree(log_path, ignore_errors=True)
model_dir= os.getcwd()+"/model_dir/" + str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
os.makedirs(model_dir, exist_ok=True)
torch.set_num_threads(4)

# keep 5000 episodes worth of replay; this suppose each episode have 10 steps
buffersize = int(5e4)
buffer = ReplayBuffer(buffersize)

# initialize policy and critic
GAMMA = 0.99 
TAU   = 1e-2
maddpg = MADDPG(discount_factor=GAMMA, tau=TAU)
logger = SummaryWriter(log_dir=log_path)

hyper_dict   = {"number_of_episodes": number_of_episodes, 
                "episodes_before_training": episodes_before_training,
                "learn_per_episode": learn_per_episode, 
                "batchsize": batchsize, 
                "noise_start_scale" : noise,
                "noise_reduction": EPSILON_DECAY,
                "noise_low_threshold": noise_low_threshold,
                "discount_factor" : GAMMA, 
                "tau" : TAU
               }
torch.save(hyper_dict, os.path.join(model_dir, 'hyperparam.pt'))


agent0_reward = []
agent1_reward = []

# print_every=100
scores_deque  = deque(maxlen=100)
score_average = 0
agent_scores_last_100 = [deque(maxlen = 100),deque(maxlen = 100)]
agent_scores_avg      = np.zeros(num_agents)

obs_iter = 0 

# all the metrics progressbar will keep track of
widget = ['episode: ', pb.Counter(),'/',str(number_of_episodes),' ',
            pb.DynamicMessage('a0_score'), ' ',
            pb.DynamicMessage('a1_score'), ' ',
            pb.DynamicMessage('a0_avg_score'), ' ',
            pb.DynamicMessage('a1_avg_score'), ' ',
            pb.DynamicMessage('final_score'), ' ',
            pb.DynamicMessage('noise_scale'), ' ',
            pb.DynamicMessage('buffer_size'), ' ',
            pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ] 

timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start() # progressbar


for episode in range(0, number_of_episodes):
   
    reward_this_episode = np.zeros((parallel_envs, num_agents))
    env_info = env.reset(train_mode=True)[brain_name] 
    all_obs = env_info.vector_observations

    obs = [list(all_obs)]
    obs_full= [np.concatenate(all_obs)]
    train_flag = (episode>=episodes_before_training)
    
    while True:
        # explore = only explore for a certain number of episodes
        # action input needs to be transposed
        if train_flag:
            actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            actions_array = torch.stack(actions).detach().cpu().numpy()
        else:
            actions_array = np.random.uniform(-1, 1, 4).reshape(num_agents, parallel_envs, action_size)
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
        logger.add_scalars('agent1/actions',  {'action[0]': actions_for_env[0][1][0], 
                                               'action[1]': actions_for_env[0][1][1]},  obs_iter) 
        if np.any(dones):
            break

    # if len(buffer) > batchsize and episode % episode_per_update < parallel_envs:
    if len(buffer) > batchsize and train_flag:
        for _ in range(learn_per_episode):
            for a_i in range(num_agents):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            maddpg.update_targets() #soft update the target network towards the actual networks

    # for i in range(parallel_envs):
    agent0_reward.append(reward_this_episode[0,0])
    agent1_reward.append(reward_this_episode[0,1])
    # if episode % parallel_envs == 0 or episode == number_of_episodes-1:
    avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
    max_rewards = [np.max(agent0_reward),  np.max(agent1_reward) ]

    agent_scores_last_100[0].append(reward_this_episode[0,0])
    agent_scores_last_100[1].append(reward_this_episode[0,1])
    agent_scores_avg = [np.mean(agent_scores_last_100[0]), np.mean(agent_scores_last_100[1])]

    for a_i, avg_rew in enumerate(avg_rewards):
        logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)
    for a_i, max_rew in enumerate(max_rewards):
        logger.add_scalar('agent%i/max_episode_rewards' % a_i, max_rew, episode)
    for a_i, avg_rew_100 in enumerate(agent_scores_avg):
        logger.add_scalar('agent%i/mean_100_episode_rewards' % a_i, avg_rew_100, episode)
    
    scores_deque.append(reward_this_episode[0].max())
    score_average_tmp  = np.mean(scores_deque)

    noise_chg = False
    if score_average_tmp > score_average:
        noise_chg = True

    score_average = score_average_tmp
    logger.add_scalar('result/final_metric', score_average, episode)

    if train_flag and noise_chg and noise >= noise_low_threshold:
        noise *= EPSILON_DECAY
    logger.add_scalars('noise/scale', {'noise': noise}, episode)

    timer.update(episode, a0_score=agent0_reward[-1], a1_score=agent1_reward[-1],
                 a0_avg_score=agent_scores_avg[0], a1_avg_score=agent_scores_avg[1], 
                 final_score=score_average, noise_scale=noise, buffer_size=len(buffer)) # progressbar

    save_info = ((episode) % save_interval == 0 or episode==number_of_episodes-parallel_envs)
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
timer.finish()