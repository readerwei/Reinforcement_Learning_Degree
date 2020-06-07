import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clipped_surrogate(policy, old_log_probs, states, actions, rewards_normalized, epsilon=0.2, beta=0.01):
    
    # convert everything into pytorch tensors and move to gpu if available
    actions       = torch.stack(actions).float().to(device=device)
    states        = torch.from_numpy(np.stack(states)).float().to(device=device)
    old_log_probs = torch.stack(old_log_probs).float().to(device=device)
    rewards       = torch.from_numpy(np.stack(rewards_normalized)).float().to(device=device)

    # convert states to policy (or probability)
    new_actions, new_log_probs = policy(states, actions)
    # ratio for clipping
    new_probs = new_log_probs.exp()
    old_probs = old_log_probs.exp()
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan  
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
   
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)


def collect_trajectories(env, brain_name, policy, state, action, action_size, num_agents=1, tmax=50): 
    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]
    done = False
# collect trajectories
    for t in range(tmax):
        state = torch.from_numpy(state).float().to(device)
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        action, log_prob = policy(state, action)
        action = action.detach()
        log_prob = log_prob.detach()

        # we take one action and move forward
        actions = np.expand_dims(action.data.numpy(), axis=0)
        env_info    = env.step(actions)[brain_name]        # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones   = env_info.local_done
        next_state  = next_states[0]
        reward  = rewards[0]
        is_done = dones[0]
      
        # store the result
        state_list.append(next_state)
        reward_list.append(reward)
        prob_list.append(log_prob)
        action_list.append(action)
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        state = next_state
        if is_done:
            done = True
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list, done


def normalize_reward(rewards, discount):
    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
      # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10
    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    return rewards_normalized
