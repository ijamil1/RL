'''
On-policy Monte Carlo RL Algorithm (without exploring starts) to find approximately optimal policy  
'''
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
#env.action_space => 0, 1, 2; discrete 
#print(env.observation_space) => sz 2 array corresponding to pos, vel

obs_space_low = env.observation_space.low
obs_space_high =  env.observation_space.high

pos_min = obs_space_low[0] #-1.2
pos_max = obs_space_high[0] #0.6
vel_min = obs_space_low[1] #-0.07
vel_max = obs_space_high[1] #0.07

epsilon = 0.1
gamma = 0.999
#need to discretize the state space; keep it simple => 10 intervals for pos x  10 intervals  for vel = 100 possible states
 
pos_interval_size = (pos_max-pos_min)/10
vel_interval_size = (vel_max-vel_min)/10

pos_interval_inclusive_lb = np.arange(pos_min, pos_max, pos_interval_size)
vel_interval_inclusive_lb = np.arange(vel_min, vel_max, vel_interval_size)

policy_dict = {} #maps each state to an initial pmf over action space; key is (pos int idx, vel int idx)
return_dict = {}  #maps each state x action pair to returns for that (s,a) pair; key is (pos int idx, vel int idx, action idx)
state_action_value_dict = {} #maps each state x action pair to estimate of q(s,a) for current policy
for i in range(10):
    for j in range(10):
        policy_dict[(i,j)] = [1/3,1/3,1/3]
        for k in range(3):
            return_dict[(i,j,k)] = []
            state_action_value_dict[(i,j,k)] = 0


def return_position_interval(pos):
    #given pos, determine the interval it is in and return the index of the interval it is in if the intervals were ordered in ascending order, 0-indexed
    for i in range(pos_interval_inclusive_lb.shape[0]):
        if i == pos_interval_inclusive_lb.shape[0]-1  or pos < pos_interval_inclusive_lb[i+1]:
            return i

def return_vel_interval(vel):
    #given vel, determine the interval it is in and return the index of the interval it is in if the intervals were ordered in ascending order, 0-indexed
    for i in range(vel_interval_inclusive_lb.shape[0]):
        if i == vel_interval_inclusive_lb.shape[0]-1  or vel < vel_interval_inclusive_lb[i+1]:
            return i

def calculate_first_visit_rewards(state_action_seq, reward_seq):
    cum_disc_rewards = 0
    for i in range(len(reward_seq)):
        cum_disc_rewards = reward_seq[len(reward_seq)-1-i] + gamma * cum_disc_rewards
        cur_state_action = state_action_seq[len(reward_seq)-1-i]
        if cur_state_action not in state_action_seq[:len(reward_seq)-1-i]:
            #first visit
            return_dict[cur_state_action].append(reward_seq[len(reward_seq)-1-i])
            state_action_value_dict[cur_state_action] = sum(return_dict[cur_state_action])/len(return_dict[cur_state_action])
            optimal_action = np.argmax([state_action_value_dict[(cur_state_action[0],cur_state_action[1],0)],state_action_value_dict[(cur_state_action[0],cur_state_action[1],1)],state_action_value_dict[(cur_state_action[0],cur_state_action[1],2)]])
            policy_dict[(cur_state_action[0], cur_state_action[1])] = [epsilon/3]*3
            policy_dict[(cur_state_action[0], cur_state_action[1])][optimal_action]+=(1-epsilon)




observation, info = env.reset(seed=42)
episode_start = True
episode_rewards = []
episode_state_action_seq = []
num_timesteps_list = []
timesteps = 0
for step_count in range(1000000000): #1 billion steps
    if step_count % 100000000 == 0:
        print(step_count/1000000000) 
    if episode_start:
        episode_start = False
        episode_rewards.clear()
        episode_state_action_seq.clear()
        timesteps = 0

    pos_idx = return_position_interval(observation[0])
    vel_idx = return_vel_interval(observation[1])
    state_pmf = policy_dict[(pos_idx,vel_idx)]
    un_rv = np.random.uniform()
    action = 0
    for i in range(len(state_pmf)):
        if un_rv <= sum(state_pmf[:i+1]):
            action = i
            break
    
    episode_state_action_seq.append((pos_idx,vel_idx, action))

    observation, reward, terminated, truncated, info = env.step(action)
    timesteps+=1
    episode_rewards.append(reward)

    if terminated or truncated:
        episode_start = True
        num_timesteps_list.append(timesteps)
        calculate_first_visit_rewards(episode_state_action_seq, episode_rewards)
        observation, info = env.reset()
env.close()


def plot_lines(x_values, y1_values):
    # Plotting the first line
    plt.plot(x_values, y1_values, label='Num Timesteps per Episode', marker='o', linestyle='-', color='blue')

    # Adding labels and title
    plt.xlabel('Episode Number')
    plt.ylabel('Timesteps per Episode')
    plt.title('On-policy Monte Carlo RL Agent Learning')

    plt.legend()
    plt.savefig('timesteps_per_episode.png')
    plt.show()

# Example usage:
# Replace these lists with your own data
x_values = np.arange(1,len(num_timesteps_list)+1)
y1_values = num_timesteps_list
plot_lines(x_values, y1_values)


print(policy_dict)
    


