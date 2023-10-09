import json
import os
import gymnasium as gym
import torch

from examples.offline_RL_workshop.custom_envs.custom_envs_registration import CustomEnv, RenderMode, register_grid_envs
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName, PolicyRestorationConfigFactoryRegistry
from examples.offline_RL_workshop.utils import extract_dimension
from tianshou.data import Batch
from itertools import accumulate
'''
def compute_cumulative_list_inverse(input_list):
    cumulative_list = []
    cumulative_sum = 0.0

    for value in reversed(input_list):
        cumulative_sum += value
        cumulative_list.insert(0, cumulative_sum)  # Insert at the beginning of the list

    return cumulative_list

'''


def compute_cumulative_list_inverse(input_list):
    # Reverse the input list and compute the cumulative sum
    cumulative_list = list(accumulate(reversed(input_list)))

    return cumulative_list


def compute_mean_discounted_cumulative_reward(gym_env, policy, num_episodes=1, gamma=0.99):
    state_shape = extract_dimension(gym_env.observation_space)
    num_steps_traj = gym_env.max_num_steps

    mean_cumulative_rewards = [0.0]*num_steps_traj
    mean_values = [0.0]*num_steps_traj
    for _ in range(num_episodes):
        done = False
        truncated = False
        state, _ = gym_env.reset()
        rewards = [0.0]*num_steps_traj
        values = [0.0]*num_steps_traj
        t = 0
        while not (done or truncated):
            # For BCQ
            tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
            policy_output = policy(tensor_state)

            if isinstance(gym_env.action_space, gym.spaces.Discrete):
                action = int(policy_output.act[0])
            else:
                action = policy_output.act[0]

                #action_array = policy_output.act
                #max_index = torch.argmax(action_array)
                #one_hot_action_array = torch.zeros_like(action_array)
                #one_hot_action_array[0, max_index] = 1


                #action = float(torch.argmax(policy_output.act[0]))
                #action = float(np.argmax(policy_output.act[0]))

            #DQN?? --> compute q values
            #value = float(torch.max(policy_output.logits[0]))
            
            # For continuous CQL
            #value = float(torch.max(policy_output.logits[0]))
            # For discrete BCQ
            value = float(torch.max(policy_output.q_value[0]))

            # For continuous BCQ
            #value = float(policy.critic1(state.reshape(1, state_shape), policy_output.act)[0])

            #action = gym_env.action_space.sample()
            next_state, reward, done, truncated, info = gym_env.step(action)

            #print(reward, reward*(gamma**t))

            rewards[t]=reward*(gamma**t)
            #print(policy_output.logits)
            #print(reward, t, value, done, truncated)
            values[t]=value

            t+=1
            state = next_state

        cumulative_rewards = compute_cumulative_list_inverse(rewards)
        mean_cumulative_rewards = [(x + y)/num_episodes for x, y in zip(cumulative_rewards, mean_cumulative_rewards)]
        #mean_values = [(x + y)/num_steps_traj for x, y in zip(values, mean_values)]
        mean_values = values


    return mean_cumulative_rewards, mean_values









policy_name = "policy.pth"
num_episodes = 1

#x x x x x x x x
#x x x x x x x x
#x x 0 0 0 x x x
#x x 0 0 0 x x x
#x x 0 0 0 x x x
#x x x x x x x x


ENV_NAME = CustomEnv.HalfCheetah_v5

config = {
    "NAME_ENV": ENV_NAME,
    "NAME_EXPERT_DATA": ENV_NAME + "_data",
    #"POLICY_NAME": PolicyType.imitation_learning,
    #"POLICY_NAME": PolicyType.bcq_continuous,
    #"POLICY_NAME": PolicyType.bcq_discrete,
    "POLICY_NAME": PolicyName.cql_continuous,
    #"POLICY_NAME": PolicyType.dqn,
    #"POLICY_NAME": PolicyType.cql_discrete,
    "RENDER_MODE": RenderMode.RGB_ARRAY_LIST,
}



#config = {
#    "NAME_ENV": "SimpleGrid-8x8-v1",
#    "NAME_EXPERT_DATA": "simple_grid-gymnasium-data-v1",
#    "POLICY_NAME": PolicyType.cql_continuous,
#    "render_mode": "rgb_array_list",
#}

#config = {
#    "NAME_ENV": "ivan_1d_grid-gymnasium-v0",
#    "NAME_EXPERT_DATA": "ivan_1d_grid-gymnasium-data-v0",
#    "POLICY_NAME": PolicyType.bcq_discrete,
#    "render_mode": "human",
#}


render_mode = config["RENDER_MODE"]
log_name = os.path.join(config["NAME_ENV"], config["POLICY_NAME"])
LOG_DIR = "../log"
log_path = os.path.join(LOG_DIR, log_name)
config_name = "config.json"

os.makedirs(log_path, exist_ok=True)

with open(os.path.join(log_path, config_name), "w") as json_file:
    json.dump(config, json_file, indent=4)


## Define model to train
exploraton_noise = False


register_grid_envs()
env = gym.make(config["NAME_ENV"], render_mode=render_mode)


#print("##### ", env.max_num_steps)

policy = PolicyRestorationConfigFactoryRegistry[config["POLICY_NAME"]](action_space=env.action_space, observation_space=env.observation_space)
policy.load_state_dict(torch.load( os.path.join(log_path, policy_name), map_location="cpu"))

mean_cumulative_rew, mean_values = compute_mean_discounted_cumulative_reward(env, policy, num_episodes=20, gamma=0.99)

import matplotlib.pyplot as plt
plt.plot(mean_cumulative_rew)
plt.show()
plt.plot(mean_values)
plt.show()

#print(mean_cumulative_rew)

print(mean_cumulative_rew)
print(mean_values)


#q_matrix, _ = get_q_value_matrix(env, policy, gamma=0.99)
#for key, value in q_matrix.items():
#    print(f"{key}:{value}")