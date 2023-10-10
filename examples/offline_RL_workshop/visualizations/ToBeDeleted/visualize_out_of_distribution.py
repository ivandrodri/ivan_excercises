import json
import os
import gymnasium as gym
import minari
import numpy as np
import torch
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName, PolicyFactoryRegistry
from examples.offline_RL_workshop.utils import extract_dimension, one_hot_to_integer, state_action_histogram, \
    compare_state_action_histograms
from tianshou.data import Collector, Batch

policy_name = "policy.pth"
num_episodes = 20

config = {
    "NAME_ENV": "SimpleGrid-8x8-v1",
    "NAME_EXPERT_DATA": "simple_grid-gymnasium-data-v1",
    #"POLICY_NAME": PolicyType.imitation_learning,
    #"POLICY_NAME": PolicyType.dqn,
    #"POLICY_NAME": PolicyType.bcq_discrete,
    #"POLICY_NAME": PolicyType.bcq_continuous,
    "POLICY_NAME": PolicyName.cql_continuous,
    "render_mode": "rgb_array_list",
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


render_mode = config["render_mode"]
log_name = os.path.join(config["NAME_ENV"], config["POLICY_NAME"])
LOG_DIR = "../log"
log_path = os.path.join(LOG_DIR, log_name)
config_name = "config.json"

os.makedirs(log_path, exist_ok=True)

with open(os.path.join(log_path, config_name), "w") as json_file:
    json.dump(config, json_file, indent=4)


## Define model to train
exploraton_noise = False

env = gym.make(config["NAME_ENV"], render_mode=render_mode)


policy = PolicyFactoryRegistry[config["POLICY_NAME"]](action_space=env.action_space, observation_space=env.observation_space)
policy.load_state_dict(torch.load( os.path.join(log_path, policy_name), map_location="cpu"))


test_collector = Collector(policy, env, exploration_noise=exploraton_noise)


data = minari.load_dataset(config["NAME_EXPERT_DATA"])
print(f"number of elements: {len(data)}")

state_shape = extract_dimension(env.observation_space)
action_shape = extract_dimension(env.action_space)
state_action_count_data = {(int1, int2): 0 for int1 in range(state_shape+1) for int2 in range(action_shape)}


for episode in data:

    for observation, action in zip(episode.observations, episode.actions):
        action_value = int(action[0]) if isinstance(action, np.ndarray) and \
                                         (action.shape == (1,) or action.shape == (4,)) else action
        state_action_count_data[(one_hot_to_integer(observation), action_value)]+=1

state_action_histogram(state_action_count_data)



state_action_count_policy = {(int1, int2): 0 for int1 in range(state_shape+1) for int2 in range(action_shape)}

for i in range(num_episodes):
    done = False
    truncated = False
    state, _ = env.reset()
    while not (done or truncated):

        # For BCQ

        '''
        action_distrib, _ = policy.imitator(state.reshape(1,state_shape))
        categorical = torch.distributions.Categorical(logits=action_distrib[0])
        action = int(categorical.sample())
        '''

        tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
        action_prob = policy(tensor_state)
        ''' 
        # BCQ - Continuous
        #categorical = torch.distributions.Categorical(logits=action_prob.logits)
        categorical = torch.distributions.Categorical(logits=action_prob.q_value)
        action = int(categorical.sample())
        '''

        if isinstance(env.action_space, gym.spaces.Box):
            action = int(torch.argmax(action_prob.act[0]))
        else:
            action = int(action_prob.act.numpy())

        state_action_count_policy[(one_hot_to_integer(state), action)] += 1
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
    print(i)

env.close()


# policy_data_distribution
state_action_histogram(state_action_count_policy)

compare_state_action_histograms(state_action_count_data, state_action_count_policy)