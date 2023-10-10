import json
import os
import gymnasium as gym
import torch
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName, PolicyFactoryRegistry
from examples.offline_RL_workshop.utils import get_q_value_matrix

policy_name = "policy.pth"
num_episodes = 20

config = {
    "NAME_ENV": "SimpleGrid-8x8-v0",
    "NAME_EXPERT_DATA": "simple_grid-gymnasium-data-v0",
    #"POLICY_NAME": PolicyType.dqn,
    "POLICY_NAME": PolicyName.bcq_discrete,
    #"POLICY_NAME": PolicyType.bcq_continuous,
    #"POLICY_NAME": PolicyType.cql_continuous,
    #"POLICY_NAME": PolicyType.imitation_learning,
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




q_value_matrix, policy_trajectory_reward = get_q_value_matrix(env, policy)
q_value_matrix = {(env.to_xy(key[0]), key[1]):value for key, value in q_value_matrix.items()}

for key, value in q_value_matrix.items():
    print(f"{key}: {value} - policy_trajectory_reward:{policy_trajectory_reward}")

# collector = Collector(policy, env, exploration_noise=True)
# collector.collect(n_episode=3000, render=1 / 35)

#t=0
#reward = 0
#gamma = 0.99
#num_steps = 13
#for i in range(num_steps):
#    reward += (-0.1)*(gamma)**i

#print("Cum Rew. : ", reward)


