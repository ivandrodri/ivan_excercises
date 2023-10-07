import json
import os
import gymnasium as gym
import numpy as np
import torch
from examples.offline.utils import load_buffer_minari
from examples.offline_RL_workshop.offline_trainings.custom_tensorboard_callbacks import CustomSummaryWriter
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyType, PolicyRestorationConfigFactoryRegistry
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger


#config = {
#    "NAME_ENV": "ivan_1d_grid-gymnasium-v0",
#    "NAME_EXPERT_DATA": "ivan_1d_grid-gymnasium-data-v0",
#    "POLICY_NAME": PolicyType.dqn,
#    "RENDER_MODE": "human",
#}

config = {
    "NAME_ENV": "SimpleGrid-8x8-v0",
    "NAME_EXPERT_DATA": "simple_grid-gymnasium-data-v0",
    "POLICY_NAME": PolicyType.dqn,
    "RENDER_MODE": "rgb_array_list",
}

TRAIN = False


log_name = os.path.join(config["NAME_ENV"], config["POLICY_NAME"])
LOG_DIR = "../log/online"
log_path = os.path.join(LOG_DIR, log_name)
config_name = "config.json"

os.makedirs(log_path, exist_ok=True)

with open(os.path.join(log_path, config_name), "w") as json_file:
    json.dump(config, json_file, indent=4)



seed = None #1626

EPOCH = 100
BATCH_SIZE = 256
UPDATE_PER_EPOCH = 80
STEP_PER_COLLECT = 128

render_mode = config["RENDER_MODE"]

#### collector test params
test_num = 1
train_num = 10
exploraton_noise = True
buffer_size = 20000





if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

## Define model to train
device = "cpu"


## Create environments
#load_custom_grid_env()
test_envs = SubprocVectorEnv(
        [lambda: gym.make(config["NAME_ENV"]) for _ in range(test_num)]
    )
train_envs = SubprocVectorEnv(
        [lambda: gym.make(config["NAME_ENV"]) for _ in range(train_num)]
    )
env = gym.make(config["NAME_ENV"], render_mode=render_mode)


policy = PolicyRestorationConfigFactoryRegistry[config["POLICY_NAME"]](action_space=env.action_space, observation_space=env.observation_space)




test_collector = Collector(policy, test_envs, exploration_noise=exploraton_noise)
train_collector = Collector(policy, train_envs, VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)


## ToDo: LOAD D4RL DATASET
buffer = load_buffer_minari(config["NAME_EXPERT_DATA"])
print("length data ", len(buffer))

def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

def stop_fn(mean_rewards):
    return False

# Tensorboard writer
custom_writer = CustomSummaryWriter(log_path, env)
custom_writer.log_custom_info()
logger = TensorboardLogger(custom_writer)


result = offpolicy_trainer(
        policy=policy,
        train_collector = train_collector,
        test_collector = test_collector,
        max_epoch = EPOCH,
        step_per_epoch = UPDATE_PER_EPOCH,
        step_per_collect = STEP_PER_COLLECT,
        episode_per_test = 10,
        batch_size = BATCH_SIZE,
        stop_fn=stop_fn,
        save_best_fn = save_best_fn,
        logger=logger
)

# Save final policy
torch.save(policy.state_dict(), os.path.join(log_path, 'policy_final.pth'))

final_policy_name = "policy.pth"
final_policy = PolicyRestorationConfigFactoryRegistry[config["POLICY_NAME"]](action_space=env.action_space, observation_space=env.observation_space)
final_policy.load_state_dict(torch.load( os.path.join(log_path, final_policy_name), map_location="cpu"))
final_collector = Collector(final_policy, env, exploration_noise=exploraton_noise)
final_collector.collect(n_episode=20, render=1 / 35)



