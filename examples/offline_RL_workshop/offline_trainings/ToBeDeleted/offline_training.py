import json
import os
import gymnasium as gym
import numpy as np
import torch
from examples.offline.utils import load_buffer_minari
from examples.offline_RL_workshop.offline_trainings.custom_tensorboard_callbacks import CustomSummaryWriter
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyType, PolicyRestorationConfigFactoryRegistry
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger


#config = {
#    "NAME_ENV": "ivan_1d_grid-gymnasium-v0",
#    "NAME_EXPERT_DATA": "ivan_1d_grid-gymnasium-data-v0",
#    "POLICY_NAME": PolicyType.dqn,
#    #"POLICY_NAME": PolicyType.bcq_discrete,
#    "RENDER_MODE": "human",
#}

config = {
    "NAME_ENV": "SimpleGrid-8x8-v0",
    "NAME_EXPERT_DATA": "simple_grid-gymnasium-data-v0",
    #"POLICY_NAME": PolicyType.imitation_learning,
    #"POLICY_NAME": PolicyType.bcq_continuous,
    "POLICY_NAME": PolicyType.bcq_discrete,
    #"POLICY_NAME": PolicyType.cql_continuous,
    #"POLICY_NAME": PolicyType.dqn,
    #"POLICY_NAME": PolicyType.cql_discrete,
    "RENDER_MODE": "rgb_array_list",
}



#config = {
#    "NAME_ENV": "SimpleGrid-8x8-v1",
#    "NAME_EXPERT_DATA": "simple_grid-gymnasium-data-v1",
#    "POLICY_NAME": PolicyType.bcq_continuous,
#    #"POLICY_NAME": PolicyType.cql_continuous,
#    "RENDER_MODE": "rgb_array_list",
#}



log_name = os.path.join(config["NAME_ENV"], config["POLICY_NAME"])
LOG_DIR = "../../log"
log_path = os.path.join(LOG_DIR, log_name)
config_name = "config.json"

os.makedirs(log_path, exist_ok=True)

with open(os.path.join(log_path, config_name), "w") as json_file:
    json.dump(config, json_file, indent=4)



seed = None #1626

EPOCH = 200
BATCH_SIZE = 64
UPDATE_PER_EPOCH = 200

render_mode = config["RENDER_MODE"]

#### collector test params
test_num = 1
exploraton_noise = True




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
env = gym.make(config["NAME_ENV"], render_mode=render_mode)


policy = PolicyRestorationConfigFactoryRegistry[config["POLICY_NAME"]](action_space=env.action_space, observation_space=env.observation_space)
test_collector = Collector(policy, test_envs, exploration_noise=exploraton_noise)


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


result = offline_trainer(
        policy,
        buffer,
        test_collector,
        EPOCH,
        UPDATE_PER_EPOCH,
        10,
        BATCH_SIZE,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )


# Save final policy
torch.save(policy.state_dict(), os.path.join(log_path, 'policy_final.pth'))

final_policy_name = "policy.pth"
final_policy = PolicyRestorationConfigFactoryRegistry[config["POLICY_NAME"]](action_space=env.action_space, observation_space=env.observation_space)
final_policy.load_state_dict(torch.load( os.path.join(log_path, final_policy_name), map_location="cpu"))
final_collector = Collector(final_policy, env, exploration_noise=exploraton_noise)
final_collector.collect(n_episode=20, render=1 / 35)

