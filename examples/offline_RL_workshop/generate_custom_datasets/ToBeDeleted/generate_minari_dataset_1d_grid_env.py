import gymnasium as gym
import minari
import numpy as np
from minari import DataCollectorV0
from minari.data_collector.callbacks import StepDataCallback

from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_custom_grid_env

NAME_ENV = "ivan_1d_grid-gymnasium-v0"
NAME_EXPERT_DATA = "ivan_1d_grid-gymnasium-data-v0"
NUM_STEPS = 1000

register_custom_grid_env()

env = gym.make(NAME_ENV)
dataset_id = NAME_EXPERT_DATA


print(f"Observation space: {env.observation_space}")

local_datasets = minari.list_local_datasets()
# Delete dataset if in local folder offline/.minari/datasets
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)


class CustomSubsetStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        #del step_data["observations"]["achieved_goal"]
        return step_data


env = DataCollectorV0(
    env,
    step_data_callback=CustomSubsetStepDataCallback,
    record_infos=False,
)

state, _ = env.reset()

num_steps = 0
for _ in range(NUM_STEPS):

    action_probs = custom_1d_grid_policy(state)
    if env.discrete_action:
        action = np.random.choice(env.action_space.n, p=action_probs)
    else:
        action_range = np.linspace(env.action_space.low, env.action_space.high, num=len(action_probs)).flatten()
        action = np.array([np.random.choice(action_range, p=action_probs)])

    next_state, reward, done, time_out, info = env.step(action)
    num_steps += 1

    if done or time_out:
        state, _ = env.reset()
        num_steps=0
    else:
        state = next_state

dataset = minari.create_dataset_from_collector_env(dataset_id=NAME_EXPERT_DATA, collector_env=env)


data = minari.load_dataset(NAME_EXPERT_DATA)
print("number of episodes collected: ",len(data))
for elem in data:
    print(elem.actions)#, elem.truncations, elem.terminations)



#ToDo:
# 1 - add info to Minari: bug in Minari - issue open: https://github.com/Farama-Foundation/Minari/issues/125
# 2 - use bcq minari
# 3 - Maybe add a test such that datset and replybuffer containes same data.
