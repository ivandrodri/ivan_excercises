import gymnasium
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import minari
from minari import DataCollectorV0, MinariDataset
from minari.data_collector.callbacks import StepDataCallback

from examples.offline.utils import load_buffer_minari
from tianshou.data import ReplayBuffer

from examples.custom_envs.custom_grid_env_d4rl import CustomGridEnv, custom_grid_env_registration


NAME_ENV = "PointMaze_UMaze-v3"
NAME_EXPERT_DATA = "point-maze-subseted_Ivan_v1"

#env = gym.make(NAME_ENV)
grid_size = 10
target_state = 7
initial_state = 2

#env = CustomGridEnv(grid_size, initial_state, target_state)
env = gymnasium.make("ivan_1d_grid-v0")




dataset_id = NAME_EXPERT_DATA

print(f"Observation space: {env.observation_space}")

observation_space_subset = spaces.Dict(
    {
        # "achieved_goal": spaces.Box(low=float('-inf'), high=float('inf'), shape=(2,), dtype=np.float64),
        "desired_goal": spaces.Box(
            low=float("-inf"), high=float("inf"), shape=(2,), dtype=np.float64
        ),
        "observation": spaces.Box(
            low=float("-inf"), high=float("inf"), shape=(4,), dtype=np.float64
        ),
    }
)


class CustomSubsetStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        #del step_data["observations"]["achieved_goal"]
        return step_data


# ToDo: I should load a dataset and if it exist has the possibility to add data to it versioning it.

# See if dataset in local folder offline/.minari/datasets
local_datasets = minari.list_local_datasets()
#dataset = None
#if dataset_id in local_datasets:
#    dataset = load_buffer_minari(dataset_id)
#    print(len(dataset))
# Delete dataset if in local folder offline/.minari/datasets
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)

env = DataCollectorV0(
    env,
    #observation_space=observation_space_subset,
    # action_space=action_space_subset,
    step_data_callback=CustomSubsetStepDataCallback,
)
num_episodes = 20

#env.reset(seed=42)
env.reset()
print(env.observation_space)

dataset=None
for episode in range(num_episodes):
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()  # Choose random actions
        observation, _, terminated, truncated, _ = env.step(action)

        #print(observation)
    env.reset()

    # Create Minari dataset and store locally
    if dataset is None:
        dataset = minari.create_dataset_from_collector_env(
            dataset_id=dataset_id,
            collector_env=env,
            algorithm_name="random_policy",
        )
    else:
        # Update local Minari dataset every 200000 steps.
        # This works as a checkpoint to not lose the already collected data
        #if isinstance(dataset, MinariDataset):
        dataset.update_dataset_from_collector_env(env)


env.reset(seed=42)







# Load minari dataset
