from typing import Sequence, Union, Optional, Callable, Any, Tuple, Dict

import d4rl
import gymnasium
import gym
import minari
import numpy as np
import torch
from minari import DataCollectorV0
from torch import nn
from torch.utils.data import DataLoader

from examples.atari.atari_wrapper import WarpFrame, FrameStack
#from examples.custom_envs.custom_grid_env_d4rl import CustomGridEnv, custom_grid_env_registration
#from examples.offline.utils import load_buffer_minari

#import minari
#from minari import DataCollectorV0
#from minari.data_collector.callbacks import StepDataCallback


input_dim = 1

class DQNVector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()

        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),  # Adjust input_dim and hidden layers as needed
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, input_dim)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, 128)),  # Adjust output_dim and hidden layers as needed
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, np.prod(action_shape)))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs)


dqn = DQNVector(input_dim=1, action_shape=1, features_only=False, output_dim=1)

data = torch.FloatTensor([[1],[2],[3]])
new_data = dqn(data).flatten(1)
print(new_data)


#from tianshou.env import SubprocVectorEnv
#from examples.custom_envs.custom_grid_env_d4rl import CustomGridEnv, custom_grid_policy
from examples.custom_envs.custom_grid_env_minari import CustomGridEnvGymnasium
#env = gymnasium.make("ivan_1d_grid-gymnasium-v0") #gym.make("PointMaze_UMaze-v3")

#env = gym.make("CartPole-v1")
#test_envs = SubprocVectorEnv([env])

#env = CustomGridEnv(grid_size=10, initial_state=2, target_state=7)

#test_num = 1
#test_envs = SubprocVectorEnv(
#        [lambda: env for _ in range(test_num)]
#    )

#print(test_envs)


#NAME_ENV = "AdroitHandPen-v1"
#NAME_EXPERT_DATA="pen-expert-v1"
#D4RL = False

'''
def collate_fn(batch):
    print(type(batch[0]))
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "seed": torch.Tensor([x.seed for x in batch]),
        "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations["image"]) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        ),
        "timesteps": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.infos["timestep"]) for x in batch],
             batch_first=True
        )
    }

NAME_EXPERT_DATA = "ivan_1d_grid-gymnasium-data-v0"
#NAME_EXPERT_DATA="pen-expert-v1"
data = minari.load_dataset(NAME_EXPERT_DATA)

for elem in data:
    print(elem.actions)

#dataloader = DataLoader(data, batch_size=20, collate_fn=collate_fn)


#for action in data[0].actions:
#replay_buffer = load_buffer_minari(NAME_EXPERT_DATA)
#print(replay_buffer.last_index)


#for elem in replay_buffer:
#    print(elem.act)

'''


'''
import h5py
import numpy as np

def convert_to_hdf5_compatible(data):
    for k, v in data.items():
        if isinstance(v, list):
            # Check if the list contains integer values
            if all(isinstance(item, int) for item in v):
                data[k] = np.array(v, dtype=np.int32)
            else:
                # If not, convert it to an empty NumPy array with dtype=int32
                data[k] = np.array([], dtype=np.int32)

# Example usage:
data = {
    'terminals': [True, False, True],
    'infos/move_sequence': [[], [1, 2, 3], [], [4, 5, 6]],
    'other_data': [1.0, 2.0, 3.0]
}

# Convert 'infos/move_sequence' to HDF5-compatible format
convert_to_hdf5_compatible(data)

# Save the data to an HDF5 file
with h5py.File('your_file.h5', 'w') as hdf5_file:
    for key, value in data.items():
        hdf5_file.create_dataset(key, data=value)


#def create_custom_env(initial_state, target_state, grid_size):
#    return CustomGridEnv(initial_state, target_state, grid_size)

#custom_grid_env_registration()
'''

#env = gym.make("ivan_1d_grid-v0")
#env.get_dataset()
##print(env.max_episode_steps)
#d4rl.qlearning_dataset(gym.make("ivan_1d_grid-v0"))

#import gymnasium as gym


'''
frame_stack = 3

env = gym.make("PongNoFrameskip", render_mode="human")
env = WarpFrame(env)
env = FrameStack(env, frame_stack)
observation, info = env.reset(seed=42)
print(observation.shape)
for _ in range(100000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   #print("action: ", action, reward)

   print(observation)

   if reward!=0:
       print("#### ", observation, reward, terminated, truncated, info)
       #break

   #if terminated or truncated:
   #   observation, info = env.reset()

'''


'''
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import minari
from minari import DataCollectorV0
from minari.data_collector.callbacks import StepDataCallback

env = gym.make("PointMaze_UMaze-v3")

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
        del step_data["observations"]["achieved_goal"]
        return step_data


dataset_id = "point-maze-subseted-v3"

# delete the test dataset if it already exists
local_datasets = minari.list_local_datasets()
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)

env = DataCollectorV0(
    env,
    observation_space=observation_space_subset,
    # action_space=action_space_subset,
    step_data_callback=CustomSubsetStepDataCallback,
)
num_episodes = 10

env.reset(seed=42)
print(env.observation_space)

for episode in range(num_episodes):
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()  # Choose random actions
        observation, _, terminated, truncated, _ = env.step(action)

        #print(observation)
    env.reset()

# Create Minari dataset and store locally
dataset = minari.create_dataset_from_collector_env(
    dataset_id=dataset_id,
    collector_env=env,
    algorithm_name="random_policy",
)

print(dataset.sample_episodes(1)[0].observations["desired_goal"].shape)

env.reset(seed=42)

dataset_loaded = load_buffer_minari(dataset_id)

print(dataset_loaded)
'''