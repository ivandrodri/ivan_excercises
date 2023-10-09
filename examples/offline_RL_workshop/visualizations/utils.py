from typing import Tuple, Dict, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName
from examples.offline_RL_workshop.utils import extract_dimension, one_hot_to_integer
from tianshou.data import ReplayBuffer, Batch

import gymnasium as gym

# ToDo: Improve this function: clean code, etc.!!


def get_state_action_data_and_policy_grid_distributions(
        data: ReplayBuffer,
        env: gym.Env,
        policy: Union[nn.Module, str, None]=None,
        num_episodes = 1,
) -> Tuple[Dict, Dict]:

    state_shape = extract_dimension(env.observation_space)
    action_shape = extract_dimension(env.action_space)

    state_action_count_data = {(int1, int2): 0 for int1 in range(state_shape + 1) for int2 in range(action_shape)}

    for episode_elem in data:
        observation = episode_elem.obs
        action = episode_elem.act

        action_value = int(action) if len(action.shape) == 0 or action.shape[0] <= 1 else np.argmax(action)
        state_action_count_data[(one_hot_to_integer(observation), action_value)] += 1

    if policy is not None:
        state_action_count_policy = {(int1, int2): 0 for int1 in range(state_shape + 1) for int2 in range(action_shape)}

        for i in tqdm(range(num_episodes), desc="Processing", ncols=100):
            done = False
            truncated = False
            state, _ = env.reset()
            while not (done or truncated):
                if policy != "random":
                    tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
                    policy_output = policy(tensor_state)
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        action = policy_output.act[0].detach().numpy()
                    else:
                        if isinstance(policy_output.act[0], torch.Tensor):
                            action = policy_output.act[0].detach().numpy()
                        else:
                            action = policy_output.act[0]
                else:
                    action = env.action_space.sample()

                action_value = int(action) if len(action.shape) == 0 or action.shape[0] <= 1 else np.argmax(action)
                state_action_count_policy[(one_hot_to_integer(state), action_value)] += 1
                next_state, reward, done, truncated, info = env.step(action_value)
                state = next_state

    else:
        state_action_count_policy = None

    return state_action_count_data, state_action_count_policy
