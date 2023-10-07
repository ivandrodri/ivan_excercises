import gymnasium as gym
import numpy as np

from examples.offline_RL_workshop.custom_envs.custom_1d_grid_env import INITIAL_STATE, TARGET_STATE
from examples.offline_RL_workshop.utils import one_hot_to_integer


def custom_1d_grid_policy(state: int, env: gym.Env, initial_state:int = INITIAL_STATE, target_state:int = TARGET_STATE):
    action_space = env.action_space
    def get_action(action_probs):

        if isinstance(action_space, gym.spaces.Discrete):
            action = np.random.choice(action_space.n, p=action_probs)
        elif isinstance(action_space, gym.spaces.Box):
            action_range = np.linspace(action_space.low, action_space.high, num=len(action_probs)).flatten()
            action = np.array([np.random.choice(action_range, p=action_probs)])
        else:
            raise ValueError(f"Only actions of type {type(gym.spaces.Discrete)} or {type(gym.spaces.Box)} are allowed")

        return action


    if initial_state < one_hot_to_integer(state) <= target_state-2:
        probs_left_right = [0.5, 0.5]
        return get_action(probs_left_right)
    elif target_state-2 < one_hot_to_integer(state) < target_state:
        probs_left_right = [1.0, 0.0]
        return get_action(probs_left_right)
    elif one_hot_to_integer(state)==initial_state:
        probs_left_right = [0.5, 0.5]
        return get_action(probs_left_right)
    else:
        probs_left_right = [0.5, 0.5]
        return get_action(probs_left_right)
