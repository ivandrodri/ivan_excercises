import random
from typing import Dict

import numpy as np
import gymnasium as gym
from examples.offline_RL_workshop.utils import one_hot_to_integer

MOVES: Dict[int,tuple] = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1)   #RIGHT
    }


def suboptimal_behavior_policy_2d_grid_discrete(state: np.ndarray, env:gym.Env) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)

    possible_directions = [2, 3, 1]
    weights = [1, 1, 3]
    random_directions = random.choices(possible_directions, weights=weights)[0]

    if state_xy[0] == 7:
        possible_directions = [0, 1, 3]
        weights = [1, 1, 3]
        random_directions = random.choices(possible_directions, weights=weights)[0]

    return random_directions


def suboptimal_behavior_policy_2d_grid_discrete_case_a(state: np.ndarray, env:gym.Env) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    possible_directions = [2, 3, 1]
    weights = [1, 1, 1]
    random_directions = random.choices(possible_directions, weights=weights)[0]
    if random_directions == 3 and (state_xy[1] >2):#(state_xy[1] >= state_xy[0]):
        possible_directions = [2, 1]
        weights = [1, 1]
        return random.choices(possible_directions, weights=weights)[0]

    return random_directions


def suboptimal_behavior_policy_2d_grid_discrete_case_a(state: np.ndarray, env:gym.Env) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    possible_directions = [2, 3, 1]
    weights = [1, 1, 1]
    random_directions = random.choices(possible_directions, weights=weights)[0]
    if random_directions == 3 and (state_xy[1] > 2):#(state_xy[1] >= state_xy[0]):
        possible_directions = [2, 1]
        weights = [1, 1]
        return random.choices(possible_directions, weights=weights)[0]

    return random_directions


def suboptimal_behavior_policy_2d_grid_discrete_case_b(state: np.ndarray, env:gym.Env) -> int:
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy == (4, 0):
        return 1
    elif state_xy[0] == 5 and state_xy[1] < 7:
        return 3
    else:
        return 1



def suboptimal_behavior_policy_ivan(state: np.ndarray, env:gym.Env) -> int:
    bla=3
    return 1



