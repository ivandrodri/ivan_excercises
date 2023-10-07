import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import Wrapper

from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import ObstacleTypes
from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.simple_grid import SimpleGridEnv


def is_instance_of_SimpleGridEnv(env):
    while isinstance(env, gym.Env):
        if isinstance(env, SimpleGridEnv):
            return True
        if hasattr(env, 'env'):
            env = env.env
        else:
            break
    return False


@dataclass
class Grid2DInitialConfig:
    obstacles: ObstacleTypes = None
    initial_state: Tuple = None
    target_state: Tuple = None

#    def to_json(self):
#        data = asdict(self)
#        data["obstacles"] = self.obstacles.value
#        return data

#    @classmethod
#    def from_json(cls, dic_data):
#        cls(**dic_data)




#initial_config = Grid2DInitialConfig(
#    obstacles=ObstacleTypes.obst_free_4x4
#)

#dic = initial_config.to_json()
#print(ObstacleTypes(dic["obstacles"]))

#print(dic["obstacles"])


#initial_config_2 = Grid2DInitialConfig.from_json(dic)
#print(dic)
#print(initial_config_2.obstacles)


# Inheritance from gym.utils.RecordConstructorArgs is needed as the wrapper must be recreated
#   with minari for dataset combination.

class InitialConfigEnvWrapper(Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, env_config: Grid2DInitialConfig = None):
        super().__init__(env)
        if is_instance_of_SimpleGridEnv(self.env):
            if env_config is not None:
                obstacles = env_config.obstacles
                if obstacles is not None:
                    self.env.set_new_obstacle_map(obstacles.value)
                initial_state = env_config.initial_state
                if initial_state is not None:
                    self.env.set_starting_point(initial_state)
                target_state = env_config.target_state
                if target_state is not None:
                    self.env.set_goal_point(target_state)
