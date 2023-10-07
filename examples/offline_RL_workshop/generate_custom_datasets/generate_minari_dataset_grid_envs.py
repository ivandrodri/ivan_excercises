import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import gymnasium as gym
import minari
from minari import DataCollectorV0
from minari.data_collector.callbacks import StepDataCallback
from minari.storage import get_dataset_path

from examples.offline_RL_workshop.behavior_policies.behavior_policy_registry import BehaviorPolicyType, \
    BehaviorPolicyRestorationConfigFactoryRegistry
from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import ObstacleTypes
from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.simple_grid import SimpleGridEnv
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs, CustomEnv
from examples.offline_RL_workshop.custom_envs.utils import is_instance_of_SimpleGridEnv, Grid2DInitialConfig, \
    InitialConfigEnvWrapper
from examples.offline_RL_workshop.generate_custom_datasets.utils import generate_compatible_minari_dataset_name
from examples.offline_RL_workshop.utils import delete_minari_data_if_exists

OVERRIDE_DATA_SET = True


@dataclass
class MinariDatasetConfig:
    env_name: str
    data_set_name: str
    num_steps: int
    behavior_policy: BehaviorPolicyType
    initial_config_2d_grid_env: Grid2DInitialConfig = None

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)

    #def to_json(self):
    #    return json.dumps(asdict(self), indent=4)

    #@classmethod
    #def from_json(cls, json_str):
    #    config_dict = json.loads(json_str)
    #    return cls(**config_dict)

    def save_to_file(self):
        data_set_path = get_dataset_path(self.data_set_name)
        file_name = "config.json"

        # ToDo: Because Enum is not serializable: There should be a better way
        obj_to_saved = asdict(self)

        if self.initial_config_2d_grid_env is not None:
            obj_to_saved["initial_config_2d_grid_env"]["obstacles"] = \
                obj_to_saved["initial_config_2d_grid_env"]["obstacles"].value

        with open(os.path.join(data_set_path, file_name), 'w') as file:
            json.dump(obj_to_saved, file, indent=4)

    @classmethod
    def load_from_file(cls, dataset_id):
        filename = get_dataset_path(dataset_id)
        with open(os.path.join(filename, "config.json"), 'r') as file:
            config_dict = json.load(file)

        if config_dict["initial_config_2d_grid_env"] is not None:
            config_dict["initial_config_2d_grid_env"] = Grid2DInitialConfig(**config_dict["initial_config_2d_grid_env"])

            config_dict["initial_config_2d_grid_env"].obstacles = \
                ObstacleTypes(config_dict["initial_config_2d_grid_env"].obstacles)

        return cls(**config_dict)


def create_minari_datasets(dataset_config: MinariDatasetConfig):

    delete_minari_data_if_exists(dataset_config.data_set_name, override_dataset=OVERRIDE_DATA_SET)
    register_grid_envs()

    env = gym.make(dataset_config.env_name)
    env = InitialConfigEnvWrapper(env, env_config=dataset_config.initial_config_2d_grid_env)


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
    for _ in range(dataset_config.num_steps):

        behavior_policy = BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[dataset_config.behavior_policy]
        if dataset_config.behavior_policy == BehaviorPolicyType.random:
            action = env.action_space.sample()
        else:
            action = behavior_policy(state, env)

        next_state, reward, done, time_out, info = env.step(action)
        num_steps += 1

        if done or time_out:
            state, _ = env.reset()
            num_steps=0
        else:
            state = next_state


    dataset = minari.create_dataset_from_collector_env(dataset_id=dataset_config.data_set_name, collector_env=env)
    dataset_config.save_to_file()



#ToDo:
# 1 - add info to Minari: bug in Minari - issue open: https://github.com/Farama-Foundation/Minari/issues/125
# 2 - use bcq minari
# 3 - Maybe add a test such that datset and replybuffer containes same data.

#path = "/home/ivan/Documents/GIT_PROJECTS/Tianshou/tianshou/offline_data/Grid_2D_4x4_discrete-obst_free_4x4-v0/config.json"
#bla = MinariDatasetConfig.load_from_file(path)
#obst_type = bla.env_config["obstacles"]
#print(obst_type)
