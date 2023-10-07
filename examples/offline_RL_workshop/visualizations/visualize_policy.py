import os
import gymnasium as gym
import torch

import tianshou.policy
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs, RenderMode
from examples.offline_RL_workshop.custom_envs.utils import is_instance_of_SimpleGridEnv, InitialConfigEnvWrapper
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyType, \
    PolicyRestorationConfigFactoryRegistry
from examples.offline_RL_workshop.offline_trainings.policy_config_data_class import OfflineTrainedPolicyConfig, \
    get_trained_policy_path
from examples.offline_RL_workshop.offline_trainings.restore_policy_model import restore_trained_offline_policy
from tianshou.data import Collector



NAME_EXPERT_DATA = "relocate-cloned-v1"
#"Grid_2D_8x8_discrete-combined_data_set-V0"
#"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_7-v0"
#"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_7-v0"
#"Grid_2D_4x4_discrete-data_obst_free_4x4_start_0_0_target_3_3-v0"
#"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_7-v0"
#"Ant-v2-data-v0"
POLICY_TYPE = PolicyType.cql_continuous
EXPLORATION_NOISE = True
POLICY_NAME = "policy.pth"

offline_policy_config = OfflineTrainedPolicyConfig(
    name_expert_data=NAME_EXPERT_DATA,
    policy_name=POLICY_TYPE,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    device="cpu"
)

policy = restore_trained_offline_policy(offline_policy_config)

name_expert_data = offline_policy_config.name_expert_data
log_name = os.path.join(name_expert_data, POLICY_TYPE)
log_path = get_trained_policy_path(log_name)
policy.load_state_dict(torch.load(os.path.join(log_path, POLICY_NAME), map_location="cpu"))


env_name = offline_policy_config.minari_dataset_config.env_name
render_mode = offline_policy_config.render_mode
env_config = offline_policy_config.minari_dataset_config.initial_config_2d_grid_env

env = InitialConfigEnvWrapper(gym.make(env_name, render_mode=render_mode),
                              env_config=env_config)


final_collector = Collector(policy, env, exploration_noise=EXPLORATION_NOISE)
final_collector.collect(n_episode=20, render=1 / 35)


















