import minari

from examples.offline_RL_workshop.behavior_policies.behavior_policy_registry import BehaviorPolicyType
from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import ObstacleTypes
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import CustomEnv
from examples.offline_RL_workshop.custom_envs.utils import Grid2DInitialConfig
from examples.offline_RL_workshop.generate_custom_datasets.generate_minari_dataset_grid_envs import \
    create_minari_datasets, MinariDatasetConfig
from examples.offline_RL_workshop.generate_custom_datasets.utils import generate_compatible_minari_dataset_name, \
    get_dataset_name_2D_grid

#ENV_NAME = CustomEnv.Grid_2D_8x8_discrete
ENV_NAME = "Ant-v2"
DATA_SET_NAME = "data"
DATA_SET_IDENTIFIER = ""
VERSION_DATA_SET = "v0"


DATA_SET_NAME += DATA_SET_IDENTIFIER
DATASET_CONFIG = {
    "env_name": ENV_NAME,
    "data_set_name": generate_compatible_minari_dataset_name(ENV_NAME, DATA_SET_NAME, VERSION_DATA_SET),
    "num_steps": 3000,
    "behavior_policy": BehaviorPolicyType.behavior_suboptimal_2d_grid_discrete_case_b,
}


# Only for 2D grid envs
INITIAL_CONDITIONS_2D_GRID = {
    "obstacles": ObstacleTypes.obst_middle_8x8,
    "initial_state": (3, 0),
    "target_state": (7, 7)
}
initial_condition_2d_grid = Grid2DInitialConfig(**INITIAL_CONDITIONS_2D_GRID)

if "Grid_2D" in ENV_NAME:
    DATASET_CONFIG["initial_config_2d_grid_env"] = initial_condition_2d_grid
    DATA_SET_NAME = get_dataset_name_2D_grid(initial_condition_2d_grid) + DATA_SET_IDENTIFIER
    DATASET_CONFIG["data_set_name"] = generate_compatible_minari_dataset_name(ENV_NAME, DATA_SET_NAME, VERSION_DATA_SET)

minari_dataset_config = MinariDatasetConfig.from_dict(DATASET_CONFIG)
create_minari_datasets(minari_dataset_config)

data = minari.load_dataset(DATASET_CONFIG["data_set_name"])
print("number of episodes collected: ", len(data))
for elem in data:
    print(elem.actions, elem.truncations, elem.terminations)
