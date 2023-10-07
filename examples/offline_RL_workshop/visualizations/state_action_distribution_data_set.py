import gymnasium as gym
from examples.offline.utils import load_buffer_minari
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs, CustomEnv, RenderMode
from examples.offline_RL_workshop.generate_custom_datasets.utils import generate_compatible_minari_dataset_name
from examples.offline_RL_workshop.utils import state_action_histogram
from examples.offline_RL_workshop.visualizations.utils import get_state_action_data_and_policy_grid_distributions


ENV_NAME = CustomEnv.Grid_2D_8x8_discrete_V1_A
DATA_SET_NAME = "data"
VERSION_DATA_SET = "v0"

DATASET_CONFIG = {
    "env_name": ENV_NAME,
    "data_set_name": generate_compatible_minari_dataset_name(ENV_NAME, DATA_SET_NAME, VERSION_DATA_SET),
    "render_mode": RenderMode.RGB_ARRAY_LIST,
}

register_grid_envs()
env = gym.make(DATASET_CONFIG["env_name"], render_mode=DATASET_CONFIG["render_mode"])

data = load_buffer_minari(DATASET_CONFIG["data_set_name"])
print(f"number of elements: {len(data)}")

state_action_count_data, state_action_count_policy = \
    get_state_action_data_and_policy_grid_distributions(data, env)
state_action_histogram(state_action_count_data)



