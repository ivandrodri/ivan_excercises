import minari
from minari import combine_datasets
from minari.utils import validate_datasets_to_combine

from examples.offline_RL_workshop.custom_envs.custom_envs_registration import CustomEnv
from examples.offline_RL_workshop.generate_custom_datasets.generate_minari_dataset_grid_envs import MinariDatasetConfig
from examples.offline_RL_workshop.generate_custom_datasets.utils import generate_compatible_minari_dataset_name
from examples.offline_RL_workshop.utils import delete_minari_data_if_exists

minari_dataset_1 = "Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_0-v0"
minari_dataset_2 = "Grid_2D_8x8_discrete-data_obst_middle_8x8_start_3_0_target_7_7-v0"
NAME_COMBINED_DATASET = "combined_data_set"



list_dataset_names = [minari_dataset_1, minari_dataset_2]
minari_datasets = [
    minari.load_dataset(dataset_id) for dataset_id in list_dataset_names
]

name_combined_dataset = generate_compatible_minari_dataset_name(
    env_name=CustomEnv.Grid_2D_8x8_discrete,
    data_set_name=NAME_COMBINED_DATASET,
    version="V0"
)

delete_minari_data_if_exists(name_combined_dataset)
combined_dataset = combine_datasets(
    minari_datasets, new_dataset_id=name_combined_dataset
)
print(f"Number of episodes in dataset A:{len(minari_datasets[0])}, in dataset B:{len(minari_datasets[1])} and  "
      f"in combined dataset: {len(combined_dataset)}")


### ToDo Attach a json for the environment
minari_combined_dataset = MinariDatasetConfig.load_from_file(minari_dataset_1)
minari_combined_dataset.data_set_name = name_combined_dataset

# ToDo: Add more than one behavior policies to config.
#minari_combined_dataset.behavior_policy = []

minari_combined_dataset.initial_config_2d_grid_env.target_state = (7, 7)
minari_combined_dataset.save_to_file()


