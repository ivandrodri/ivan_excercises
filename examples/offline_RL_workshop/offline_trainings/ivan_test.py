from minari.storage import get_dataset_path
import os
import json

path_env = os.environ.get('PATH')
print("ENVS ", path_env)

ld_library_path = os.environ.get('LD_LIBRARY_PATH')
minari_datasets_path = os.environ.get('MINARI_DATASETS_PATH')
trained_policy_path = os.environ.get('TRAINED_POLICY_PATH')

print("LD_LIBRARY_PATH:", ld_library_path)
print("MINARI_DATASETS_PATH:", minari_datasets_path)
print("TRAINED_POLICY_PATH:", trained_policy_path)

dataset_id = "relocate-cloned-v1"
filename = get_dataset_path(dataset_id)
#filename = "/tianshou/offline_data/relocate-cloned-v1/"

print("######### ", filename)

with open(os.path.join(filename, "config.json"), 'r') as file:
        config_dict = json.load(file)

print(config_dict)

