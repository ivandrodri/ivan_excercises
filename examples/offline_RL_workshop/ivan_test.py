from minari.storage import get_dataset_path
import os
import json

#os.environ['MINARI_DATASETS_PATH'] = "/tianshou/offline_data"

dataset_id = "relocate-cloned-v1"
filename = get_dataset_path(dataset_id)
#filename = "/tianshou/offline_data/relocate-cloned-v1/"

print("######### ", filename)

with open(os.path.join(filename, "config.json"), 'r') as file:
        config_dict = json.load(file)

print(config_dict)
