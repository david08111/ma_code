from .load_config import Config
import os

def create_config_dict(config_file_path):
    config = Config()
    config_dict = config(os.path.abspath(config_file_path))

    dataset_config_dict = config(os.path.abspath(config_dict["data"]["datasets_file_path"]))

    config_dict["data"]["datasets_split"] = dataset_config_dict

    # config_dict["data"].update(dataset_config_dict)

    return config_dict