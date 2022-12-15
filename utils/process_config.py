from .load_config import Config
import os

def create_config_dict(config_file_path):
    config = Config()
    config_dict = config(os.path.abspath(config_file_path))

    ## dataset_config
    dataset_config_dict = config(os.path.abspath(config_dict["data"]["datasets_file_path"]))

    config_dict["data"]["datasets_split"] = dataset_config_dict

    ## augmentations_config

    augmentations_config_dict = config(os.path.abspath(config_dict["data"]["augmentations_file_path"]))

    config_dict["data"]["augmentations"] = augmentations_config_dict

    ## model architecture_config
    if "model_architecture_file_path" in config_dict["model"].keys():
        model_architecture_config_dict = config(os.path.abspath(config_dict["model"]["model_architecture_file_path"]))

        config_dict["model"]["architecture_config"] = model_architecture_config_dict

        set_var_vals_by_name(config_dict, config_dict)

    return config_dict

def set_var_vals_by_name(config, global_config_dict):
    iter_items_list = config.keys() if isinstance(config, dict) else range(len(config)) if isinstance(config, list) else []
    for key in iter_items_list:
        if isinstance(config[key], dict):
            set_var_vals_by_name(config[key], global_config_dict)
            continue
        if isinstance(config[key], list):
            # for elem in config[key]:
            set_var_vals_by_name(config[key], global_config_dict)
            continue
        if isinstance(config[key], str) and "?" in config[key]:
            query_var_name = config[key].replace("?", "")
            result_dict = {}
            search_recursively_config_dict(query_var_name, global_config_dict, result_dict)
            if result_dict:
                config[key] = result_dict[query_var_name]
            else:
                raise ValueError("Config variable substitution for variable " + query_var_name + " could not be found!")

def search_recursively_config_dict(query_key, config, result_dict):
    iter_items_list = config.keys() if isinstance(config, dict) else range(len(config)) if isinstance(config, list) else []
    for key in iter_items_list:
        if isinstance(config[key], dict):
            search_recursively_config_dict(query_key, config[key], result_dict)
            continue
        if isinstance(config[key], list):
            search_recursively_config_dict(query_key, config[key], result_dict)
            continue
        if key == query_key and config[key] and not isinstance(config[key], dict) and not isinstance(config[key], list): ## possible error source if variable not unique or same name used as broader name
            result_dict[key] = config[key]
        # if isinstance(config_dict[key], str):
        #     return search_recursively_config_dict(query_key, config_dict[key])

