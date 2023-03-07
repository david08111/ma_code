import torch
import argparse
import yaml
import os
import sys
from utils import Config, create_config_dict, update_config_dict
from data_handling import DataHandler, custom_collate_fn, custom_collate_fn2
from training import Net_trainer
from training import Loss_Wrapper
from training import Metrics_Wrapper
from models import Model
from torch.utils.data import DataLoader

# def update_config_dict(config_dict_base, config_dict_update):
#     pass

def set_unique_save_log_paths(config_dict, args_list_update):
    for elem in args_list_update:
        key_val_split = elem.split("=")

        val = key_val_split[1]
        key = key_val_split[0]

        # remove "--" from front

        key = key[2:]

        # split by nested dictionary elems

        nested_dict_level_list = key.split(".")

        # for name in nested_dict_level_list[:-1]:
        #     config_dict["training"]["save_path"] = os.path.join(config_dict["training"]["save_path"], name)
        #     config_dict["logging"]["save_path"] = os.path.join(config_dict["logging"]["save_path"],
        #                                                         name[-1] + "_" + val)

        # config_dict["training"]["save_path"] = os.path.join(config_dict["training"]["save_path"], nested_dict_level_list[-1] + "_" + val)
        # config_dict["logging"]["save_path"] = os.path.join(config_dict["logging"]["save_path"],
        #                                                    nested_dict_level_list[-1] + "_" + val)

        run_name = nested_dict_level_list[0]
        for name in nested_dict_level_list[1:]:
            run_name += "-" + name

        run_name += "-" + val

        config_dict["training"]["save_path"] = os.path.join(config_dict["training"]["save_path"], run_name)
        config_dict["logging"]["save_path"] = os.path.join(config_dict["logging"]["save_path"], run_name)

    return config_dict

def add_nested_dict_from_list(config_dict, nested_dict_level_list, val):
    if len(nested_dict_level_list) > 1:
        key = nested_dict_level_list.pop(0)
        if key in config_dict.keys():
            add_nested_dict_from_list(config_dict[key], nested_dict_level_list, val)
        else:
            config_dict[key] = {}
            add_nested_dict_from_list(config_dict[key], nested_dict_level_list, val)
    elif len(nested_dict_level_list) == 1:
        parse_string = f'{{"{nested_dict_level_list[0]}": {val}}}'
        config_dict_tmp = yaml.safe_load(parse_string)
        final_val = config_dict_tmp[nested_dict_level_list[0]]

        config_dict[nested_dict_level_list[0]] = final_val

def create_nested_dict_from_list(nested_dict_level_list, val):
    if len(nested_dict_level_list) > 1:
        key = nested_dict_level_list.pop(0)
        return {key: create_nested_dict_from_list(nested_dict_level_list)}
    elif len(nested_dict_level_list) == 1:
        return {nested_dict_level_list[0]: val}

def convert_arg_list2config_dict(config_dict, arg_list):

    # config_dict = {}

    for elem in arg_list:
        # split into key and value with "="
        key_val_split = elem.split("=")

        val = key_val_split[1]
        key = key_val_split[0]

        # remove "--" from front

        key = key[2:]

        # split by nested dictionary elems

        nested_dict_level_list = key.split(".")

        # nested_dict_elem = create_nested_dict_from_list(nested_dict_level_list, val)
        add_nested_dict_from_list(config_dict, nested_dict_level_list, val)

    return config_dict

def train_net(config_dict):
    torch.manual_seed(10)

    torch.backends.cudnn.benchmark = True
    # config = Config()

    # config_dict = config(os.path.abspath(config_path))
    config_dict = update_config_dict(config_dict)

    # dataset_config_dict = create_dataset_config(os.path.abspath(config_dict["data"]["datasets_file_path"]), config_dict)
    # dataset_config_dict = config(os.path.abspath(config_dict["data"]["datasets_file_path"]))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if not config_dict["training"]["use_cpp"] else "cpu")

    data = {
        "train_loader": DataLoader(
            dataset=DataHandler(config_dict["data"]["datasets_split"]["train_set"], config_dict, device), batch_size=config_dict["data"]["batch_size"], shuffle=True,
            num_workers=config_dict["data"]["num_workers"], drop_last=True, pin_memory=True, collate_fn=custom_collate_fn2),

        "val_loader": DataLoader(
            dataset=DataHandler(config_dict["data"]["datasets_split"]["val_set"], config_dict, device),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            pin_memory=True, collate_fn=custom_collate_fn2)
        # "val_loader": DataLoader(
        #     dataset=DataHandler(config_dict["data"]["datasets_split"]["val_set"], config_dict["data"], device),
        #     batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
        #     pin_memory=True, collate_fn=custom_collate_fn_2)
    }

    model = Model(config_dict)
    # device = torch.device("cpu")

    model.to(device)
    # {config_dict["data"]["metrics"][key]: Metrics_Wrapper(config_dict["data"]["metrics"][key]) for key in
    #  config_dict["data"]["metrics"]}
    criterions = {
        "criterion_train" : Loss_Wrapper(config_dict["loss"]["train_loss"]),
        "criterion_val" : Loss_Wrapper(config_dict["loss"]["val_loss"]),
        "criterion_metrics" : { list(metric_elem_dict.keys())[0]: Metrics_Wrapper(metric_elem_dict) for metric_elem_dict in config_dict["loss"]["metrics"]}
    }


    net_trainer = Net_trainer(model, config_dict["training"]["save_path"], config_dict["training"]["save_freq"], config_dict["training"]["metrics_calc_freq"], config_dict["training"]["num_epochs"], config_dict["optimizer"], config_dict["scheduler"], config_dict["training"]["best_eval_mode"], device, criterions, config_dict=config_dict)


    net_trainer.train(model, device, data)

if __name__ == "__main__":
    import wandb
    arglist = sys.argv[1:]
    if arglist[0] == "--config" and os.path.isfile(arglist[1]):
        config = Config()
        config_dict_base = config(os.path.abspath(arglist[1]))
        # config_dict = create_config_dict(os.path.abspath(config_path))

    arglist_update = arglist[2:]

    arglist_final = []
    # for i in range(len(arglist) - 1):
    #     if not (arglist[i][:2] == "--" and arglist[i+1][:2] == "--"):

    arg_grouping_correct = False
    while(not arg_grouping_correct):
        arglist_tmp = []
        correct_run = True
        for i in range(len(arglist_update) - 1):
            if not (arglist_update[i][:2] == "--" and arglist_update[i+1][:2] == "--"):
                elem_combined = arglist_update[i] + arglist_update[i+1]
                arglist_tmp.append(elem_combined)
                correct_run = False
                arglist_update = arglist_tmp + arglist_update[i+2:]
                break
            else:
                arglist_tmp.append(arglist_update[i])
        if correct_run:
            arg_grouping_correct = True
            # arglist_update = arglist_tmp


    # config_dict_sweep = convert_arg_list2config_dict(sys.argv[2:])

    config_dict_sweep = convert_arg_list2config_dict(config_dict_base, arglist_update)

    config_dict_final = set_unique_save_log_paths(config_dict_sweep, arglist_update)

    train_net(config_dict_sweep)




