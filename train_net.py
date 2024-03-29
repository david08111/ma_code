import torch
import argparse
import os
# import rmm, pprint
import cudf
import cuml
from utils import Config, create_config_dict, update_config_dict
from data_handling import DataHandler, custom_collate_fn, custom_collate_fn2
from training import Net_trainer
from training import Loss_Wrapper
from training import Metrics_Wrapper
from models import Model
from torch.utils.data import DataLoader


def train_net(config_path, verbose):
    # torch.manual_seed(10)

    # torch.backends.cudnn.benchmark = True

    # config = Config()



    # config_dict = config(os.path.abspath(config_path))
    config_dict = create_config_dict(os.path.abspath(config_path))

    # config_dict["training"]["num_epochs"] = 1

    # dataset_config_dict = create_dataset_config(os.path.abspath(config_dict["data"]["datasets_file_path"]), config_dict)
    # dataset_config_dict = config(os.path.abspath(config_dict["data"]["datasets_file_path"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if not config_dict["training"]["use_cpp"] else "cpu")

    # rmm.reinitialize(pool_allocator=True,
    #                  initial_pool_size=config_dict["training"]["cuml_mem_alloc"],
    #                  maximum_pool_size=6e9)
    # mr = rmm.mr.get_current_device_resource()
    # stats_pool_memory_resource = rmm.mr.StatisticsResourceAdaptor(mr)
    # rmm.mr.set_current_device_resource(stats_pool_memory_resource)

    torch.manual_seed(10)
    torch.backends.cudnn.benchmark = True
    # torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)

    # from mmseg.apis import inference_model, init_model, show_result_pyplot
    # # import mmcv
    #
    # config_file = '/work/scratch/dziuba/repos/ma_code/models/mmsegmentation/configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py'
    # # checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    #
    # # build the model from a config file and a checkpoint file
    # model = init_model(config_file, device='cuda')



    data = {
        "train_loader": DataLoader(
            dataset=DataHandler(config_dict["data"]["datasets_split"]["train_set"], config_dict, device),
            batch_size=config_dict["data"]["batch_size"], shuffle=True,
            num_workers=config_dict["data"]["num_workers"], drop_last=True, pin_memory=True,
            collate_fn=custom_collate_fn2),

        "val_loader": DataLoader(
            dataset=DataHandler(config_dict["data"]["datasets_split"]["val_set"], config_dict, device),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            pin_memory=True, collate_fn=custom_collate_fn2)
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


    net_trainer = Net_trainer(model, config_dict["training"]["save_path"], config_dict["training"]["save_freq"], config_dict["training"]["metrics_calc_freq"], config_dict["training"]["num_epochs"], config_dict["optimizer"], config_dict["scheduler"], config_dict["training"]["best_eval_mode"], device, criterions, config_dict["training"]["AMP"], config_dict=config_dict)


    net_trainer.train(model, device, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    train_net(args.config, args.verbose)

    # import cProfile
    # config_dict = args.config
    #
    # cProfile.run('train_net(config_dict, args.verbose)')