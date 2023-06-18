import torch
import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from utils import Config, create_config_dict, update_config_dict
from data_handling import DataHandler, DataHandlerPlainImages, custom_collate_fn, custom_collate_fn2, custom_collate_plain_images
from torch.utils.data import DataLoader


def dataloader_train_test(config_path, epochs, output_vis_path):

    config_dict = create_config_dict(os.path.abspath(config_path))

    device = "cpu"

    torch.manual_seed(10)

    os.makedirs(output_vis_path, exist_ok=True)
    os.makedirs(os.path.join(output_vis_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_vis_path, "val"), exist_ok=True)

    data = {
        "train_loader": DataLoader(
            dataset=DataHandler(config_dict["data"]["datasets_split"]["train_set"], config_dict, device),
            batch_size=1, shuffle=False,
            num_workers=config_dict["data"]["num_workers"], drop_last=True,
            collate_fn=custom_collate_fn2),

        "val_loader": DataLoader(
            dataset=DataHandler(config_dict["data"]["datasets_split"]["val_set"], config_dict, device),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            collate_fn=custom_collate_fn2)
    }
    for epoch in range(0, epochs + 1):
        tqdm.write("Epoch " + str(epoch) + ":")
        tqdm.write("-" * 70)
        for batch_id, datam in enumerate(tqdm(data["train_loader"], desc="Dataloader Test - Train", file=sys.stdout)):
            [inputs, masks, annotations_data] = datam

            inputs = np.moveaxis(inputs.numpy()[0], 0, -1) * 255

            inputs = inputs.astype(np.uint8)

            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)


            cv2.imwrite(os.path.join(output_vis_path, "train", annotations_data[0]["image_id"] + "_" + str(epoch) + ".png"), inputs)


        for batch_id, datam in enumerate(tqdm(data["val_loader"], desc="Dataloader Test - Val", file=sys.stdout)):
            [inputs, masks, annotations_data] = datam

            inputs = np.moveaxis(inputs.numpy()[0], 0, -1) * 255

            inputs = inputs.astype(np.uint8)

            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)

            cv2.imwrite(os.path.join(output_vis_path, "val", annotations_data[0]["image_id"] + "_" + str(epoch) + ".png"), inputs)

def dataloader_plainimages_test(config_path, input_path, output_vis_path):

    config_dict = create_config_dict(os.path.abspath(config_path))

    device = "cpu"

    torch.manual_seed(10)

    os.makedirs(os.path.join(output_vis_path, "plainimagesloader"), exist_ok=True)

    augmentations = None
    for key in config_dict["data"]["augmentations"]["transformations"].keys():
        if "crop" in key:
            augmentations = dict(config_dict["data"]["augmentations"])
            augmentations["transformations"] = {key: augmentations["transformations"][key]}

    data = {
        # "pred_loader": DataLoader(
        #     dataset=DataHandlerPlainImages(in_path, config_dict["data"]["img_height"], config_dict["data"]["img_width"],
        #                                    config_dict["model"]["channels"], \
        #                                    device, config_dict["data"]["num_workers"]),
        #     batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
        #     pin_memory=True)
        "pred_loader": DataLoader(
            dataset=DataHandlerPlainImages(input_path, config_dict["data"]["img_height"], config_dict["data"]["img_width"],
                                           config_dict["model"]["channels"], \
                                           device, config_dict["data"]["num_workers"], load_orig_size=True,
                                           augmentations_config=augmentations),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            collate_fn=custom_collate_plain_images)
    }


    for batch_id, datam in enumerate(tqdm(data["pred_loader"], desc="Dataloader PlainImages Test - Val", file=sys.stdout)):
        [inputs, file_paths] = datam

        file_name = os.path.basename(file_paths[0])
        fname = file_name.rsplit(".", 1)[0]
        fname = fname.rsplit("_leftImg8bit")[0]

        inputs = np.moveaxis(inputs.numpy()[0], 0, -1) * 255

        inputs = inputs.astype(np.uint8)

        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(output_vis_path, "plainimagesloader", fname + ".png"), inputs)

def dataloader_plainimages_singleimg_testaugmentations(config_path, input_path_plain_img, num_augm, output_vis_path):

    config_dict = create_config_dict(os.path.abspath(config_path))

    device = "cpu"

    torch.manual_seed(10)

    os.makedirs(os.path.join(output_vis_path, "augmentations_single_img"), exist_ok=True)


    augmentations = None
    for key in config_dict["data"]["augmentations"]["transformations"].keys():
        if "crop" in key:
            augmentations = dict(config_dict["data"]["augmentations"])
            augmentations["transformations"] = {key: augmentations["transformations"][key]}

    data = {
        # "pred_loader": DataLoader(
        #     dataset=DataHandlerPlainImages(in_path, config_dict["data"]["img_height"], config_dict["data"]["img_width"],
        #                                    config_dict["model"]["channels"], \
        #                                    device, config_dict["data"]["num_workers"]),
        #     batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
        #     pin_memory=True)
        "pred_loader": DataLoader(
            dataset=DataHandlerPlainImages(input_path_plain_img, config_dict["data"]["img_height"],
                                           config_dict["data"]["img_width"],
                                           config_dict["model"]["channels"], \
                                           device, config_dict["data"]["num_workers"], load_orig_size=True,
                                           augmentations_config=augmentations),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            collate_fn=custom_collate_plain_images)
    }

    for i in range(num_augm):
        for batch_id, datam in enumerate(
                tqdm(data["pred_loader"], desc="Dataloader PlainImages - Single Image Augmentations", file=sys.stdout)):
            [inputs, file_paths] = datam

            file_name = os.path.basename(file_paths[0])
            fname = file_name.rsplit(".", 1)[0]
            fname = fname.rsplit("_leftImg8bit")[0]

            inputs = np.moveaxis(inputs.numpy()[0], 0, -1) * 255

            inputs = inputs.astype(np.uint8)

            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)

            cv2.imwrite(os.path.join(output_vis_path, "augmentations_single_img", fname + str(i) + ".png"), inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("-e", "--epochs", type=int,
                        help="Amount of epochs")
    parser.add_argument("-n", "--num_augmentations", type=int,
                        help="Amount of augmentations")
    parser.add_argument("-d", "--in_plainimgloader_path", type=str,
                        help="Input path to PlainImageDataloader")
    parser.add_argument("-a", "--in_plainimgloader_single_path", type=str,
                        help="Input path to single img for augmentations test")
    parser.add_argument("-s", "--output_save_path", type=str,
                        help="Save path to visualize output")
    args = parser.parse_args()

    # dataloader_train_test(args.config, args.epochs, os.path.join(args.output_save_path, "trainloader"))

    dataloader_plainimages_test(args.config, args.in_plainimgloader_path, os.path.join(args.output_save_path, "plainimagesloader"))

    # dataloader_plainimages_singleimg_testaugmentations(args.config, args.in_plainimgloader_single_path, args.num_augmentations, os.path.join(args.output_save_path, "augmentations"))