import argparse
import torch
from torch.utils.data import DataLoader
from utils import Config, load_cam_settings, visualize_3d_bboxes
from Networks import Net_Wrapper
import os
import time
from tqdm import tqdm
import cv2
from shutil import copyfile
import numpy as np
from matplotlib import pyplot as plt
from data_handling import DataHandler, Bbox3D_DataLoader, custom_collate_fn_2
from utils import Evaluation_Logger



def predict(visualize, in_path, out_path, weight_file, config_file, copy):

    config = Config()
    config_dict = config(os.path.abspath(config_file))

    cam_config = load_cam_settings(config_dict["data"]["camera_config"])

    net = Net_Wrapper(config_dict["network"]["architecture"], config_dict["network"]["in_channels"], config_dict["network"]["classes"], config_dict["data"]["img_size"], config_dict["network"]["architecture_config"])

    net.eval()

    state = torch.load(os.path.abspath(weight_file))

    try:
        net.model.load_state_dict(state["state_dict"]["model"])
    except KeyError:
        try:
            net.model.load_state_dict(state["model"])
        except KeyError:
            net.model.load_state_dict(state)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net.to(device)

    data = {
        "pred_loader": DataLoader(
            dataset=DataHandler(in_path, config_dict["data"]["img_size"],
                                config_dict["data"]["img_size"], config_dict["network"]["in_channels"], device,
                                rotation=0, translation=0, scaling=0, hor_flip=False, ver_flip=False, config_data=config_dict["data"]),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            pin_memory=True, collate_fn=custom_collate_fn_2)
    }


    with torch.no_grad():

        if visualize:
            pred_path = os.path.join(os.path.abspath(out_path), "pred")
            vis_path = os.path.join(os.path.abspath(out_path),
                                    "visualization")
            os.makedirs(vis_path, exist_ok=True)
        else:
            pred_path = os.path.abspath(out_path)

        os.makedirs(pred_path, exist_ok=True)


        for batch_id, datam in enumerate(tqdm(data["pred_loader"], desc="Predict")):
            [inputs, labels, input_file_name, label_file_name, crop_pos_list, cam_extr_list, img_list_full] = datam

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # labels = labels.narrow(3, 0, 1).contiguous()
            inputs = inputs.permute((0, 3, 2, 1))

            outputs = net.model(inputs)

            outputs = net.create_final_outputs(outputs, cam_config["fx"], inputs.shape[2])

            # inputs = inputs.permute((0, 3, 2, 1))

            fname = input_file_name[0].rsplit(".", 1)[0]

            # test = inputs.cpu().detach().numpy()
            # plt.imshow(test[0])
            # plt.show()

            inputs = inputs.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            # crop_resize_factor = crop_resize_factor.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            labels = labels[:, 1:]


            if visualize:
                # vis_img = np.maximum(inputs.cpu().numpy(), labels.cpu().numpy())
                # vis_img[vis_img == vis_img] = 128
                # vis_img = np.maximum(vis_img, outputs.numpy())
                # cv2.imwrite(os.path.join(vis_path, input_file_name), vis_img)

                # inputs = inputs.cpu().numpy()[0, :, :, :] * 255
                # inputs = inputs.astype(np.uint8)

                # outputs = np.concatenate([outputs[0, :, :, :], outputs[0, :, :, :], outputs[0, :, :, :]], axis=2)
                # outputs = outputs.astype(np.uint8)

                visualized_img = visualize_3d_bboxes(inputs, outputs, labels, crop_pos_list, cam_config, cam_extr_list, img_list_full)
                visualized_img *= 255
                # overlayed_img = np.array(overlayed_img, dtype=np.uint8)
                cv2.imwrite(os.path.join(vis_path, input_file_name[0]), cv2.cvtColor(visualized_img[0], cv2.COLOR_RGB2BGR))
            if copy:
                copyfile(os.path.join(in_path, input_file_name[0]), os.path.join(pred_path, input_file_name[0]))
                copyfile(os.path.join(in_path, label_file_name[0]), os.path.join(pred_path, label_file_name[0]))
                # cv2.imwrite(os.path.join(pred_path, input_file_name[0]), inputs[0])
                # cv2.imwrite(os.path.join(pred_path, label_file_name[0]), labels)

            Bbox3D_DataLoader.save_bbox(os.path.join(pred_path, input_file_name[0] + "_pred" + "." + "json"), outputs, cam_config)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="If Flag is specified, results will be plotted")
    parser.add_argument("-d", "--in_path", type=str, help="Input Data Dir")
    parser.add_argument("-s", "--out_path", default="./outputs", type=str,
                        help="Output Data Dir")
    parser.add_argument("-w", "--weight_file", type=str, help="Model Weights")
    # parser.add_argument("-t", "--threshold", type=str, help="threshold for classification")
    parser.add_argument("-c", "--config_file", type=str, help="Configuration")
    parser.add_argument("-cp", "--copy", action="store_true",
                        help="Select if images get copied")

    args = parser.parse_args()

    predict(args.visualize, args.in_path, args.out_path, args.weight_file, args.config_file, args.copy)