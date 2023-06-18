import torch
import argparse
import os
import sys
import json
import cv2
import numpy as np
from shutil import copyfile
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import Config, create_config_dict, update_config_dict, create_visualization_panopticapi
from data_handling import DataHandler, DataHandlerPlainImages, custom_collate_fn, custom_collate_fn2, custom_collate_plain_images
from training import Metrics_Wrapper, EmbeddingHandler, EmbeddingHandlerDummy, Loss_Wrapper, Net_trainer
from models import Model
from torch.utils.data import DataLoader



def predict(visualize, in_path, out_path, weight_file, config_path, copy):
    ##########
    torch.manual_seed(10)
    torch.backends.cudnn.benchmark = True
    # torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)
    ############


    config_dict = create_config_dict(os.path.abspath(config_path))

    use_cpp = config_dict["training"]["use_cpp"]
    use_amp = config_dict["training"]["AMP"]

    if use_cpp:
        amp_device = "cpu"
    else:
        amp_device = "cuda"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(config_dict)
    # device = torch.device("cpu")

    model.to(device)

    model.eval()

    state = torch.load(os.path.abspath(weight_file))

    model.model.load_state_dict(state["model"])

    dataset_segment_info = None

    dataset_split_dict_path = config_dict["data"]["datasets_split"]["train_set"]["sets"]
    set_name = list(dataset_split_dict_path.keys())[0]
    dataset_segment_info_path = dataset_split_dict_path[set_name]["segment_info_file_path"]

    with open(dataset_segment_info_path, 'r') as f:
        dataset_segment_info = json.load(f)

    dataset_category_dict = dataset_segment_info["categories"]
    dataset_category_id_dict = {el['id']: el for el in dataset_category_dict}

    embedding_handler_config = config_dict["training"]["embedding_handler"]
    embedding_handler = EmbeddingHandler(embedding_handler_config["embedding_storage"],
                                         embedding_handler_config["embedding_sampler"],
                                         embedding_handler_config["storage_step_update_sample_size"],
                                         dataset_category_id_dict,
                                         model.model_architecture_embedding_dims, device)
    embedding_handler.load_state_dict(state["embedding_handler"])


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
            dataset=DataHandlerPlainImages(in_path, config_dict["data"]["img_height"], config_dict["data"]["img_width"], config_dict["model"]["channels"], \
                                           device, config_dict["data"]["num_workers"], load_orig_size=True, augmentations_config=augmentations),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            pin_memory=True, collate_fn=custom_collate_plain_images)
    }

    output_annotations = {
        "images": [],
        "annotations": [],
        "categories": dataset_category_dict
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


        for batch_id, datam in enumerate(tqdm(data["pred_loader"], desc="Prediction",  file=sys.stdout)):
            [inputs, file_paths] = datam

            file_name = os.path.basename(file_paths[0])
            fname = file_name.rsplit(".", 1)[0]
            fname = fname.rsplit("_leftImg8bit")[0]

            inputs = inputs.to(device, non_blocking=True)


            # inputs = inputs.permute((0, 3, 2, 1))

            with torch.autocast(device_type=amp_device, enabled=use_amp):
                outputs, output_items = model(inputs)

            annotations_data = [{"image_id": fname}]

            final_outputs, final_output_segmentation_data = model.create_output_from_embeddings(outputs, [dataset_category_id_dict], annotations_data,
                                                                                              embedding_handler=embedding_handler)

            final_output_segmentation_data[0]["file_name"] = fname + "_instanceIds.png"
            output_annotations["images"].append({
                "id": final_output_segmentation_data[0]["image_id"],
                "width": final_outputs.shape[2],
                "height": final_outputs.shape[3],
                "file_name": file_name
            })
            output_annotations["annotations"].append(final_output_segmentation_data[0])
            # test = inputs.cpu().detach().numpy()
            # plt.imshow(test[0])
            # plt.show()

            # inputs = inputs.cpu().detach().numpy()
            final_outputs = final_outputs.cpu().detach().numpy()
            # plt.imshow(final_outputs[0])
            # plt.show()

            save_path = os.path.join(out_path, "pred", final_output_segmentation_data[0]["file_name"])
            # cv2.imwrite(save_path, final_outputs[0])
            final_outputs = np.moveaxis(final_outputs[0], 0, 2)
            # final_outputs = np.moveaxis(final_outputs, 0, 1)
            # final_outputs = final_outputs[0]
            cv2.imwrite(save_path, cv2.cvtColor(final_outputs, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(save_path, final_outputs)
            if copy:
                copyfile(os.path.join(in_path, file_name), os.path.join(pred_path, file_name))

    if visualize:
        vis_imgs_dict = create_visualization_panopticapi(output_annotations, pred_path, in_path)

        for vis_file_name in vis_imgs_dict.keys():
            cv2.imwrite(os.path.join(vis_path, vis_file_name), cv2.cvtColor(vis_imgs_dict[vis_file_name], cv2.COLOR_RGB2BGR))


    with open(os.path.join(out_path, "annotations_data.json"), 'w') as f:
        json.dump(output_annotations, f)

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