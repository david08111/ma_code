from .data_handler import *
import cv2
import csv
import numpy as np
import os
import json
import torch.nn.functional as F
import cv2
import csv
import numpy as np
import torch

class Img_DataLoader():
    def __init__(self):
        """
            Image Dataloader class
        """
        pass

    @staticmethod
    def load_image(file_path, width=1024, height=512, channels=3):
        """ Loads image from path

        Args:
            file_path: Filepath to load image from
            width: Input image width to resize img from
            height: Input image width to resize img from
            channels: Number of image input channels

        Returns:
            Image as nd array, file name of image
        """

        if not os.path.isfile(file_path):
            raise NameError("File " + file_path + " does not exist!")

        # img_data = np.zeros((net_input_width, net_input_height, net_input_channels))

        # try:

        img_load = cv2.imread(file_path)
        img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2RGB)

        # aug_crop_aspect_ratio = iaa.CenterPadToAspectRatio(1)
        # img_data = aug_crop_aspect_ratio(images=[img_load])[0]



        # if img_load.shape[0] > img_load.shape[1]:
        #     raise str("Width/Height ratio < 0 for file " + file_path)

        # img_data = cv2.resize(cv2.imread(file_path), (net_input_width, net_input_height))
        # plt.imshow(img_data)
        # plt.show()
        # except:
        #     print(file_path)
        # file_name = os.path.basename(file_path)


        img_data = img_load.astype(np.uint8)

        # return img_data, file_name
        return img_data

    @staticmethod
    def load_images(file_list, width=256, height=256, channels=3):
        """ Loads multiple images from path

        Args:
            file_list: List with file paths to load images from
            width: Input image width to resize img from
            height: Input image width to resize img from
            channels: Number of image input channels

        Returns:
            Images as nd array, list containing according file names
        """
        # if os.path.isdir(img_data_path):
        #     file_list = os.listdir(img_data_path)
        #     number_files = len(file_list)
        # elif os.path.isfile(img_data_path):
        #     file_list = img_data_path
        #     number_files = len(file_list)
        # else:
        #     raise Exception("Dir path or file does not exist!")
        number_files = len(file_list)


        img_data = np.zeros((number_files, width, height, channels))
        file_names = []
        file_paths = []

        # if os.path.isdir(img_data_path):
        #     for root, dirs, files in os.walk(img_data_path):
        counter = 0
        for file in file_list:
            # file_img = cv2.imread(file)
            try:
                img_data[counter] = cv2.resize(cv2.imread(file), (width, height))
            except:
                print(file)
            file_names.append(os.path.basename(file))
            # file_paths.append(os.path.join(img_data_path, file))
            counter += 1
        # elif os.path.isfile(img_data_path):
        #     file_img = cv2.imread(img_data_path)
        #     img_data[0] = cv2.resize(file_img, (net_input_width, net_input_height))
        #     file_names.append(os.path.basename(img_data_path))

        img_data = img_data.astype(np.uint8)

        return img_data, file_names


class Bbox3D_DataLoader():
    """
    Bbox Dataloader class
    """
    @staticmethod
    def load_bbox(file_path):
        """
        Loads bbox data from specified file path
        Args:
            file_path: Path to the bbox data

        Returns:
            Nd array containing the bbox data, file name
        """
        if not os.path.isfile(file_path):
            raise NameError("File " + file_path + " does not exist!")

        file_name = os.path.basename(file_path)

        if "json" in file_name:
            return Bbox3D_DataLoader.load_bbox_json(file_path)
        else:
            assert ("No conform bbox data found!")
        # return torch.FloatTensor(json_data), file_name

    ##### load json files
    @staticmethod
    def load_bbox_json(file_path):
        """
            Loads bbox data from specified file path - json file
        Args:
            file_path: Path to the bbox data

        Returns:
            Nd array containing the bbox data, file name
        """
        if not os.path.isfile(file_path):
            raise NameError("File " + file_path + " does not exist!")

        file_name = os.path.basename(file_path)

        with open(file_path, "r") as file_handler:
            json_data = json.load(file_handler)

        # add dimension to front for batch recognition
        # for elem in json_data:
        #     elem.insert(0, 0)
        #
        # print("list")
        # print(json_data)

        if not json_data["object_data_list"]:
            raise("No object data in data instance: " + file_name)

        bbox_data, roi_crop_list = Bbox3D_DataLoader._load_crops(json_data)




        return bbox_data, roi_crop_list, json_data["camera_extrinsic"], file_name
        # return torch.FloatTensor(json_data), file_name

    @staticmethod
    def _load_crops(json_data):
        obj_indx = 0

        bbox_data = np.zeros((len(json_data["object_data_list"]), 7), dtype=np.float32)
        # bbox_data = np.zeros((1, 11), dtype=np.float32)

        roi_crop_list = []

        for elem in json_data["object_data_list"]:
            # obj_data_temp = np.zeros((1, 7), dtype=np.float32)
            bbox_data[obj_indx, 0] = elem["rel_camera_depth"]
            bbox_data[obj_indx, 1] = elem["rel_yaw"]
            bbox_data[obj_indx, 2] = elem["3d_bbox_mean_pt_projected"][0] - (max(elem["2d_bbox_mean_pt"][0] - 0.5 * elem["2d_bbox_dim"][0], 0)) # only for croppped images   --- rescale due to later resize?
            bbox_data[obj_indx, 3] = elem["3d_bbox_mean_pt_projected"][1] - (max(elem["2d_bbox_mean_pt"][1] - 0.5 * elem["2d_bbox_dim"][1], 0))
            bbox_data[obj_indx, 4] = elem["3d_bbox_dim"][0]
            bbox_data[obj_indx, 5] = elem["3d_bbox_dim"][1] # in blender coords/dimensions
            bbox_data[obj_indx, 6] = elem["3d_bbox_dim"][2]

            roi_crop_list.append({"obj_indx": obj_indx,
                                  "2d_bbox_mean_pt": elem["2d_bbox_mean_pt"],
                                  "2d_bbox_dim": elem["2d_bbox_dim"]})

            obj_indx += 1

        return bbox_data, roi_crop_list

    @staticmethod
    def load_bboxes(file_list):
        """
            Load multiple bbox data files from specified file path list
        Args:
            file_path: Path to the bbox data

        Returns:
            Nd array containing the bbox data
                """
        bbox_data_list = []
        file_name_list = []

        for file in file_list:
            bbox_data, file_name = Bbox3D_DataLoader.load_bbox(file)

            bbox_data_list.append(bbox_data)
            file_name_list.append(file_name)

        return bbox_data_list, file_name_list

    @staticmethod
    def save_bbox(file_path, bbox_data, camera_config):
        """
            Saves specified bbox data to given file path
        Args:
            file_path: File path to save bbox data to
            bbox_data: Ndarray containing bbox data
        """
        file_name = os.path.basename(file_path)

        if "json" in file_name:
            return Bbox3D_DataLoader.save_bbox_json(file_path, bbox_data, camera_config)
        else:
            assert ("No conform bbox data format!")

    @staticmethod
    def save_bbox_json(file_path, bbox_data, camera_config):
        """
            Saves specified bbox data to given file path - json file
        Args:
            file_path: File path to save bbox data to
            bbox_data: Ndarray containing bbox data
        """

        final_gt_dict = {}
        bbox_data = bbox_data.astype(np.float64)
        final_gt_dict["camera_intrinsic"]  = camera_config["camera_intrinsic"].tolist()
        final_gt_dict["object_data_list"] = {"3d_bbox_dim": [bbox_data[0, 4], bbox_data[0, 5], bbox_data[0, 6]],
                              "3d_bbox_mean_pt_projected": [bbox_data[0, 2], bbox_data[0, 3]],
                              "rel_yaw": bbox_data[0, 1],
                              "rel_camera_depth": [bbox_data[0, 0]]}

        with open(file_path, "w") as file_handler:
            json.dump(final_gt_dict, file_handler)


def custom_collate_fn(batch_list):
    """
        Combines data from sampled batch to return as torch dataloader batch output
    Args:
        batch_list:

    Returns:
        List with image data as nd array, bbox data as nd array, image file name, bbox file name
    """
    img_file_names = [batch_list[i][2] for i in range(len(batch_list))]
    bbox_file_names = [batch_list[i][3] for i in range(len(batch_list))]

    img_data = np.stack([batch_list[i][0] for i in range(len(batch_list))], axis=0)

    max_obj_nr = max([batch_list[i][1].shape[0] for i in range(len(batch_list))])


    # bbox_data = np.stack([batch_list[i][1] for i in range(len(batch_list))], axis=0)

    bbox_data = -1 * np.ones((len(batch_list), max_obj_nr, 5), dtype=np.float32)

    for i in range(len(batch_list)):
        bbox_data[i,:batch_list[i][1].shape[0],:] = batch_list[i][1]

    # bbox_data = np.rollaxis(bbox_data, 2, 0)

    return [torch.from_numpy(img_data), torch.from_numpy(bbox_data), img_file_names, bbox_file_names]


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def custom_collate_fn_2(batch_list):
    """
        Combines data from sampled batch to return as torch dataloader batch output
    Args:
        batch_list:

    Returns:
        List with image data as nd array, bbox data as nd array, image file name, bbox file name
    """
    # Drop invalid images
    # batch = [data for data in batch_list if data is not None]
    batch_size = len(batch_list)
    img_crops = np.concatenate([elem[0] for elem in batch_list])
    bb_targets = np.concatenate([elem[1] for elem in batch_list])
    cam_extr_list = [batch_list[i][5] for i in range(len(batch_list)) for j in range(batch_list[i][1].shape[0])]
    # cam_rot_euler_list = [batch_list[i][6] for i in range(len(batch_list)) for j in range(batch_list[i][1].shape[0])]
    img_data_full_list = [batch_list[i][6] for i in range(len(batch_list)) for j in range(batch_list[i][1].shape[0])]
    img_labels = []
    bbox_labels = []
    crop_pos_list = np.concatenate([elem[4] for elem in batch_list])
    for elem in batch_list:
        for i in range(elem[1].shape[0]):
            img_labels.append(elem[2])
            bbox_labels.append(elem[3])

    bb_targets = np.insert(bb_targets, 0, 0, axis=1)
    for i in range(bb_targets.shape[0]):
        bb_targets[i, 0] = i


    return [torch.from_numpy(img_crops[:batch_size]), torch.from_numpy(bb_targets[:batch_size]), img_labels[:batch_size], bbox_labels[:batch_size], crop_pos_list[:batch_size], cam_extr_list[:batch_size], img_data_full_list[:batch_size]]