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
from matplotlib import pyplot as plt
from panopticapi.utils import IdGenerator

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
        # print(file_path)
        # if img_load is None:
        #     print("FAIL!")
        #     print(file_path)
        # print(img_load)
        img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2RGB)

        img_data = cv2.resize(img_load, (width, height))

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


        # moved dtype adaption to float to custom_collate_fn due to problems with augmentation using albumentationsa
        # img_data = img_data.astype(np.uint8)

        # img_data = img_data.astype(np.float32)


        # mask_data = np.zeros((img_data.shape[:2]))
        #
        # unique_colors = np.unique(np.reshape(img_data, (-1, 3)), axis=0)
        # for color in unique_colors.tolist():
        #     mask_data[img_data == unique_colors] = color[0] + 256 * color[1] + 256 * 256 * color[2]
        # return img_data, file_name
        return img_data
# deprecated
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

class Mask_DataLoader():
    def __init__(self):
        """
            Image Dataloader class
        """
        pass

    @staticmethod
    def load_image(file_path, width=1024, height=512, segmentation_info=None, categories_id=None):
        """ Loads mask from file path - coco format

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
        if not categories_id or not segmentation_info:
            raise ValueError("No categories given!")
        # img_data = np.zeros((net_input_width, net_input_height, net_input_channels))

        # try:

        img_load = cv2.imread(file_path)

        # plt.imshow(img_load)
        # plt.show()

        img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2RGB)

        # plt.imshow(img_load)
        # plt.show()

        img_load = cv2.resize(img_load, (width, height), interpolation=cv2.INTER_NEAREST) # interpolation method important otherwise interpolation artifacts as new segments !

        # plt.imshow(img_load)
        # plt.show()
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


        # img_data = img_load.astype(np.uint8)

        segments_tmp = {elem["id"]: elem for elem in segmentation_info["segments_info"]}

        img_data = img_load.astype(np.float32)

        # plt.imshow(img_data)
        # plt.show()

        # mask_data = np.zeros((img_data.shape[:2]), dtype=np.float32)
        mask_data = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.float32) # mask_data: HxWx(segment_id, category_id, isthing)

        unique_colors = np.unique(np.reshape(img_data, (-1, 3)), axis=0)[1:] # skip segment_id=0
        for color in unique_colors.tolist():
            # test = img_data[:, :] == color
            # test2 = np.where(img_data == color)
            # test3 = np.all(img_data == color, axis=2)
            # plt.imshow(test3)
            # plt.show()
            color_indx = np.all(img_data == color, axis=2)
            segment_id = color[0] + 256 * color[1] + 256 * 256 * color[2]
            mask_data[color_indx, 0] = segment_id
            cat_id = segments_tmp[segment_id]["category_id"]
            mask_data[color_indx, 1] = cat_id
            mask_data[color_indx, 2] = categories_id[cat_id]["isthing"]

        # plt.imshow(mask_data)
        # plt.show()
        # return img_data, file_name
        # plt.imshow(mask_data[:, :, 0])
        # plt.show()
        return mask_data
# deprecated
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


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def custom_collate_fn(batch_list):
    """
        Combines data from sampled batch to return as torch dataloader batch output
    Args:
        batch_list:

    Returns:
        List with image data as nd array, bbox data as nd array, image file name, bbox file name
    """

    img_data = np.stack([batch_list[i]["img"] for i in range(len(batch_list))], axis=0)

    img_data = np.moveaxis(img_data, 3, 1)

    img_data = img_data.astype(np.float32)


    mask_data = np.stack([batch_list[i]["annotation_mask"] for i in range(len(batch_list))], axis=0)

    mask_data = np.moveaxis(mask_data, 3, 1)

    # mask_data = np.moveaxis(mask_data, 3, 1)

######## for debugging of mask interpolation artifacts

    # plt.imshow(mask_data[0, :, :])
    # plt.show()
    #
    # unique_ids = np.unique(mask_data[0, :, :])
    # for id in unique_ids.tolist():
    #     tmp_vis_img = np.zeros(mask_data[0, :, :].shape)
    #     # test = img_data[:, :] == color
    #     # test2 = np.where(img_data == color)
    #     # test3 = np.all(img_data == color, axis=2)
    #     # plt.imshow(test3)
    #     # plt.show()
    #     tmp_vis_img[mask_data[0, :, :] == id] = 1
    #     plt.imshow(tmp_vis_img)
    #     plt.show()

    # plt.imshow(mask_data)
    # plt.show()
############

    # segments_info_data = [batch_list[i]["annotations_data"].pop("segments_info") for i in range(len(batch_list))]

    segments_row_size = 0
    for i in range(len(batch_list)):
        segments_row_size += len(batch_list[i]["annotations_data"]["segments_info"])

    segments_id_data = np.zeros((segments_row_size, 6)) # (batch_no, segment_id, category_id, isthing, iscrowd, area)

    row_counter = 0
    for i in range(len(batch_list)):
        for j in range(len(batch_list[i]["annotations_data"]["segments_info"])):
            segment_info = batch_list[i]["annotations_data"]["segments_info"][j]
            segments_id_data[row_counter] = np.array([i, segment_info["id"], segment_info["category_id"], batch_list[i]["categories_isthing"][segment_info["category_id"]], segment_info["iscrowd"], segment_info["area"]])
            row_counter += 1

    annotations_data = [batch_list[i]["annotations_data"]["image_id"] for i in range(len(batch_list))]
    # annotations_data = {"image_id": [batch_list[i]["annotations_data"]["image_id"] for i in range(len(batch_list))],
    #              "categories_isthing": [batch_list[i]["categories_isthing"] for i in range(len(batch_list))]}


    return [torch.from_numpy(img_data), torch.from_numpy(mask_data), torch.from_numpy(segments_id_data), annotations_data]

def custom_collate_fn2(batch_list):
    """
        Combines data from sampled batch to return as torch dataloader batch output
    Args:
        batch_list:

    Returns:
        List with image data as nd array, bbox data as nd array, image file name, bbox file name
    """

    img_data = np.stack([batch_list[i]["img"] for i in range(len(batch_list))], axis=0)

    img_data = np.moveaxis(img_data, 3, 1)

    img_data = img_data.astype(np.float32)


    mask_data = np.stack([batch_list[i]["annotation_mask"] for i in range(len(batch_list))], axis=0)

    mask_data = np.moveaxis(mask_data, 3, 1)

    # mask_data = np.moveaxis(mask_data, 3, 1)

######## for debugging of mask interpolation artifacts

    # plt.imshow(mask_data[0, :, :])
    # plt.show()
    #
    # unique_ids = np.unique(mask_data[0, :, :])
    # for id in unique_ids.tolist():
    #     tmp_vis_img = np.zeros(mask_data[0, :, :].shape)
    #     # test = img_data[:, :] == color
    #     # test2 = np.where(img_data == color)
    #     # test3 = np.all(img_data == color, axis=2)
    #     # plt.imshow(test3)
    #     # plt.show()
    #     tmp_vis_img[mask_data[0, :, :] == id] = 1
    #     plt.imshow(tmp_vis_img)
    #     plt.show()

    # plt.imshow(mask_data)
    # plt.show()
############

    # segments_info_data = [batch_list[i]["annotations_data"].pop("segments_info") for i in range(len(batch_list))]

    # segments_row_size = 0
    # for i in range(len(batch_list)):
    #     segments_row_size += len(batch_list[i]["annotations_data"]["segments_info"])
    #
    # segments_id_data = np.zeros((segments_row_size, 6)) # (batch_no, segment_id, category_id, isthing, iscrowd, area)
    #
    # row_counter = 0
    # for i in range(len(batch_list)):
    #     for j in range(len(batch_list[i]["annotations_data"]["segments_info"])):
    #         segment_info = batch_list[i]["annotations_data"]["segments_info"][j]
    #         segments_id_data[row_counter] = np.array([i, segment_info["id"], segment_info["category_id"], batch_list[i]["categories_isthing"][segment_info["category_id"]], segment_info["iscrowd"], segment_info["area"]])
    #         row_counter += 1

    # annotations_data = [batch_list[i]["annotations_data"]["image_id"] for i in range(len(batch_list))]
    # annotations_data = {"image_id": [batch_list[i]["annotations_data"]["image_id"] for i in range(len(batch_list))],
    #                     "segments_info": [batch_list[i]["annotations_data"]["segments_info"] for i in range(len(batch_list))],
    #                     "categories_isthing": [batch_list[i]["categories_isthing"] for i in range(len(batch_list))]}

    annotations_data = []

    for i in range(len(batch_list)):
        batch_dict_temp = {"image_id": batch_list[i]["annotations_data"]["image_id"],
                           "segments_info": batch_list[i]["annotations_data"]["segments_info"],
                           "categories_isthing": batch_list[i]["categories_isthing"]
                           }
        annotations_data.append(batch_dict_temp)


    return [torch.from_numpy(img_data), torch.from_numpy(mask_data), annotations_data]

def custom_collate_plain_images(batch_list):
    """
        Combines data from sampled batch to return as torch dataloader batch output
    Args:
        batch_list:

    Returns:
        List with image data as nd array, bbox data as nd array, image file name, bbox file name
    """

    img_data = np.stack([batch_list[i][0] for i in range(len(batch_list))], axis=0)

    img_data = np.moveaxis(img_data, 3, 1)

    img_data = img_data.astype(np.float32)

    file_path_list = [batch_list[i][1] for i in range(len(batch_list))]


    return [torch.from_numpy(img_data), file_path_list]