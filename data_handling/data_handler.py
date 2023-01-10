from .data_loader import Img_DataLoader, Mask_DataLoader
from .data_loader import Bbox3D_DataLoader
from .augmentations import Augmentation_Wrapper
import torch
import torchvision
import os
import random
import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import cityscapesscripts
from cityscapesscripts.helpers.labels import Label, labels as ct_scripts_label_list
import json
import copy

class DataHandler(torch.utils.data.Dataset):
    def __init__(self, dataset_config_set, general_config, device, *args, **kwargs):
        """ Data Handler class
        Gets path and options to load and handle data in the specified way
        Args:
            data_path: Path to the dataset
            img_height: Height of the image
            img_width: Width of the image
            img_channels: Number of channels in the image
            device: 'cpu' or 'cuda'
            rotation: Max rotation to sample from uniformly
            translation: Max translation to sample from uniformly
            scaling: Max scaling to sample from uniformly
            hor_flip: Wether to randomly horizontally flip
            ver_flip: Wether to randomly vertically flip
            config_data: Configuration dictionary
        """

        self.dataset_config_set = dataset_config_set
        self.dataset_general_settings = dict(general_config["data"])
        self.dataset_general_settings.pop("datasets_split")


        self.dataset_general_settings["device"] = device



        self.augmenter = Augmentation_Wrapper(self.dataset_general_settings["augmentations"])

        self._process_dataset_config(self.dataset_config_set, self.dataset_general_settings)

        # self.img_data_list = []
        # self.bbox_data_list = []
        #
        # self.img_data_list = sorted(self.img_data_list, key=str.lower)
        # self.bbox_data_list = sorted(self.bbox_data_list, key=str.lower)
        # # self.img_data_list = ImgData(img_data_list, img_height, img_width, img_channels)
        # # self.label_data_list = ImgData(label_data_list, img_height, img_width, img_channels)
    def _process_dataset_config(self, dataset_config_sets, dataset_general_settings):
        # List of dataset classes that contain dataset related information
        # for combination of different dataset types (e.g. cityscapes and mapillary vistas) use unify option so that Unify class calculates union of classes and corresponding processes

        dataset_local_settings = dataset_config_sets["settings"]

        dataset_sets = dataset_config_sets["sets"]

        self.dataset_cls_list = []

        if dataset_local_settings["unify"]:
            self.dataset_cls_list = Unified_Dataset(dataset_config_sets, **dataset_general_settings)
        else:
            for key in dataset_sets.keys():
                curr_dataset_type = dataset_sets[key]["dataset_type"]
                if curr_dataset_type == "cityscapes":
                    self.dataset_cls_list.append(Cityscapes_Dataset(**dataset_sets[key], **dataset_local_settings, **dataset_general_settings))
                    # self.dataset_cls_list.append(Cityscapes_Dataset(dataset_config_sets[key], img_width, img_height, dataset_config_settings))
                # if curr_dataset_type == "mapillary_vistas":
                else:
                    raise ValueError("Selected dataset type " + curr_dataset_type + "does not exist!")


        self.dataset_length_list = [len(dataset_cls) for dataset_cls in self.dataset_cls_list]
        # self.dataset_length_list.insert(0, 0)


    def __len__(self):
        """ Returns length of dataset

        Returns:
            Length of dataset
        """
        dataset_length = 0

        for dataset_cls in self.dataset_cls_list:
            dataset_length += len(dataset_cls)

        if dataset_length == 0:
            raise ValueError("Dataset is empty!")

        return dataset_length

    def __getitem__(self, idx):
        """ Gets specific item from dataset with specified index

        Args:
            idx: Index to get data instance

        Returns:
            Data item from dataset with specified index
        """

        dataset_cls_indx = None
        dataset_cls_offset_indx = None

        dataset_length_sum = 0
        for i in range(len(self.dataset_length_list)):
            dataset_length_sum += self.dataset_length_list[i]
            if dataset_length_sum > idx:
                dataset_cls_indx = i
                dataset_cls_offset_indx = idx - dataset_length_sum + self.dataset_length_list[i]


        data_item_dict = self.dataset_cls_list[dataset_cls_indx][dataset_cls_offset_indx]

        augmented_data_item_dict = self.augmenter.apply_augmentation(data_item_dict)

        ###
        max_type_value = np.iinfo(augmented_data_item_dict["img"].dtype).max

        augmented_data_item_dict["img"] = augmented_data_item_dict["img"].astype(np.float32)

        augmented_data_item_dict["img"] /= max_type_value# normalization to uint8
        ###

        return augmented_data_item_dict


class Base_Dataset_COCO(ABC):
    # def __init__(self, dataset_config, img_width, img_height, *args, **kwargs):
    #     self.dataset_format = "COCO"
    #     self.dataset_name = dataset_config["dataset_name"]
    #     self.dataset_type = dataset_config["dataset_type"]
    #     self.img_data_path = dataset_config["img_data_path"]
    #     self.segment_info_file_path = dataset_config["segment_info_file_path"]
    #     self.segment_masks_path = dataset_config["segment_masks_path"]
    #
    #     with open(self.segment_info_file_path, 'r') as f:
    #         self.segment_info = json.load(f)
    #
    #     self.categories = self.segment_info["categories"]
    #     self.categories_id = {el['id']: el for el in self.segment_info['categories']}
    #     self.categories_name = {el['name']: el for el in self.segment_info['categories']}
    #
    #     # self.annotations_data = self.segment_info["annotations"]
    #     self.annotations_data = {el['image_id']: el for el in self.segment_info['annotations']}  #dictionary
    #
    #     # self.license_data = self.segment_info["licenses"]
    #     # self.info_data = self.segment_info["info"]
    #     self.images_meta_data = self.segment_info["images"]   # list
    #
    #     self.img_width = img_width
    #     self.img_height = img_height

    def __init__(self, dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, *args, **kwargs):
        self.dataset_format = "COCO"
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.img_data_path = img_data_path
        self.segment_info_file_path = segment_info_file_path
        self.segment_masks_path = segment_masks_path

        with open(self.segment_info_file_path, 'r') as f:
            self.segment_info = json.load(f)

        self.categories = self.segment_info["categories"]
        self.categories_id = {el['id']: el for el in self.segment_info['categories']}
        self.categories_name = {el['name']: el for el in self.segment_info['categories']}
        self.categories_isthing = {el['id']: el["isthing"] for el in self.segment_info['categories']}

        # self.annotations_data = self.segment_info["annotations"]
        self.annotations_data = {el['image_id']: el for el in self.segment_info['annotations']}  #dictionary

        # self.license_data = self.segment_info["licenses"]
        # self.info_data = self.segment_info["info"]
        self.images_meta_data = self.segment_info["images"]   # list

        self.img_width = img_width
        self.img_height = img_height


    def get_category_by_name(self, category_name):
        return self.categories_name[category_name]


    def get_category_by_id(self, id):
        return self.categories_id[id]

    def get_name_by_id(self, id):
        return self.categories_id[id]["name"]

    def get_id_by_name(self, name):
        return self.categories_name[name]["id"]

    def __len__(self):
        """ Returns length of dataset

        Returns:
            Length of dataset
        """

        return len(self.images_meta_data)

    def __getitem__(self, idx):
        """ Gets specific item from dataset with specified index

        Args:
            idx: Index to get data instance

        Returns:
            Data item from dataset with specified index
        """

        if idx > self.__len__():
             raise ValueError("Index " + idx + " out of range for dataset " + self.dataset_name)

        img_metadata = self.images_meta_data[idx]

        img_id = img_metadata["id"]

        # img_file_name = img_metadata["file_name"]

        img = Img_DataLoader.load_image(os.path.join(self.img_data_path, self.images_meta_data[idx]["file_name"]), self.img_width, self.img_height)

        annotation = self.annotations_data[img_id]

        annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height, self.segment_info["annotations"][idx], self.categories_id)

        data_item_dict = {"img": img,
                          "annotation_mask": annotation_mask,
                          "annotations_data": annotation,
                          "categories_isthing": self.categories_isthing}

        return data_item_dict
        # return img, annotation




class Cityscapes_Dataset(Base_Dataset_COCO):
    def __init__(self, dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, *args, **kwargs):
        super().__init__(dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, *args, **kwargs)
        # self.label_list = ct_scripts_label_list  # special data information from cityscapescripts - do not use in general

    def __getitem__(self, idx):
        """ Gets specific item from dataset with specified index

        Args:
            idx: Index to get data instance

        Returns:
            Data item from dataset with specified index
        """

        if idx > self.__len__():
             raise ValueError("Index " + idx + " out of range for dataset " + self.dataset_name)

        img_metadata = self.images_meta_data[idx]

        img_id = img_metadata["id"]

        img_city_name_dir = img_id.split("_")[0]

        # img_file_name = img_metadata["file_name"]

        img = Img_DataLoader.load_image(os.path.join(self.img_data_path, img_city_name_dir, self.images_meta_data[idx]["file_name"]), self.img_width, self.img_height)

        annotation = self.annotations_data[img_id]

        # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)

        annotation_mask = Mask_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height, self.segment_info["annotations"][idx], self.categories_id)

        annotation.pop("file_name")
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(annotation_mask)
        # plt.show()

        data_item_dict = {"img": img,
                    "annotation_mask": annotation_mask,
                    "annotations_data": annotation,
                    "categories_isthing": self.categories_isthing}

        return data_item_dict

# WIP
class Unified_Dataset():
    def __init__(self, dataset_sets):
        pass

    # @abstractmethod
    # def get_category_by_name(self, category_name):
    #     """ abstract method """
    #
    # @abstractmethod
    # def get_category_by_id(self, id):
    #     """ abstract method """
    #
    # @abstractmethod
    # def get_name_by_id(self, id):
    #     """ abstract method """
    #
    # @abstractmethod
    # def get_id_by_name(self, name):
    #     """ abstract method """