from .data_loader import Img_DataLoader
from .data_loader import Bbox3D_DataLoader
from .augmentations import Augmenter
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



        self.augmenter = Augmenter(self.dataset_general_settings["augmentations"])

        self._process_dataset_config(self.dataset_config_set, self.dataset_general_settings)

        # self.img_data_list = []
        # self.bbox_data_list = []
        #
        # for elem in os.listdir(data_path):
        #     if os.path.isfile(os.path.join(data_path, elem)):
        #         if not config_data["label_ext"] in elem:
        #            self.img_data_list.append(os.path.join(data_path, elem))
        #         else:
        #            self.bbox_data_list.append(os.path.join(data_path, elem))
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
        for i in range(len(self.dataset_length_list)-1):
            if self.dataset_length_list[i] > dataset_length_sum and self.dataset_length_list[i+1] < dataset_length_sum:
                dataset_cls_indx = i
                dataset_cls_offset_indx = idx - dataset_length_sum - 1


        data_item_dict = self.dataset_cls_list[dataset_cls_indx][dataset_cls_offset_indx]

        augmented_data_item_dict = self.augmenter.apply_augmentation(data_item_dict)

        plt.imshow(augmented_data_item_dict["img"])
        plt.show()
        plt.imshow(augmented_data_item_dict["annotation"])
        plt.show()

        return data_item_dict

        # img_data, img_file_name = Img_DataLoader.load_image(self.img_data_list[idx], self.img_width, self.img_heigth, self.img_channels)
        #
        # # plt.imshow(img_data)
        # # plt.show()
        #
        # bbox_data, roi_crop_list, cam_extr, bbox_file_name = Bbox3D_DataLoader.load_bbox(self.bbox_data_list[idx])
        #
        # cam_extr = np.array(cam_extr)
        #
        # cropped_img_list = []
        # crop_resize_factor = []
        #
        # crop_pos_list = []
        #
        # for i in range(len(roi_crop_list)):
        #     crop_dict = roi_crop_list[i]
        #     mid_pt = crop_dict["2d_bbox_mean_pt"]
        #     box_dim = crop_dict["2d_bbox_dim"]
        #     left_border = min(max(img_data.shape[1] - int(mid_pt[0] + 0.5*box_dim[0]), 0), img_data.shape[0])
        #     right_border = min(max(img_data.shape[1] - int(mid_pt[0] - 0.5 * box_dim[0]), 0), img_data.shape[0])
        #     # left_border = min(max(int(mid_pt[0] - 0.5 * box_dim[0]), 0), img_data.shape[0])
        #     # right_border = min(max(int(mid_pt[0] + 0.5 * box_dim[0]), 0), img_data.shape[0])
        #     top_border = min(max(int(mid_pt[1]-0.5 * box_dim[1]), 0), img_data.shape[1])#low in value
        #     bottom_border = min(max(int(mid_pt[1] + 0.5 * box_dim[1]), 0), img_data.shape[1]) #high in value
        #     # cropped_img = img_data[int(mid_pt[1]-0.5*box_dim[1]):int(mid_pt[1]+0.5*box_dim[1]), int(self.img_width - mid_pt[0]-0.5*box_dim[0]):int(self.img_width - mid_pt[0]+0.5*box_dim[0]), :]
        #     crop_pos_list.append([left_border, right_border, top_border, bottom_border])
        #     cropped_img = img_data[top_border:bottom_border, left_border:right_border, :]
        #
        #     # crop_resize_factor.append(np.array([cropped_img.shape[0] / self.img_width, cropped_img.shape[1] / self.img_heigth]))
        #
        #     # plt.imshow(cropped_img)
        #     # plt.show()
        #
        #     bbox_data[i, 2] = self.img_width * (right_border - left_border - bbox_data[i, 2]) / cropped_img.shape[1]
        #     bbox_data[i, 3] = self.img_heigth * bbox_data[i, 3] / cropped_img.shape[0]
        #
        #     # crop_resize_factor.append(
        #     #     np.array([cropped_img.shape[0] / img_data.shape[0], cropped_img.shape[1] / img_data.shape[1]]))
        #
        #     # plt.imshow(cropped_img)
        #     # plt.show()
        #
        #     # bbox_data[i, 2] = img_data.shape[1] * bbox_data[i, 2] / cropped_img.shape[1]
        #     # bbox_data[i, 3] = img_data.shape[0] * bbox_data[i, 3] / cropped_img.shape[0]
        #
        #     cropped_img = cv2.resize(cropped_img, (self.img_width, self.img_heigth))
        #
        #     # plt.imshow(cropped_img)
        #     # plt.show()
        #
        #     cropped_img_list.append(cropped_img)
        #
        #
        # cropped_img_data = np.asarray(cropped_img_list)
        # bbox_data = np.asarray(bbox_data)
        # image_data_full = np.asarray(img_data)
        #
        # cropped_img_data = cropped_img_data.astype(np.float32)
        # bbox_data = bbox_data.astype(np.float32)
        # image_data_full = image_data_full.astype(np.float32)
        #
        #
        # cropped_img_data = cropped_img_data / 255
        # image_data_full = image_data_full / 255
        #
        #
        # return [cropped_img_data, bbox_data, img_file_name, bbox_file_name, crop_pos_list, cam_extr, image_data_full]

 ## deprecated
    def check_bbox_data(self, bbox_data):
        """ Checks wether bbox data is in reasonable range (from 0 to 1) for x,y,w,h

        Args:
            bbox_data:

        Returns:
            Wether bbox data is in reasonable range
        """
        # test = bbox_data.min()
        if bbox_data[:, 1:5].min() >= 0 and bbox_data[:, 1:5].max() <= 1:
            return True
        else:
            return False

## deprecated
    def augment_img(self, img, transform_dict):
        """ Augments image data with specified transform dict

        Args:
            img: Image data
            transform_dict: Dictionary containing transformation data

        Returns:
            Augmented image data
        """
        img = torchvision.transforms.functional.to_pil_image(img)
        if "rotation" in transform_dict.keys() or "translation" in transform_dict.keys() or "scaling" in transform_dict.keys():
            img = torchvision.transforms.functional.affine(img, transform_dict["rotation"], transform_dict["translation"], transform_dict["scaling"], 0)

        if "horizontal_flip" in transform_dict.keys():
            img = torchvision.transforms.functional.hflip(img)

        if "vertical_flip" in transform_dict.keys():
            img = torchvision.transforms.functional.vflip(img)

        return torchvision.transforms.functional.center_crop(img, (self.img_heigth, self.img_width))

## deprecated
    def augment_bbox(self, bbox, transform_dict):
        """ Augments bbox data with specified transform dict

        Args:
            bbox: Bbox data
            transform_dict: Dictionary containing transformation data

        Returns:
            Augmented bbox data
        """
        for i in range(bbox.shape[0]):
            bbox_coord = np.ones(3)
            bbox_coord[0:2] = bbox[i,1:3]

            if "rotation" in transform_dict.keys() or "translation" in transform_dict.keys() or "scaling" in transform_dict.keys():
                transform_matr = np.array([[np.cos((np.pi * int(transform_dict["rotation"]) / 180)), -np.sin((np.pi * int(transform_dict["rotation"]) / 180)), float(transform_dict["translation"][0])], [np.sin((np.pi * int(transform_dict["rotation"]) / 180)), np.cos((np.pi * int(transform_dict["rotation"]) / 180)), float(transform_dict["translation"][1])],[0,0,1]])
                bbox_coord = np.matmul(transform_matr, bbox_coord)

            if "vertical_flip" in transform_dict.keys():
                bbox_coord[1] = 1 - bbox_coord[1]

            if "horizontal_flip" in transform_dict.keys():
                bbox_coord[0] = 1 - bbox_coord[0]

            # if "scaling" in transform_dict.keys():
            #     bbox_hw = bbox[i,3:5]
            #
            #     transform_matr_scale = np.array([[float(transform_dict["scaling"]), 0], [0, float(transform_dict["scaling"])]])
            #     bbox_hw = np.matmul(transform_matr_scale, bbox_hw)
            #     # bbox_coord = np.matmul(transform_matr_scale, bbox_coord[0:2])   ###
            #
            #     # bbox[i,1:3] = bbox_coord[0:2]
            #     bbox[i,3:5] = bbox_hw

            bbox[i,1:3] = bbox_coord[0:2]


        return bbox

## deprecated
    def create_transform(self, rotation, translation, scaling, hor_flip, ver_flip):
        """ Creates a transform ditionary from given parameters

        Args:
            rotation: Max rotation to sample from uniformly
            translation: Max translation to sample from uniformly
            scaling: Max scaling to sample from uniformly
            hor_flip: Wether to randomly horizontally flip
            ver_flip: Wether to randomly vertically flip

        Returns:
            Dictionary containing transformations
        """
        transform_dict = {}
        if rotation or translation or scaling or hor_flip or ver_flip:
            if rotation or translation or scaling:
                if not rotation:
                    transform_dict["rotation"] = 0
                else:
                    transform_dict["rotation"] = random.randint(0, rotation)
                if translation:
                    rdm_transl = random.uniform(0, translation / self.img_width)
                    transform_dict["translation"] = (rdm_transl, rdm_transl)
                else:
                    transform_dict["translation"] = 0
                if scaling:
                    rdm_scaling = random.uniform(scaling * -1, scaling)
                    transform_dict["scaling"] = 1 + rdm_scaling
                else:
                    transform_dict["scaling"] = 0
            if hor_flip:
                if random.random() >= 0.5:
                    transform_dict["horizontal_flip"] = True
            if ver_flip:
                if random.random() >= 0.5:
                    transform_dict["vertical_flip"] = True

            return transform_dict

        return None

## deprecated
    def create_transform2(self, rotation, translation, scaling, hor_flip, ver_flip):
        """ Creates a transform ditionary from given parameters

        Args:
            rotation: Max rotation to sample from uniformly
            translation: Max translation to sample from uniformly
            scaling: Max scaling to sample from uniformly
            hor_flip: Wether to randomly horizontally flip
            ver_flip: Wether to randomly vertically flip

        Returns:
            Dictionary containing transformations
        """
        transform_list = []

        if rotation or translation or scaling or hor_flip or ver_flip:
            if rotation or translation or scaling:
                if rotation:
                    transform_list.append(iaa.Affine(rotate=(-rotation, rotation)))
                if translation:
                    # rdm_transl = random.uniform(0, translation / self.img_width)
                    transform_list.append(iaa.Affine(translate_px=(-translation, translation)))
                if scaling:
                    # rdm_scaling = random.uniform(scaling * -1, scaling)
                    transform_list.append(iaa.Affine((1 - scaling, 1 + scaling)))
            if hor_flip:
                transform_list.append(iaa.Fliplr(0.5))
            if ver_flip:
                transform_list.append(iaa.Flipud(0.5))

            return iaa.Sequential(transform_list)

        return None

# class Dataset_Wrapper():
#     def __init__(self):
#         pass

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

        img = Img_DataLoader.load_image(os.path.join(self.img_data_path, self.images_meta_data[idx]), self.img_width, self.img_height)

        annotation = self.annotations_data[img_id]

        data_item_dict = {"img": img,
                    "annotation": annotation}

        return data_item_dict
        # return img, annotation




class Cityscapes_Dataset(Base_Dataset_COCO):
    def __init__(self, dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, *args, **kwargs):
        super().__init__(dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, *args, **kwargs)
        # self.label_list = ct_scripts_label_list  # special data information from cityscapescripts - do not use in general


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