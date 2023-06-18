from .data_loader import Img_DataLoader, Mask_DataLoader
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
import multiprocessing
from tqdm import tqdm
import itertools

import ctypes

class DataHandlerPlainImages(torch.utils.data.Dataset):
    def __init__(self, image_path, img_height, img_width, channels, device, num_workers, load_ram=False, load_orig_size=True, augmentations_config=None):
        self.image_path = image_path
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.load_ram = load_ram
        self.device = device
        self.num_workers = num_workers

        if load_orig_size:
            test_city_name = os.listdir(image_path)[0]
            test_img_name = os.listdir(os.path.join(image_path, test_city_name))[0]
            img_load_test = cv2.imread(os.path.join(image_path, test_city_name, test_img_name))
            self.load_ram_img_height = img_load_test.shape[0]
            self.load_ram_img_width = img_load_test.shape[1]
        else:
            self.load_ram_img_height = self.img_height
            self.load_ram_img_width = self.img_width

        if augmentations_config:
            self.augmenter = Augmentation_Wrapper(augmentations_config)
        else:
            self.augmenter = None

        # self.dataset_file_list = []
        # for root, dirs, files in os.walk(image_path):
        #     for filename in files:
        #         self.dataset_file_list.append(os.path.join(root, filename))
        self.dataset_file_list = [os.path.join(root, file) for root, dirs, files in os.walk(image_path) for file in files]
        # self.dataset_file_list = [os.path.join(image_path, file) for file in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, file))]

        if self.load_ram:
            img_loaded_block = multiprocessing.RawArray(ctypes.c_uint8, len(self) * self.load_ram_img_height * self.load_ram_img_width * self.channels)
            annotation_mask_loaded_block = multiprocessing.RawArray(ctypes.c_float, len(self) * self.load_ram_img_height * self.load_ram_img_width * self.channels)

            array_shape = (len(self), self.load_ram_img_height, self.load_ram_img_width, self.channels)
            print("Loading Dataset into the RAM...")
            multi_proc_proc_chunk_size = 15
            indx_partition_list = list(partition(range(len(self)), multi_proc_proc_chunk_size))

            with multiprocessing.Pool(max(self.num_workers, 1), initializer=init_worker_img_only, initargs=(self, img_loaded_block, array_shape)) as pool:
                with tqdm(total=len(indx_partition_list)) as pbar:
                    for result in pool.imap_unordered(load_ram_item3, indx_partition_list):

                        pbar.update()

            self.img_loaded_block = tonumpyarray(img_loaded_block, shape=(len(self), self.load_ram_img_height, self.load_ram_img_width, self.channels), dtype=np.uint8)

            mem_size = (self.img_loaded_block.nbytes) / 1000000000
            print("Dataset loading finished!")
            print(f"Dataset Mem Size: {mem_size} GB")

    def __len__(self):
        return len(self.dataset_file_list)

    def __getitem__(self, idx):

        if self.load_ram:
            img = self.img_loaded_block[idx]
            file_path = self.dataset_file_list[idx]

            if self.augmenter:
                img = self.augmenter.apply_augmentation_plain_img(img)

            if img.shape[0] > self.img_height and img.shape[1] > self.img_width:
                img = cv2.resize(img, (self.img_width, self.img_height))

            max_type_value = np.iinfo(img.dtype).max

            img = img.astype(np.float32)

            img /= max_type_value  # normalization to uint8

            return img, file_path
        else:

            file_path = self.dataset_file_list[idx]
            img = Img_DataLoader.load_image(file_path, self.load_ram_img_width, self.load_ram_img_height)

            if self.augmenter:
                img = self.augmenter.apply_augmentation_plain_img(img)

            if img.shape[0] > self.img_height and img.shape[1] > self.img_width:
                img = cv2.resize(img, (self.img_width, self.img_height))

            max_type_value = np.iinfo(img.dtype).max

            img = img.astype(np.float32)

            img /= max_type_value  # normalization to uint8

            return img, file_path


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

        self.load_ram = self.dataset_general_settings["load_ram"]

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
        if augmented_data_item_dict["img"].shape[0] > self.dataset_general_settings["img_height"] and augmented_data_item_dict["img"].shape[1] > self.dataset_general_settings["img_width"]:
            augmented_data_item_dict["img"] = cv2.resize(augmented_data_item_dict["img"], (self.dataset_general_settings["img_width"], self.dataset_general_settings["img_height"]))

        if augmented_data_item_dict["annotation_mask"].shape[0] > self.dataset_general_settings["img_height"] and augmented_data_item_dict["annotation_mask"].shape[1] > self.dataset_general_settings["img_width"]:
            augmented_data_item_dict["annotation_mask"] = cv2.resize(augmented_data_item_dict["annotation_mask"], (
            self.dataset_general_settings["img_width"], self.dataset_general_settings["img_height"]),
                                                         interpolation=cv2.INTER_NEAREST)

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

    def __init__(self, dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, load_ram, load_orig_size=True, num_workers=4, channels=3, *args, **kwargs):
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

        self.load_ram = load_ram

        self.num_workers = num_workers

        self.channels = channels


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

        annotation_mask = Mask_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height, self.segment_info["annotations"][idx], self.categories_id)

        data_item_dict = {"img": img,
                          "annotation_mask": annotation_mask,
                          "annotations_data": annotation,
                          "categories_isthing": self.categories_isthing}

        # data_item_dict = {"img": img,
        #                   "annotation_mask": annotation_mask,
        #                   "annotations_data": annotation}

        return data_item_dict
        # return img, annotation


def init_worker(cls_obj):
    global shared_cls_obj
    # store argument in the global variable for this process
    shared_cls_obj = cls_obj
    pass

def tonumpyarray(mp_arr, shape, dtype):
    """Convert shared multiprocessing array to numpy array.

    no data copying
    """
    return np.frombuffer(mp_arr, dtype=dtype).reshape(shape)

def init_worker3(cls_obj, img_loaded_block, annotation_mask_loaded_block, shape):
    global shared_cls_obj, shared_img_loaded_block, shared_annotation_mask_loaded_block
    # store argument in the global variable for this process
    shared_cls_obj = cls_obj

    shared_img_loaded_block = tonumpyarray(img_loaded_block,
                                         shape=shape,
                                         dtype=np.uint8)
    shared_annotation_mask_loaded_block = tonumpyarray(annotation_mask_loaded_block,
                                                     shape=shape,
                                                     dtype=np.float32)

    # shared_img_loaded_block = img_loaded_block
    # shared_annotation_mask_loaded_block = annotation_mask_loaded_block

def init_worker_img_only(cls_obj, img_loaded_block, shape):
    global shared_cls_obj, shared_img_loaded_block
    # store argument in the global variable for this process
    shared_cls_obj = cls_obj

    shared_img_loaded_block = tonumpyarray(img_loaded_block,
                                         shape=shape,
                                         dtype=np.uint8)


def init_worker2(img_loaded_block, annotation_mask_loaded_block):
    global shared_img_loaded_block, shared_annotation_mask_loaded_block
    # store argument in the global variable for this process
    shared_img_loaded_block = img_loaded_block
    shared_annotation_mask_loaded_block = annotation_mask_loaded_block

def load_ram_item2(indx_list):
    # print(indx_list)
    # print(multiprocessing.current_process())
    for i in indx_list:
        img_metadata = shared_cls_obj.images_meta_data[i]

        img_id = img_metadata["id"]

        img_city_name_dir = img_id.split("_")[0]

        # img_file_name = img_metadata["file_name"]

        shared_cls_obj.img_loaded_block[i] = Img_DataLoader.load_image(
            os.path.join(shared_cls_obj.img_data_path, img_city_name_dir, shared_cls_obj.images_meta_data[i]["file_name"]),
            shared_cls_obj.img_width, shared_cls_obj.img_height)

        annotation = shared_cls_obj.annotations_data[img_id]

        # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
        # if "file_name" not in annotation.keys():
        #     print(annotation)
        shared_cls_obj.annotation_mask_loaded_block[i] = Mask_DataLoader.load_image(
            os.path.join(shared_cls_obj.segment_masks_path, annotation["file_name"]), shared_cls_obj.img_width,
            shared_cls_obj.img_height,
            shared_cls_obj.segment_info["annotations"][i], shared_cls_obj.categories_id)

        # return shared_cls_obj

# def load_ram_item(i):
#     print(i)
#     # print(multiprocessing.current_process())
#     img_metadata = shared_cls_obj.images_meta_data[i]
#
#     img_id = img_metadata["id"]
#
#     img_city_name_dir = img_id.split("_")[0]
#
#     # img_file_name = img_metadata["file_name"]
#
#     shared_cls_obj.img_loaded_block[i] = Img_DataLoader.load_image(
#         os.path.join(shared_cls_obj.img_data_path, img_city_name_dir, shared_cls_obj.images_meta_data[i]["file_name"]),
#         shared_cls_obj.img_width, shared_cls_obj.img_height)
#
#     annotation = shared_cls_obj.annotations_data[img_id]
#
#     # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
#     # if "file_name" not in annotation.keys():
#     #     print(annotation)
#     shared_cls_obj.annotation_mask_loaded_block[i] = Mask_DataLoader.load_image(
#         os.path.join(shared_cls_obj.segment_masks_path, annotation["file_name"]), shared_cls_obj.img_width,
#         shared_cls_obj.img_height,
#         shared_cls_obj.segment_info["annotations"][i], shared_cls_obj.categories_id)

def load_ram_item(indx_list):
    print(indx_list)
    # print(multiprocessing.current_process())
    for i in indx_list:
        img_metadata = shared_cls_obj.images_meta_data[i]

        img_id = img_metadata["id"]

        img_city_name_dir = img_id.split("_")[0]

        # img_file_name = img_metadata["file_name"]

        shared_cls_obj.img_loaded_block[i] = Img_DataLoader.load_image(
            os.path.join(shared_cls_obj.img_data_path, img_city_name_dir, shared_cls_obj.images_meta_data[i]["file_name"]),
            shared_cls_obj.img_width, shared_cls_obj.img_height)

        annotation = shared_cls_obj.annotations_data[img_id]

        # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
        # if "file_name" not in annotation.keys():
        #     print(annotation)
        shared_cls_obj.annotation_mask_loaded_block[i] = Mask_DataLoader.load_image(
            os.path.join(shared_cls_obj.segment_masks_path, annotation["file_name"]), shared_cls_obj.img_width,
            shared_cls_obj.img_height,
            shared_cls_obj.segment_info["annotations"][i], shared_cls_obj.categories_id)

        # return shared_cls_obj

def load_ram_item3(indx_list):
    # print(indx_list)
    # print(multiprocessing.current_process())
    for i in indx_list:
        img_metadata = shared_cls_obj.images_meta_data[i]

        img_id = img_metadata["id"]

        img_city_name_dir = img_id.split("_")[0]

        # img_file_name = img_metadata["file_name"]

        shared_img_loaded_block[i] = Img_DataLoader.load_image(
            os.path.join(shared_cls_obj.img_data_path, img_city_name_dir, shared_cls_obj.images_meta_data[i]["file_name"]),
            shared_cls_obj.load_ram_img_width, shared_cls_obj.load_ram_img_height)

        annotation = shared_cls_obj.annotations_data[img_id]

        # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
        # if "file_name" not in annotation.keys():
        #     print(annotation)
        shared_annotation_mask_loaded_block[i] = Mask_DataLoader.load_image(
            os.path.join(shared_cls_obj.segment_masks_path, annotation["file_name"]), shared_cls_obj.load_ram_img_width,
            shared_cls_obj.load_ram_img_height,
            shared_cls_obj.segment_info["annotations"][i], shared_cls_obj.categories_id)

    # return (shared_img_loaded_block, shared_annotation_mask_loaded_block)

def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

class Cityscapes_Dataset(Base_Dataset_COCO):
    def __init__(self, dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, load_ram=False, load_orig_size=True, num_workers=4, channels=3, *args, **kwargs):
        super().__init__(dataset_name, dataset_type, img_data_path, segment_info_file_path, segment_masks_path, img_width, img_height, load_ram, load_orig_size, num_workers, channels, *args, **kwargs)
        # self.label_list = ct_scripts_label_list  # special data information from cityscapescripts - do not use in general

        if load_orig_size:
            test_city_name = os.listdir(img_data_path)[0]
            test_img_name = os.listdir(os.path.join(img_data_path, test_city_name))[0]
            img_load_test = cv2.imread(os.path.join(img_data_path, test_city_name, test_img_name))
            self.load_ram_img_height = img_load_test.shape[0]
            self.load_ram_img_width = img_load_test.shape[1]
        else:
            self.load_ram_img_height = self.img_height
            self.load_ram_img_width = self.img_width

        if self.load_ram:
            # self.img_loaded_block = np.empty((len(self), self.img_height, self.img_width, self.channels), dtype=np.uint8)
            # self.annotation_mask_loaded_block = np.empty((len(self), self.img_height, self.img_width, self.channels), dtype=np.float32)

            # img_loaded_block = np.empty((len(self), self.img_height, self.img_width, self.channels),
            #                                  dtype=np.uint8)
            # annotation_mask_loaded_block = np.empty((len(self), self.img_height, self.img_width, self.channels),
            #                                              dtype=np.float32)


            img_loaded_block = multiprocessing.RawArray(ctypes.c_uint8, len(self) * self.load_ram_img_height * self.load_ram_img_width * self.channels)
            annotation_mask_loaded_block = multiprocessing.RawArray(ctypes.c_float, len(self) * self.load_ram_img_height * self.load_ram_img_width * self.channels)

            array_shape = (len(self), self.load_ram_img_height, self.load_ram_img_width, self.channels)
            print("Loading Dataset into the RAM...")
            # tmp_ref = self

            # def init_worker(cls_obj):
            #     global shared_cls_obj
            #     # store argument in the global variable for this process
            #     shared_cls_obj = cls_obj

            # def load_ram_item(i):
            #     img_metadata = tmp_ref.images_meta_data[i]
            #
            #     img_id = img_metadata["id"]
            #
            #     img_city_name_dir = img_id.split("_")[0]
            #
            #     # img_file_name = img_metadata["file_name"]
            #
            #     shared_cls_obj.img_loaded_block[i] = Img_DataLoader.load_image(
            #         os.path.join(shared_cls_obj.img_data_path, img_city_name_dir, shared_cls_obj.images_meta_data[i]["file_name"]),
            #         shared_cls_obj.img_width, shared_cls_obj.img_height)
            #
            #     annotation = shared_cls_obj.annotations_data[img_id]
            #
            #     # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
            #     # if "file_name" not in annotation.keys():
            #     #     print(annotation)
            #     shared_cls_obj.annotation_mask_loaded_block[i] = Mask_DataLoader.load_image(
            #         os.path.join(shared_cls_obj.segment_masks_path, annotation["file_name"]), shared_cls_obj.img_width, shared_cls_obj.img_height,
            #         shared_cls_obj.segment_info["annotations"][i], shared_cls_obj.categories_id)
            multi_proc_proc_chunk_size = 15
            indx_partition_list = list(partition(range(len(self)), multi_proc_proc_chunk_size))
            # with multiprocessing.Pool(max(self.num_workers, 1), initializer=init_worker, initargs=(self,)) as pool:
            #     with tqdm(total=len(indx_partition_list)) as pbar:
            #         for result in pool.imap_unordered(load_ram_item, indx_partition_list):
            #             pbar.update()
                # for result in tqdm(pool.map(load_ram_item, range(len(self)))):
                #     pass
            with multiprocessing.Pool(max(self.num_workers, 1), initializer=init_worker3, initargs=(self, img_loaded_block, annotation_mask_loaded_block, array_shape)) as pool:
                with tqdm(total=len(indx_partition_list)) as pbar:
                    for result in pool.imap_unordered(load_ram_item3, indx_partition_list):
                        # test = tonumpyarray(img_loaded_block,
                        #                  shape=array_shape,
                        #                  dtype=np.uint8)
                        # test2 = tonumpyarray(annotation_mask_loaded_block,
                        #                              shape=array_shape,
                        #                              dtype=np.float32)
                        pbar.update()

            self.img_loaded_block = tonumpyarray(img_loaded_block, shape=(len(self), self.load_ram_img_height, self.load_ram_img_width, self.channels), dtype=np.uint8)
            self.annotation_mask_loaded_block = tonumpyarray(annotation_mask_loaded_block, shape=(len(self), self.load_ram_img_height, self.load_ram_img_width, self.channels), dtype=np.float32)
            # self.img_loaded_block = img_loaded_block
            # self.annotation_mask_loaded_block = annotation_mask_loaded_block
            # for i in tqdm(range(len(self))):
            #     img_metadata = self.images_meta_data[i]
            #
            #     img_id = img_metadata["id"]
            #
            #     img_city_name_dir = img_id.split("_")[0]
            #
            #     # img_file_name = img_metadata["file_name"]
            #
            #     self.img_loaded_block[i] = Img_DataLoader.load_image(
            #         os.path.join(self.img_data_path, img_city_name_dir, self.images_meta_data[i]["file_name"]),
            #         self.img_width, self.img_height)
            #
            #     annotation = self.annotations_data[img_id]
            #
            #     # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
            #     # if "file_name" not in annotation.keys():
            #     #     print(annotation)
            #     self.annotation_mask_loaded_block[i] = Mask_DataLoader.load_image(
            #         os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height,
            #         self.segment_info["annotations"][i], self.categories_id)

            mem_size = (self.img_loaded_block.nbytes + self.annotation_mask_loaded_block.nbytes) / 1000000000
            print("Dataset loading finished!")
            print(f"Dataset Mem Size: {mem_size} GB")

    # def load_ram_item(i):
    #     img_metadata = tmp_ref.images_meta_data[i]
    #
    #     img_id = img_metadata["id"]
    #
    #     img_city_name_dir = img_id.split("_")[0]
    #
    #     # img_file_name = img_metadata["file_name"]
    #
    #     self.img_loaded_block[i] = Img_DataLoader.load_image(
    #         os.path.join(self.img_data_path, img_city_name_dir, self.images_meta_data[i]["file_name"]),
    #         self.img_width, self.img_height)
    #
    #     annotation = self.annotations_data[img_id]
    #
    #     # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
    #     # if "file_name" not in annotation.keys():
    #     #     print(annotation)
    #     self.annotation_mask_loaded_block[i] = Mask_DataLoader.load_image(
    #         os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height,
    #         self.segment_info["annotations"][i], self.categories_id)

    def __getitem__(self, idx):
        """ Gets specific item from dataset with specified index

        Args:
            idx: Index to get data instance

        Returns:
            Data item from dataset with specified index
        """

        if idx > self.__len__():
             raise ValueError("Index " + idx + " out of range for dataset " + self.dataset_name)

        if self.load_ram:
            img_metadata = self.images_meta_data[idx]

            img_id = img_metadata["id"]

            # img_city_name_dir = img_id.split("_")[0]

            # img_file_name = img_metadata["file_name"]

            img = self.img_loaded_block[idx]

            annotation = self.annotations_data[img_id]

            # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
            # if "file_name" not in annotation.keys():
            #     print(annotation)
            annotation_mask = self.annotation_mask_loaded_block[idx]

            # from matplotlib import pyplot as plt
            # plt.imshow(annotation_mask)
            # plt.show()

            data_item_dict = {"img": img,
                              "annotation_mask": annotation_mask,
                              "annotations_data": annotation,
                              "categories_isthing": self.categories_isthing}
            # data_item_dict = {"img": img,
            #                   "annotation_mask": annotation_mask,
            #                   "annotations_data": annotation}

            return data_item_dict
        else:
            img_metadata = self.images_meta_data[idx]

            img_id = img_metadata["id"]

            img_city_name_dir = img_id.split("_")[0]

            # img_file_name = img_metadata["file_name"]

            img = Img_DataLoader.load_image(os.path.join(self.img_data_path, img_city_name_dir, self.images_meta_data[idx]["file_name"]), self.load_ram_img_width, self.load_ram_img_height)

            annotation = self.annotations_data[img_id]

            # annotation_mask = Img_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.img_width, self.img_height)
            # if "file_name" not in annotation.keys():
            #     print(annotation)
            annotation_mask = Mask_DataLoader.load_image(os.path.join(self.segment_masks_path, annotation["file_name"]), self.load_ram_img_width, self.load_ram_img_height, self.segment_info["annotations"][idx], self.categories_id)

            # annotation.pop("file_name")
            # plt.imshow(img)
            # plt.show()
            # plt.imshow(annotation_mask)
            # plt.show()

            data_item_dict = {"img": img,
                        "annotation_mask": annotation_mask,
                        "annotations_data": annotation,
                        "categories_isthing": self.categories_isthing}
            # data_item_dict = {"img": img,
            #                   "annotation_mask": annotation_mask,
            #                   "annotations_data": annotation}

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