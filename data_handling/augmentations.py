from abc import ABC, abstractmethod
import os
import albumentations as albtions
from matplotlib import pyplot as plt
import numpy as np
import cv2


class Augmentation_Wrapper():
    def __init__(self, augmentation_config):

        self.augmentation_config = augmentation_config

        self.transform_list = self.parse_augmentation_config(augmentation_config)

        self.transform = self.create_transform(self.transform_list)

    def parse_augmentation_config(self, config):
        transformation_list = []
        for key in config["transformations"].keys():
            if key == "affine":
                transform = albtions.augmentations.Affine(**config["transformations"][key], mask_interpolation=cv2.INTER_NEAREST)
            elif key == "randomcrop":
                transform = albtions.augmentations.RandomCrop(**config["transformations"][key])
            elif key == "coarse_dropout":
                transform = albtions.augmentations.CoarseDropout(**config["transformations"][key])
            elif key == "elastictransform":
                transform = albtions.augmentations.ElasticTransform(**config["transformations"][key])
            elif key == "horizontalflip":
                transform = albtions.augmentations.HorizontalFlip(**config["transformations"][key])
            # elif key == "maskdropout":  # resulted in problems
            #     transform = albtions.augmentations.MaskDropout(**config["transformations"][key])
            elif key == "pixeldropout":
                transform = albtions.augmentations.PixelDropout(**config["transformations"][key])
            elif key == "advancedblur":
                transform = albtions.augmentations.AdvancedBlur(**config["transformations"][key])
            elif key == "colorjitter":
                transform = albtions.augmentations.ColorJitter(**config["transformations"][key])
            elif key == "downscale":
                transform = albtions.augmentations.Downscale(**config["transformations"][key], interpolation=cv2.INTER_NEAREST)
            elif key == "emboss":
                transform = albtions.augmentations.Emboss(**config["transformations"][key])
            elif key == "gaussnoise":
                transform = albtions.augmentations.GaussNoise(**config["transformations"][key])
            elif key == "isonoise":
                transform = albtions.augmentations.ISONoise(**config["transformations"][key])
            elif key == "motionblur":
                transform = albtions.augmentations.MotionBlur(**config["transformations"][key])
            elif key == "randomfog":
                transform = albtions.augmentations.RandomFog(**config["transformations"][key])
            elif key == "randomrain":
                transform = albtions.augmentations.RandomRain(**config["transformations"][key])
            elif key == "randomshadow":
                transform = albtions.augmentations.RandomShadow(**config["transformations"][key])
            elif key == "randomsnow":
                transform = albtions.augmentations.RandomSnow(**config["transformations"][key])
            elif key == "sharpen":
                transform = albtions.augmentations.Sharpen(**config["transformations"][key])
            elif key == "superpixels":
                transform = albtions.augmentations.Superpixels(**config["transformations"][key])
            else:
                raise("Augmentation transformation " + str(key) + " is not implemented!")

            transformation_list.append(transform)

        return transformation_list

    def create_transform(self, transformation_list):
        return albtions.Compose(transformation_list)

    def apply_augmentation(self, data_dict):
        img = data_dict["img"]
        mask = data_dict["annotation_mask"]

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()

        # test = data_dict["annotations_data"]["image_id"].strip()
        # if test == "konstanz_000000_000835":
        #     pass
        #
        # print("Image ID: " + data_dict["annotations_data"]["image_id"])
        # print("Img dtype: " + str(img.dtype))
        # print("Mask dtype: " + str(mask.dtype))

        transformed = self.transform(image=img, mask=mask)

        data_dict["img"] = transformed["image"]
        data_dict["annotation_mask"] = transformed["mask"]

        # plt.imshow(transformed["image"])
        # plt.show()
        # plt.imshow(transformed["mask"])
        # plt.show()


        return data_dict

        # return {"img": transformed["image"],
        #         "annotation_mask": transformed["mask"],
        #         "annotations_data": data_dict["annotations_data"]}
