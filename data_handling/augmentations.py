from abc import ABC, abstractmethod
import os
import albumentations as albtions


class Augmenter():
    def __init__(self, augmentation_config):

        self.augmentation_config = augmentation_config

        self.transform_list = self.parse_augmentation_config(augmentation_config)

        self.transform = self.create_transform(self.transform_list)

    def parse_augmentation_config(self, config):
        transformation_list = []
        for key in config["transformations"].keys():
            if key == "affine":
                transform = albtions.augmentations.Affine(**config["transformations"][key])
            elif key == "randomcrop":
                transform = albtions.augmentations.RandomCrop(**config["transformations"][key])
            elif key == "coarse_dropout":
                transform = albtions.augmentations.CoarseDropout(**config["transformations"][key])
            elif key == "elastictransform":
                transform = albtions.augmentations.ElasticTransform(**config["transformations"][key])
            elif key == "horizontalflip":
                transform = albtions.augmentations.HorizontalFlip(**config["transformations"][key])
            elif key == "maskdropout":
                transform = albtions.augmentations.MaskDropout(**config["transformations"][key])
            elif key == "pixeldropout":
                transform = albtions.augmentations.PixelDropout(**config["transformations"][key])
            elif key == "advancedblur":
                transform = albtions.augmentations.AdvancedBlur(**config["transformations"][key])
            elif key == "colorjitter":
                transform = albtions.augmentations.ColorJitter(**config["transformations"][key])
            elif key == "downscale":
                transform = albtions.augmentations.Downscale(**config["transformations"][key])
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
        mask = data_dict["annotation"]
        transformed = self.transform(image=img, mask=mask)

        return {"img": transformed["image"],
                "annotation": transformed["mask"]}
