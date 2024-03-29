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
            elif key == "randomcontrast":
                transform = albtions.augmenations.RandomContrast(**config["transformations"][key])
            elif key == "randombrightness":
                transform = albtions.augmenations.RandomBrightnessContrast(**config["transformations"][key])
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
                raise NotImplementedError("Augmentation transformation " + str(key) + " is not implemented!")

            transformation_list.append(transform)

        return transformation_list

    def create_transform(self, transformation_list):
        return albtions.Compose(transformation_list)

    def apply_augmentation(self, data_dict):
        img = data_dict["img"]
        mask = data_dict["annotation_mask"]

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask[:, :, 0])
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
        data_dict["annotation_mask"] = transformed["mask"]#.astype(np.uint32)

        # plt.imshow(data_dict["annotation_mask"][:, :, 0])
        # plt.show()

        annotations_data_segments_dict_old = {el["id"]: el for el in data_dict["annotations_data"]["segments_info"]}
        annotations_data_segments_dict_tmp = []
        # annotations_data_segments_dict_tmp = {}
        # for i in range(len(data_dict["annotations_data"]["segments_info"])):
        #     el = data_dict["annotations_data"]["segments_info"][i]
        #     annotations_data_segments_dict_tmp[el["id"]] = el
        #     annotations_data_segments_dict_tmp[el["id"]]["indx"] = i
            # calculate bbox and area for segments new
        segment_ids, segment_id_areas = np.unique(data_dict["annotation_mask"][:, :, 0], return_counts=True)
        segment_ids = segment_ids.astype(np.int64)
        for segment_id, segment_id_area in zip(segment_ids, segment_id_areas):
            if segment_id == 0:
                continue
            # if segment_id in annotations_data_segments_dict_old.keys():
            annotations_data_segments_dict_tmp.append({"id": segment_id,
                                                      "category_id": annotations_data_segments_dict_old[segment_id]["category_id"],
                                                      "iscrowd": annotations_data_segments_dict_old[segment_id]["iscrowd"],
                                                      "area": int(segment_id_area)})
                # annotations_data_segments_dict_tmp[segment_id]["area"] = int(segment_id_area)
                # annotations_data_segments_dict_tmp[segment_id].pop("bbox", None)
                # segment_id_indx = annotations_data_segments_dict_tmp[segment_id][indx]
                # data_dict["annotations_data"]["segments_info"][segment_id_indx]["area"] = segment_id_area
                # data_dict["annotations_data"]["segments_info"][segment_id_indx].pop("bbox")
            # else:
            #     raise ValueError("Segment ID " + segment_id + " not in " + data_dict["annotations_data"]["image_id"])

        # plt.imshow(transformed["image"])
        # plt.show()
        # plt.imshow(transformed["mask"])
        # plt.show()
        data_dict["annotations_data"]["segments_info"] = annotations_data_segments_dict_tmp

        return data_dict

        # return {"img": transformed["image"],
        #         "annotation_mask": transformed["mask"],
        #         "annotations_data": data_dict["annotations_data"]}

    def apply_augmentation_plain_img(self, img):

        transformed_img = self.transform(image=img)

        return transformed_img["image"]
