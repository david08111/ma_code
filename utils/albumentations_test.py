import albumentations as A
import cv2
import os
from matplotlib import pyplot as plt

def albumentations_config_save(config_path, transform):
    A.save(transform, config_path, data_format="yaml")

def albumentations_config_load(config_path):
    return A.load(config_path, data_format="yaml")


def visualize_transform(img, mask, transform):

    transformed = transform(image=img, mask=mask)

    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.show()

if __name__ == "__main__":

    augmentations_config_path = "/work/scratch/dziuba/repos/ma_code/cfg/albumentations.yaml"

    img_path = "/work/scratch/dziuba/datasets/COCO_panoptic/val2017/000000004765.jpg"
    mask_path = "/work/scratch/dziuba/datasets/COCO_panoptic/panoptic_annotations_trainval2017/annotations/panoptic_val2017/000000004765.jpg"

    img = cv2.imread(img_path)
    mask = cv2.imread(img_path)

    test = A.__dict__()

    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    transform_dict = A.to_dict(transform)


    albumentations_config_save(augmentations_config_path, transform)

    loaded_transform = albumentations_config_load(augmentations_config_path)

    visualize_transform(img, mask, transform)