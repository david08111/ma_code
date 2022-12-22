'''
Visualization demo for panoptic COCO sample_data

The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries


def visualize(json_file, segmentations_folder, img_folder):
    # whether from the PNG are used or new colors are generated
    generate_new_colors = True

    # json_file = './sample_data/panoptic_examples.json'
    # segmentations_folder = './sample_data/panoptic_examples/'
    # img_folder = './sample_data/input_images/'
    # panoptic_coco_categories = './panoptic_coco_categories.json'
    # panoptic_coco_categories = json_file["categories"]
    coco_d = None

    with open(json_file, 'r') as f:
        coco_d = json.load(f)

    # ann = np.random.choice(coco_d['annotations'])
    # ann = coco_d['annotations'][0]
    # with open(panoptic_coco_categories, 'r') as f:
    #     categories_list = json.load(f)
    categories_list = coco_d["categories"]
    categegories = {category['id']: category for category in categories_list}

    images_id_dict = {coco_d["images"][i]["id"]: coco_d["images"][i] for i in range(len(coco_d["images"]))}
    annotations_id_dict = {coco_d["annotations"][i]["image_id"]: coco_d["annotations"][i] for i in range(len(coco_d["annotations"]))}

    for ann_key in annotations_id_dict.keys():
        ann = annotations_id_dict[ann_key]
        # find input img that correspond to the annotation
        img = None

        image_info = images_id_dict[ann_key]

        try:
            img = np.array(
                Image.open(os.path.join(img_folder, image_info['file_name']))
            )
        except:
            try:
                base_folder_name = image_info['file_name'].split("_")[0]
                img = np.array(
                    Image.open(os.path.join(img_folder, base_folder_name, image_info['file_name']))
                )
            except:

                print("Undable to find correspoding input image.")


        segmentation = np.array(
            Image.open(os.path.join(segmentations_folder, ann['file_name'])),
            dtype=np.uint8
        )
        segmentation_id = rgb2id(segmentation)
        # find segments boundaries

        ##### workaround for boundary function since there is a not solved bug when source image type != 8bit

        # segmentation_mask_unique = np.zeros((segmentation_id.shape), dtype=np.uint8)
        # segmentation_unique_u8int = np.array(segmentation_id)
        #
        # unique_segmentation_ids = list(np.unique(segmentation_unique_u8int))[1:]
        # if len(unique_segmentation_ids) > 255:
        #     raise ValueError("Cant create proper visualization")
        #
        # unique_counter = 1
        # for id in unique_segmentation_ids:
        #     segmentation_mask_unique[segmentation_unique_u8int == id] = unique_counter
        #     unique_counter += 1
        #
        # segmentation_id = segmentation_id.astype(np.int64)

        #############
        background = np.array(segmentation_id)
        background[background!=0] = 1

        boundaries_segment_id = find_boundaries(segmentation_id, mode='outer', background=0)
        boundaries_background = find_boundaries(background, mode='outer', background=0)

        boundaries_segment_id[boundaries_background] = 0

        boundaries_segment_id = boundaries_segment_id.astype(np.uint8) * 255

        boundaries = boundaries_segment_id
        contours = dilation(boundaries)
        ####
        # boundaries = find_boundaries(segmentation_mask_unique, mode='outer', background=0).astype(np.uint8) * 255
        # contours = find_boundaries(segmentation_id, mode='outer', background=0).astype(np.uint8) * 255
        # contours = dilation(contours)
        #
        # plt.imshow(boundaries_segment_id)
        # plt.show()
        # plt.imshow(contours)
        # plt.show()

        if generate_new_colors:
            segmentation[:, :, :] = 0
            color_generator = IdGenerator(categegories)
            for segment_info in ann['segments_info']:
                color = color_generator.get_color(segment_info['category_id'])
                mask = segmentation_id == segment_info['id']
                segmentation[mask] = color

        # depict boundaries
        # segmentation[boundaries] = [0, 0, 0]

        # contours = find_boundaries(segmentation, mode="outer", background=0).astype(np.uint8) * 255
        # contours = dilation(contours)

        contours = np.expand_dims(contours, -1).repeat(4, -1)
        contours_img = Image.fromarray(contours, mode="RGBA")
        # contours_img = Image.fromarray(contours)

        segmentation_debug = segmentation

        img = Image.fromarray(img)
        segmentation = Image.fromarray(segmentation)

        out = Image.blend(img, segmentation, 0.5).convert(mode="RGBA")
        # out = Image.blend(out, segmentation, 0.5)
        out = Image.alpha_composite(out, contours_img)
        out.convert(mode="RGB")#.save(out_path)

        # if img is None:
        #     plt.figure()
        #     plt.imshow(segmentation)
        #     plt.axis('off')
        # else:
        #     plt.figure(figsize=(9, 5))
        #     plt.subplot(121)
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.subplot(122)
        #     plt.imshow(segmentation)
        #     plt.axis('off')
        #     plt.tight_layout()
        plt.imshow(contours_img)
        plt.show()

        plt.imshow(out)
        plt.show()
        pass

if __name__ == "__main__":
    json_file_path = "/work/scratch/dziuba/datasets/Cityscapes_COCO/gtFine/gtFine_val.json"
    segmentations_folder_path = "/work/scratch/dziuba/datasets/Cityscapes_COCO/gtFine/val"
    imgs_folder_path = "/work/scratch/dziuba/datasets/Cityscapes/leftImg8bit/val"
    visualize(json_file_path, segmentations_folder_path, imgs_folder_path)