import numpy as np
from panopticapi.utils import IdGenerator, rgb2id

# DELETE
def convert_rgb_mask2segmentid_mask(rgb_mask, categories_dict):
    unique_rgb_vals = np.unique(rgb_mask.reshape(3, -1), axis=1)

    # id_gen = IdGenerator(categories_dict)

    for rgb_val in unique_rgb_vals:
        segment_id_mask = rgb_mask[0]

