import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


def save_results(save_path, img_data, predictions):
    # plt.imshow(img_big_rescaled[0, :, :, :])
    # plt.show()
    # plt.imshow(prediction[0, :, :, 0])
    # plt.show()

    # if not os.path.exists("/home/david/hiwijob/street_segmentation/skeyenet/results"):
    #     os.makedirs("/home/david/hiwijob/street_segmentation/skeyenet/results")
    #
    # for i in range(len(img_data)):
    #     cv2.imwrite(os.path.join("/home/david/hiwijob/street_segmentation/skeyenet/results", ("patch" + str(i) + ".jpg")), img_data[i])
    #     cv2.imwrite(os.path.join("/home/david/hiwijob/street_segmentation/skeyenet/results", ("pred_patch" + str(i) + ".jpg")), predictions[i, :, :, 0])

    thresh_val = 0.1

    for i in range(len(predictions)):

        predictions[i] = (predictions[i] > thresh_val).astype(np.uint8) * 256

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(img_data)):
        cv2.imwrite(os.path.join(save_path, ("img_" + str(i) + ".png")), img_data[i])
        cv2.imwrite(os.path.join(save_path, ("pred_" + str(i) + ".png")), predictions[i, :, :, 0])

        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        ax[0].imshow(img_data[i])
        ax[1].imshow(predictions[i, :, :, 0])
        plt.savefig(os.path.join(save_path, ("comp_" + str(i) + ".png")))

        plt.clf()
        plt.cla()
        plt.close()

        plt.imshow(img_data[i])
        plt.imshow(predictions[i, :, :, 0], cmap="jet", alpha=0.5)
        plt.savefig(os.path.join(save_path, ("masked_" + str(i) + ".png")))

