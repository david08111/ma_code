import logging
import os

import torch
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter

class TrainLogger(): # Wrapper for Logging to txt + TensorBoard + Wandb

    last_epoch_img = 0
    num_log_img_counter = 0
    num_log_img_and_mask_counter = 0

    def __init__(self, name, save_path, img_log_freq=10, num_log_img=20, hyperparams_dict=None):
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        self.img_log_freq = img_log_freq
        self.num_log_img = num_log_img

        logging.basicConfig(filename=os.path.join(save_path, name + ".log"), format='Time - %(asctime)s - %(levelname)s - %(message)s', filemode='w', encoding='utf-8', level=logging.INFO, force=True)

        # self.logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # logging.setFormatter(self.logging_formatter)
        self.tb_logger = SummaryWriter(os.path.join(save_path, name + "_tb"), filename_suffix=".log")

        wandb_path = os.path.join(save_path, name + "_wandb")
        if not os.path.isdir(wandb_path):
            os.makedirs(wandb_path, exist_ok=True)
        wandb.init(project="MA", entity="david08111", dir=wandb_path)


    def add_text(self, text, level, epoch):
        log_msg = f"Epoch {epoch} - {text}"
        logging.log(level, log_msg)

    def add_scalar(self, name, value, epoch):
        log_msg = f"Epoch {epoch} - {name} - {value}"
        logging.log(logging.INFO, log_msg)

        self.tb_logger.add_scalar(name, value, epoch)

        wandb.log({name: value}, step=epoch)


    def add_image(self, name, img, annotations_data, epoch):
        if self.last_epoch_img != epoch:
            self.last_epoch_img = epoch
            self.num_log_img_counter = 0

        if epoch % self.img_log_freq == 0 and self.num_log_img_counter < self.num_log_img:
            for b in range(img.shape[0]):
                log_msg = f"Epoch {epoch} - {name} - {annotations_data[b]['image_id']} - LOGGED"
                logging.log(logging.DEBUG, log_msg)

                caption = f"Epoch {epoch} - {name} - {annotations_data[b]['image_id']}"

                # self.tb_logger.add_images(caption, img[b], epoch, dataformats="NHWC")
                self.tb_logger.add_images(caption, img[b], epoch, dataformats="HWC")

                # caption = f"Epoch {epoch} - {name} - {annotations_data['image_id']}"
                wandb_img = wandb.Image(img[b], caption=caption)

                wandb.log({name: wandb_img})

                self.num_log_img_counter += 1
                # self.tb_logger.add_image

    def add_image_and_mask(self, name, img, mask_gt, annotations_data_gt, mask_output, annotation_data_output, epoch, categories_dict=None):
        """

        Args:
            name:
            img: torch tensor
            mask_gt: torch tensor
            annotations_data_gt:
            mask_output:
            annotation_data_output:
            epoch:
            categories_dict:

        Returns:

        """
        if self.last_epoch_img != epoch:
            self.last_epoch_img = epoch
            self.num_log_img_and_mask_counter = 0

        if all(x == categories_dict[0] for x in categories_dict):
            categories_dict = categories_dict[0]
        else:
            raise ValueError("Implementation doesnt support multiple dataset category associations!") # conversion to unified categories should work

        if epoch % self.img_log_freq == 0 and self.num_log_img_and_mask_counter < self.num_log_img:
            for b in range(img.shape[0]):
            # for b in range(1):
                log_msg = f"Epoch {epoch} - {name} - {annotations_data_gt[b]['image_id']} - LOGGED"
                logging.log(logging.DEBUG, log_msg)

                caption_img = f"Epoch {epoch} - {name} - IMG - {annotations_data_gt[b]['image_id']}"
                caption_mask_gt = f"Epoch {epoch} - {name} - MASK GT - {annotations_data_gt[b]['image_id']}"
                caption_mask_pred = f"Epoch {epoch} - {name} - MASK PRED - {annotations_data_gt[b]['image_id']}"
                self.tb_logger.add_images(caption_img, img[b], epoch, dataformats="CHW")
                self.tb_logger.add_images(caption_mask_gt, mask_gt[b], epoch, dataformats="CHW")
                self.tb_logger.add_images(caption_mask_pred, mask_output[b], epoch, dataformats="CHW")

                mask_output_tmp = mask_output.to(dtype=torch.float32)

                segmentid_mask_gt = mask_gt[b, 0]
                # segmentid_mask_gt = mask_gt[b, 0] + 256 * mask_gt[b, 1] + 256 * 256 * mask_gt[b, 2]
                segmentid_mask_pred = mask_output_tmp[b, 0] + 256 * mask_output_tmp[b, 1] + 256 * 256 * mask_output_tmp[b, 2]

                # if isinstance(segmentid_mask_gt, torch.Tensor) and isinstance(segmentid_mask_pred, torch.Tensor):
                #     segmentid_mask_gt = segmentid_mask_gt.to(torch.int32)
                #     segmentid_mask_pred = segmentid_mask_pred.to(torch.int32)
                # elif isinstance(segmentid_mask_gt, np.ndarray) and isinstance(segmentid_mask_pred, np.ndarray):
                #     segmentid_mask_gt = segmentid_mask_gt.astype(np.int32)
                #     segmentid_mask_pred = segmentid_mask_pred.astype(np.int32)

                segmentid_mask_gt = segmentid_mask_gt.detach().cpu().numpy()
                segmentid_mask_pred = segmentid_mask_pred.detach().cpu().numpy()

                segmentid_mask_gt_final = np.zeros(segmentid_mask_gt.shape, dtype=np.uint8)
                segmentid_mask_pred_final = np.zeros(segmentid_mask_pred.shape, dtype=np.uint8)

                caption = f"Epoch {epoch} - {name} - {annotations_data_gt[b]['image_id']}"
                class_labels_gt = {}
                class_labels_pred = {}
                if categories_dict:

                    segment_uint8_counter = 1

                    for elem in annotations_data_gt[b]["segments_info"]:
                        class_labels_gt[segment_uint8_counter] = categories_dict[elem["category_id"]]["name"]
                        # test1 = np.unique(segmentid_mask_gt)
                        # test = segmentid_mask_gt == float(elem["id"])
                        test2 = elem["category_id"]
                        test3 = categories_dict[test2]["name"]
                        segmentid_mask_gt_final[segmentid_mask_gt == float(elem["id"])] = segment_uint8_counter
                        segment_uint8_counter += 1
                        if segment_uint8_counter > 255:
                            self.wandb_alert("Overflow", "Panoptic Segmentation mask calculation (for visualization) exceeds the class amount of 255! - visualization might be wrong")

                    segment_uint8_counter = 1

                    for elem in annotation_data_output[b]["segments_info"]:
                        class_labels_pred[segment_uint8_counter] = categories_dict[elem["category_id"]]["name"]
                        test1 = np.unique(segmentid_mask_pred)
                        test = segmentid_mask_pred == float(elem["id"])
                        test2 = elem["category_id"]
                        test3 = categories_dict[test2]["name"]
                        segmentid_mask_pred_final[segmentid_mask_pred == float(elem["id"])] = segment_uint8_counter
                        segment_uint8_counter += 1
                        if segment_uint8_counter > 255:
                            self.wandb_alert("Overflow",
                                             "Panoptic Segmentation mask calculation (for visualization) exceeds the class amount of 255! - visualization might be wrong")

                wandb_img = wandb.Image(img[b], caption=caption, masks={
                    "ground_truth": {
                        "mask_data": segmentid_mask_gt_final,
                        "class_labels": class_labels_gt
                    },
                    "predictions": {
                        "mask_data": segmentid_mask_pred_final,
                        "class_labels": class_labels_pred
                    }
                })

                wandb.log({f"IMG {b}": wandb_img})

                self.num_log_img_and_mask_counter += 1


    def add_embedding(self, name, embeddings, data_pts_names=None, column_names=None, epoch=None):
        """

        Args:
            name:
            metadata:
            embeddings: ndarray
            epoch:

        Returns:

        """
        self.tb_logger.add_embedding(mat=embeddings, metadata=data_pts_names, tag=name, global_step=epoch)

        embeddings_test = embeddings.tolist()
        wandb_table = wandb.Table(columns=list(range(embeddings.shape[1])), data=embeddings_test)

        wandb.log({name: wandb_table}, step=epoch)

    def add_figure(self):
        raise NameError("Not implemented yet!")

    def add_graph(self, model, inputs):
        self.tb_logger.add_graph(model, input_to_model=inputs)

        wandb.watch(model, log="all", log_graph=True)

    def add_hyperparams(self):
        raise NameError("Not implemented yet!")

    def flush(self):
        self.tb_logger.flush()

    def close(self):
        self.tb_logger.close()

    def wandb_alert(self, title, text, level=wandb.AlertLevel.WARN):
        wandb.alert(title, text, level)

    def wandb_add_scalar(self, data_dict):
        wandb.lob(data_dict)

    def wandb_add_graph(self, model, loss):
        wandb.watch(model, criterion=loss, log="all", log_graph=True)
