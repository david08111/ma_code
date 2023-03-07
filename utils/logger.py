import logging
import os

import torch
import numpy as np
import wandb
from utils.sampler import SamplerWrapper
from copy import deepcopy

from functools import reduce
from torch.utils.tensorboard import SummaryWriter

def flatten_dict(dd, separator ='.', prefix =''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def create_nested_dict_from_list(config_dict, name_list, value, epoch):
    if len(name_list) > 1:
        elem = name_list.pop(0)
        config_dict[elem] = {}
        create_nested_dict_from_list(config_dict[elem], name_list, value, epoch)
    else:
        elem = name_list.pop(0)
        config_dict[elem] = value
        config_dict["Epoch"] = epoch

class TrainLogger(): # Wrapper for Logging to txt + TensorBoard + Wandb

    last_epoch_img = 0
    last_epoch_embedd = 0
    num_log_img_counter = 0
    num_log_img_and_mask_counter = 0
    num_log_embedds_counter = 0

    graph_logged = False

    def __init__(self, name, save_path, log_graph=False, img_log_freq=10, num_log_img=10, embedd_log_freq=10, num_log_embedds=1, wandb_config=None, hyperparams_dict={}, embedding_max_sample_size=5000, sampler=None):
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        self.img_log_freq = img_log_freq
        self.num_log_img = num_log_img
        self.num_log_embedds = num_log_embedds

        self.embedd_log_freq = embedd_log_freq
        self.embedding_max_sample_size = embedding_max_sample_size

        self.log_graph = log_graph

        if sampler:
            sampler_name = list(sampler.keys())[0]
            sampler_config = sampler[sampler_name]
            self.sampler = SamplerWrapper(sampler_name, sampler_config)
        else:
            sampler_name = "nthstep"
            sampler_config = {}
            self.sampler = SamplerWrapper(sampler_name, sampler_config)

        logging.basicConfig(filename=os.path.join(save_path, name + ".log"), format='Time - %(asctime)s - %(levelname)s - %(message)s', filemode='w', encoding='utf-8', level=logging.INFO, force=True)

        # self.logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # logging.setFormatter(self.logging_formatter)
        self.tb_logger = SummaryWriter(os.path.join(save_path, name + "_tb"), filename_suffix=".log")

        flattened_dict = flatten_dict(hyperparams_dict)

        for key in flattened_dict.keys():
            if not isinstance(flattened_dict[key], int) or not isinstance(flattened_dict[key], float) or not isinstance(flattened_dict[key], str) or not isinstance(flattened_dict[key], bool) or not isinstance(flattened_dict[key], torch.Tensor):
                flattened_dict[key] = str(flattened_dict[key])

        # self.tb_logger.add_hparams(flattened_dict, {"test": 1})

        wandb_path = os.path.join(save_path, name + "_wandb")
        if not os.path.isdir(wandb_path):
            os.makedirs(wandb_path, exist_ok=True)
        if wandb_config:
            wandb.init(**wandb_config, dir=wandb_path, config=hyperparams_dict, resume=True)
        else:
            wandb.init(project="MA", entity="david08111", dir=wandb_path, config=hyperparams_dict, resume=True)

    def get_caption_from_name(self, name):
        if isinstance(name, list):
            caption_name = name[0]
            for elem in name[1:]:
                caption_name = f"{caption_name} - {elem}"
            return caption_name
        else:
            return name

    def get_wandb_section_from_name(self, name):
        if isinstance(name, list):
            caption_name = name[0]
            for elem in name[1:]:
                caption_name = f"{caption_name}/{elem}"
            return caption_name
        else:
            return name

    def get_log_msg_from_name_epoch_value(self, name, value, epoch):
        if isinstance(name, list):
            caption_name = name[0]
            for elem in name[1:]:
                caption_name = f"{caption_name} - {elem}"
            log_msg = f"Epoch {epoch} - {caption_name} - {value}"
            return log_msg
        else:
            return f"Epoch {epoch} - {name} - {value}"

    def get_log_dict_from_name_epoch_value(self, name, value, epoch):
        if isinstance(name, list):
            wandb_log_dict = {}
            name_list_tmp = list(name)
            create_nested_dict_from_list(wandb_log_dict, name_list_tmp, value, epoch)
            return wandb_log_dict
        else:
            {name: value,
             "Epoch": epoch}

    def add_text(self, text, level, epoch):
        text = self.get_caption_from_name(text)

        log_msg = f"Epoch {epoch} - {text}"
        logging.log(level, log_msg)

    # def add_scalar_string_scalar(self, name, value, epoch):
    #
    #     log_msg = f"Epoch {epoch} - {name} - {value}"
    #     logging.log(logging.INFO, log_msg)
    #
    #     self.tb_logger.add_scalar(name, value, epoch)
    #
    #     wandb.log({name: value,
    #                "Epoch": epoch}, commit=True)
    #
    # def add_scalar_list_scalar(self, name_list, value, epoch):
    #     caption_name = self.get_caption_from_name_list(name_list)
    #     log_msg = f"Epoch {epoch} - {caption_name} - {value}"
    #     logging.log(logging.INFO, log_msg)
    #
    #     self.tb_logger.add_scalar(caption_name, value, epoch)
    #
    #     wandb_log_dict = self.get_nested_dict_from_name_list_and_value(name_list, value, epoch)
    #     # wandb.log({name: value,
    #     #            "Epoch": epoch}, commit=True)
    #     wandb.log(wandb_log_dict, commit=True)

    def add_scalar(self, name, value, epoch):
        # if isinstance(name, str) and not isinstance(value, dict):
        #     self.add_scalar_string_scalar(name, value, epoch)
        #
        # elif isinstance(name, list) and not isinstance(value, dict):
        #     self.add_scalar_list_scalar(name, value, epoch)
        # else:
        #     raise NameError("Not implemented yet!")
        log_msg = self.get_log_msg_from_name_epoch_value(name, value, epoch)
        logging.log(logging.INFO, log_msg)

        caption_name = self.get_caption_from_name(name)
        self.tb_logger.add_scalar(caption_name, value, epoch)

        wandb_caption = self.get_wandb_section_from_name(name)

        # wandb_log_dict = self.get_log_dict_from_name_epoch_value(name, value, epoch)
        wandb.log({wandb_caption: value,
                   "Epoch": epoch}, commit=True)
        # wandb.log(wandb_log_dict, commit=True)




    def add_image(self, name, img, annotations_data, epoch):
        if self.last_epoch_img != epoch:
            self.last_epoch_img = epoch
            self.num_log_img_counter = 0

        if epoch % self.img_log_freq == 0 and self.num_log_img_counter < self.num_log_img:
            for b in range(img.shape[0]):
                name_tmp = deepcopy(name)
                if isinstance(name, list):
                    name_tmp.append(annotations_data[b]['image_id'])
                else:
                    name_tmp += " - " + annotations_data[b]['image_id']

                caption = self.get_caption_from_name(name_tmp)

                log_msg = f"Epoch {epoch} - {caption} - LOGGED"
                logging.log(logging.DEBUG, log_msg)

                # caption = f"Epoch {epoch} - {name} - {annotations_data[b]['image_id']}"


                # caption = self.get_caption_from_name(name)

                # self.tb_logger.add_images(caption, img[b], epoch, dataformats="NHWC")
                self.tb_logger.add_images(caption, img[b], epoch, dataformats="HWC")

                # caption = f"Epoch {epoch} - {name} - {annotations_data['image_id']}"
                wandb_img = wandb.Image(img[b], caption=caption)

                wandb_caption = self.get_wandb_section_from_name(name)

                wandb.log({wandb_caption: wandb_img,
                           "Epoch": epoch})

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

                name_tmp = deepcopy(name)
                if isinstance(name, list):
                    name_tmp.append(annotations_data_gt[b]['image_id'])
                else:
                    name_tmp += " - " + annotations_data_gt[b]['image_id']

                caption = self.get_caption_from_name(name_tmp)

                log_msg_gt = f"Epoch {epoch} - {caption} - LOGGED"

                # log_msg = f"Epoch {epoch} - {name} - {annotations_data_gt[b]['image_id']} - LOGGED"
                logging.log(logging.DEBUG, log_msg_gt)

                caption_img = f"{caption} - IMG"
                caption_mask_gt = f"{caption} - MASK GT"
                caption_mask_pred = f"{caption} - MASK PRED"
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

                #######################


                # caption = f"Epoch {epoch} - {name} - {annotations_data_gt[b]['image_id']}"
                class_labels_gt = {}
                class_labels_pred = {}
                if categories_dict:

                    segment_uint8_counter = 1

                    for elem in annotations_data_gt[b]["segments_info"]:
                        class_labels_gt[segment_uint8_counter] = categories_dict[elem["category_id"]]["name"]
                        # test1 = np.unique(segmentid_mask_gt)
                        # test = segmentid_mask_gt == float(elem["id"])
                        # test2 = elem["category_id"]
                        # test3 = categories_dict[test2]["name"]
                        segmentid_mask_gt_final[segmentid_mask_gt == float(elem["id"])] = segment_uint8_counter
                        segment_uint8_counter += 1
                        if segment_uint8_counter > 255:
                            self.wandb_alert("Overflow", "Panoptic Segmentation mask calculation (for visualization) exceeds the class amount of 255! - visualization might be wrong")

                    segment_uint8_counter = 1

                    for elem in annotation_data_output[b]["segments_info"]:
                        class_labels_pred[segment_uint8_counter] = categories_dict[elem["category_id"]]["name"]
                        # test1 = np.unique(segmentid_mask_pred)
                        # test = segmentid_mask_pred == float(elem["id"])
                        # test2 = elem["category_id"]
                        # test3 = categories_dict[test2]["name"]
                        segmentid_mask_pred_final[segmentid_mask_pred == float(elem["id"])] = segment_uint8_counter
                        segment_uint8_counter += 1
                        if segment_uint8_counter > 255:
                            self.wandb_alert("Overflow",
                                             "Panoptic Segmentation mask calculation (for visualization) exceeds the class amount of 255! - visualization might be wrong")
                ##############
                wandb_img_caption = f"Epoch {epoch} - {caption}"
                wandb_img = wandb.Image(img[b], caption=wandb_img_caption, masks={
                    "ground_truth": {
                        "mask_data": segmentid_mask_gt_final,
                        "class_labels": class_labels_gt
                    },
                    "predictions": {
                        "mask_data": segmentid_mask_pred_final,
                        "class_labels": class_labels_pred
                    }
                })

                wandb_caption = self.get_wandb_section_from_name(name_tmp)

                wandb.log({wandb_caption: wandb_img,
                           "Epoch": epoch})

                self.num_log_img_and_mask_counter += 1


    def add_embeddings(self, name, output_embeddings, data_pts_names=None, annotations_data=None, column_names=None, epoch=-1):
        """

        Args:
            name:
            metadata:
            embeddings: ndarray
            epoch:

        Returns:

        """

        if self.last_epoch_embedd != epoch:
            self.last_epoch_embedd = epoch
            self.num_log_embedds_counter = 0

        for b in range(output_embeddings.shape[0]):

            if epoch % self.embedd_log_freq == 0 and self.num_log_embedds_counter < self.num_log_embedds:
                name = self.get_caption_from_name(name)

                if annotations_data:
                    caption_segment_ids = f"{name} - Segment Ids - {annotations_data[b]['image_id']}"
                    caption_cat_ids = f"{name} - Category Ids - {annotations_data[b]['image_id']}"
                else:
                    caption_segment_ids = f"{name} - Segment Ids - OUTPUT {b}"
                    caption_cat_ids = f"{name} - Category Ids - OUTPUT {b}"

                final_embedding = output_embeddings[b, ...].cpu().detach().numpy()
                data_pts_names = data_pts_names[b, :, :, :].cpu().detach().numpy()
                # test = list(final_embedding[0, ...].shape)
                no_embedding_samples = reduce(lambda x, y: x*y, list(final_embedding[0, ...].shape))

                if no_embedding_samples > self.embedding_max_sample_size:
                    final_embedding = self.sampler.sample(final_embedding)
                    data_pts_names = self.sampler.sample(data_pts_names)

                    data_pts_names_segment_ids = data_pts_names[0, ...]
                    data_pts_names_cat_ids = data_pts_names[1, ...]
                else:

                    # final_embedding = final_embedding[:, :int(final_embedding.shape[1] / 8),
                    #                   :int(final_embedding.shape[2] / 8)]
                    final_embedding = final_embedding.reshape(-1, final_embedding.shape[0])
                    # data_pts_names = data_pts_names[b, 0, :, :].cpu().detach().numpy()
                    # data_pts_names = data_pts_names[:int(data_pts_names.shape[0] / 8),
                    #                  :int(data_pts_names.shape[1] / 8)]

                    # data_pts_names_segment_ids = data_pts_names[0, ...]
                    # data_pts_names_cat_ids = data_pts_names[1, ...]

                    data_pts_names_segment_ids = data_pts_names.flatten()
                    data_pts_names_cat_ids = data_pts_names.flatten()

                data_pts_names_segment_ids = data_pts_names_segment_ids.tolist()
                data_pts_names_cat_ids = data_pts_names_cat_ids.tolist()

                final_embedding = final_embedding.T



                self.tb_logger.add_embedding(mat=final_embedding, metadata=data_pts_names_segment_ids, tag=caption_segment_ids, global_step=epoch)
                self.tb_logger.add_embedding(mat=final_embedding, metadata=data_pts_names_cat_ids,
                                             tag=caption_cat_ids, global_step=epoch)

                final_embedding = final_embedding.tolist()
                if not column_names:
                    columns = list(range(len(final_embedding[0])))
                else:
                    columns = column_names
                wandb_table = wandb.Table(columns=columns, data=final_embedding)

                wandb.log({f"Epoch {epoch} - {caption_segment_ids}": wandb_table,
                           "Epoch": epoch})

                self.num_log_embedds_counter += 1

    def add_figure(self):
        raise NameError("Not implemented yet!")

    def add_graph(self, model, inputs=None):
        if not self.graph_logged and self.log_graph:
            if inputs != None:
                self.tb_logger.add_graph(model, input_to_model=inputs)
            # self.tb_logger.add_graph(model, inputs)

            wandb.watch(model, log="all", log_graph=True)

            self.graph_logged = True

    def add_histogram(self, name, histogram_data_bins, bins, epoch):

        # caption_name = self.get_caption_from_name(name)

        wandb_caption = self.get_wandb_section_from_name(name)
        # WIP - add tb variant
        # self.tb_logger.add_histogram(caption_name, data_list, epoch, bins)
        #
        # np_hist = np.histogram(data_list, bins)
        # histogram_data_bins = histogram_data_bins.astype(np.int64)
        # histogram_data_bins = histogram_data_bins
        # bins[-1] = bins[-2] * 10
        # histogram_data_bins /= 1000
        np_hist = (histogram_data_bins, bins)
        # np_hist = (bins, histogram_data_bins)

        # from matplotlib import pyplot as plt
        # plt.bar(bins[:-1], histogram_data_bins)
        # plt.show()
        #
        # test_data = np.random.normal(scale=50, size=500)
        # bins_new = np.arange(-2, 2, 0.2)
        #
        # hist_data, bins_data = np.histogram(test_data, bins=bins_new)
        # hist_d = np.histogram(test_data, bins=20)
        #
        # wandb.log({"test-log": wandb.Histogram(np_histogram=hist_d),
        #            "Epoch": epoch}, commit=True)

        wandb.log({wandb_caption: wandb.Histogram(np_histogram=np_hist),
                   "Epoch": epoch}, commit=True)

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

    def img_needed_for_loging(self, epoch):
        """
        Wether there is an image needed for logging
        Args:
            epoch:

        Returns:

        """
        if self.last_epoch_embedd != epoch:
            self.last_epoch_embedd = epoch
            self.num_log_embedds_counter = 0

        if self.last_epoch_img != epoch:
            self.last_epoch_img = epoch
            self.num_log_img_and_mask_counter = 0

        if epoch % self.img_log_freq == 0 and self.num_log_img_and_mask_counter < self.num_log_img and epoch % self.embedd_log_freq == 0 and self.num_log_embedds_counter < self.num_log_embedds:
            return True
        else:
            return False

    def reset_img_emb_counter(self):
        self.num_log_img_and_mask_counter = 0
        self.num_log_embedds_counter = 0
        self.num_log_img_counter = 0
