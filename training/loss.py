import torch
import torch.nn as nn
import torch.nn.modules as nn_modules
import torch.nn.functional as F
import numpy as np
import sys
import logging
from .metrics_new import pq_compute_custom, PQStat


float_precision_eps = sys.float_info.epsilon
float_precision_thr = float_precision_eps * 2

class Loss_Wrapper():
    def __init__(self, loss_config):
        self.loss_type = list(loss_config.keys())[0] # if multiple keys then only first one describes loss type
        self.loss_config = loss_config[self.loss_type]
        self.loss = self.parse_loss_config(self.loss_type, self.loss_config)

    def parse_loss_config(self, loss_type, loss_config):
        if loss_type == "weighted_sum":
            return Weighted_sum(**loss_config)
        elif loss_type == "average_sum":
            return Average_sum(**loss_config)
        elif loss_type == "info_nce":
            return InfoNCE(**loss_config)
        elif loss_type == "discriminative_contrast": # from the related paper
            return Discriminative_contrast_loss(**loss_config)
        elif loss_type == "spherical_contrast_panoptic":
            return Panoptic_spherical_contrastive_loss(**loss_config)
        elif loss_type == "spherical_contrast_panoptic_flexible":
            return Panoptic_spherical_contrastive_flexible_loss(**loss_config)
        elif loss_type == "reverse_huber":
            return ReverseHuberLoss(**loss_config)
        elif loss_type == "reverse_huber_threshold":
            return ReverseHuberLossThreshold(**loss_config)
        elif loss_type == "radius_cross_entropy_mse":
            return RadiusConditionedCrossEntropyMSE(**loss_config)
        elif loss_type == "sem_segm_cross_entropy":
            return SemanticSegmentationCrossEntropy(**loss_config)
        ## deprecated
        elif loss_type == "mse":
            return nn_modules.MSELoss(**loss_config)
        elif loss_type == "l1":
            return nn_modules.L1Loss(**loss_config)
        elif loss_type == "cross_entropy":
            return nn_modules.CrossEntropyLoss(**loss_config)
        elif loss_type == "bin_cross_entropy":
            return nn_modules.BCELoss(**loss_config)
        elif loss_type == "bin_cross_entropy_logits":
            return nn_modules.BCEWithLogitsLoss(**loss_config)
        elif loss_type == "kullback_leibler":
            return nn_modules.KLDivLoss(**loss_config)
        elif loss_type == "poisson_nnll":
            return nn_modules.PoissonNLLLoss(**loss_config)
        elif loss_type == "nll":
            return nn_modules.NLLLoss2d(**loss_config)
        elif loss_type == "smooth_l1":
            return nn_modules.SmoothL1Loss(**loss_config)
        elif loss_type == "cosine_embedding":
            return nn_modules.CosineEmbeddingLoss(**loss_config)
        elif loss_type == "bin_dice":
            return BinaryDiceLoss(**loss_config)
        elif loss_type == "dice":
            return DiceLoss(**loss_config)


class Metrics_Wrapper():
    def __init__(self, metric_config):

        # self.metric_config = dict(metric_config)

        self.metric_name = list(metric_config.keys())[0]
        self.metric_config = metric_config[self.metric_name]

        # self.metric_type = metric_config.pop()
        # if "num_thresholds" in metric_config.keys():
        #     metric_config.pop("num_thresholds")
        self.metric = self.set_metric(self.metric_name, self.metric_config)

    def set_metric(self, metric_name, metric_config):
        if metric_name == "bin_dice":
            return BinaryDiceLoss(**metric_config)
        elif metric_name == "dice":
            return DiceLoss(**metric_config)
        elif metric_name == "iou":
            return IoU(**metric_config)
        elif metric_name == "accuracy":
            return Accuracy(**metric_config)
        elif metric_name == "precision":
            return Precision(**metric_config)
        elif metric_name == "recall":
            return Recall(**metric_config)
        elif metric_name == "f1_score":
            return F1_Score(**metric_config)
        elif metric_name == "false_pos_rate":
            return False_Pos_Rate(**metric_config)
        elif metric_name == "class_cases":
            return Classification_cases(**metric_config)
        # elif metric_name == "bbox3d_iou":
            # return BBometric_namex3D_IOU(**metric_config)
        elif metric_name == "panoptic_quality":
            return Panoptic_Quality(**metric_config)
        elif metric_name == "recognition_quality":
            return Recognition_Quality(**metric_config)
        elif metric_name == "segmentation_quality":
            return Segmentation_Quality(**metric_config)
        # elif metric_type == "confusion_matrix":
        #     raise NotImplementedError

class Panoptic_Quality(nn.Module):
    def __init__(self, filter=None):
        super().__init__()
        self.filter = filter
        self.metric_tmp = PQStat()
        self.metric = None

    def reset(self):
        self.metric_tmp = PQStat()
        self.metric = None

    def forward(self, output_img, mask_img, pred_data_dict, gt_data_dict, categories, *args, **kwargs):

        output_img_proc = output_img.detach().cpu().numpy()
        mask_img_proc = mask_img.detach().cpu().numpy()


        new_pq_stat = pq_compute_custom(output_img_proc, mask_img_proc, pred_data_dict, gt_data_dict, categories, *args, **kwargs)

        self.metric_tmp += new_pq_stat
        # implement averaging over batch

    def process_end_batch(self, *args, **kwargs):

        if all(x == kwargs["categories"][0] for x in kwargs["categories"]):
            categories_dict = kwargs["categories"][0]
        else:
            raise ValueError("Implementation doesnt support multiple dataset category associations!") # conversion to unified categories should work

        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        results = {}
        for name, isthing in metrics:
            results[name], per_class_results = self.metric_tmp.pq_average(categories_dict, isthing=isthing)
            if name == 'All':
                results["per_class"] = per_class_results
            # results[name + '_per_class'] = per_class_results


        self.metric = results

    def log(self, logger, name, epoch, *args, **kwargs):
        if "categories" in kwargs:
            if all(x == kwargs["categories"][0] for x in kwargs["categories"]):
                categories_dict = kwargs["categories"][0]
            else:
                raise ValueError(
                    "Implementation doesnt support multiple dataset category associations!")  # conversion to unified categories should work
        else:
            categories_dict = None

        caption_name = logger.get_caption_from_name(name)

        for group_name in self.metric.keys():
            if "per_class" in group_name:
                for cat_id in self.metric[group_name].keys():

                    for metric_name in self.metric[group_name][cat_id].keys():
                        if categories_dict:
                            cat_id_name = categories_dict[cat_id]["name"]
                        else:
                            cat_id_name = cat_id
                        if metric_name == "pq":
                            metric_display_name = "Panoptic Quality"
                        elif metric_name == "sq":
                            metric_display_name = "Segmentation Quality"
                        elif metric_name == "rq":
                            metric_display_name = "Recognition Quality"
                        else:
                            continue
                        caption = f"{caption_name} - {metric_display_name} - {cat_id_name}"
                        logger.add_text(f"{caption} - {self.metric[group_name][cat_id][metric_name]}", logging.INFO, epoch)
                        caption_list = name + [metric_display_name, "Class", cat_id_name]
                        logger.add_scalar(caption_list, self.metric[group_name][cat_id][metric_name], epoch)
            else:
                for metric_name in self.metric[group_name].keys():
                    if metric_name == "pq":
                        metric_display_name = "Panoptic Quality"
                    elif metric_name == "sq":
                        metric_display_name = "Segmentation Quality"
                    elif metric_name == "rq":
                        metric_display_name = "Recognition Quality"
                    else:
                        continue
                    caption = f"{caption_name} - {metric_display_name} - {group_name}"
                    logger.add_text(f"{caption} - {self.metric[group_name][metric_name]}", logging.INFO, epoch)
                    # caption_list = name + [metric_display_name, group_name]
                    # caption_list = name + [f"{metric_display_name} - {group_name}"]
                    caption_list = name + [metric_display_name, group_name]
                    logger.add_scalar(caption_list, self.metric[group_name][metric_name], epoch)



# WIP
class Segmentation_Quality(nn.Module):
    def __init__(self, filter=None):
        super().__init__()

        self.filter = filter

    def forward(self, outputs, labels, *args, **kwargs):
        results = pq_compute_custom(outputs, labels, kwargs["categories"], *args, **kwargs)
        # need to filter for pq
        return results
# WIP
class Recognition_Quality(nn.Module):
    def __init__(self, filter=None):
        super().__init__()

        self.filter = filter

    def forward(self, outputs, labels, *args, **kwargs):
        results = pq_compute_custom(outputs, labels, kwargs["categories"], *args, **kwargs)
        # need to filter for pq
        return results

class SemanticSegmentationCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weight = torch.tensor(weight, dtype=torch.float, device=device)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, outputs, masks, annotations_data, *args, **kwargs):

        dataset_category_id_list = sorted(list(annotations_data[0]["categories_isthing"].keys()))

        target_mask = torch.zeros(outputs.shape, device=outputs.device)

        target_mask = torch.permute(target_mask, (1, 0, 2, 3))

        # unique_indices = torch.unique(masks[:, 1, :, :]).detach().cpu().int().numpy()

        for i in range(outputs.shape[1]):
            # indx_c = (masks[:, 1, :, :] == c).int()
            cat_id = dataset_category_id_list[i]
            target_mask[i, masks[:, 1, :, :] == cat_id] = 1

        target_mask = torch.permute(target_mask, (1, 0, 2, 3))

        # test = target_mask.detach().cpu().numpy()
        #
        # from matplotlib import pyplot as plt
        # for i in range(test.shape[1]):
        #     plt.imshow(test[0, i])
        #     plt.show()

        loss = self.cross_entropy(outputs, target_mask)

        return loss

    def process_end_batch(self):
        pass

    def log(self, logger, name, epoch, *args, **kwargs):
        pass

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature

    def forward(self, outputs, masks, annotations_data, *args, **kwargs):
        raise ValueError("Discriminative_contrast_loss not implemented yet!")

class Discriminative_contrast_loss(nn.Module):
    def __init__(self, margin_variance, margin_distance, weighting_list):
        super().__init__()

        self.margin_variance = margin_variance
        self.margin_distance = margin_distance
        self.weighting_list = weighting_list

    def forward(self, outputs_dict, labels_dict):
        raise ValueError("Discriminative_contrast_loss not implemented yet!")

class Panoptic_spherical_contrastive_loss(nn.Module):
    def __init__(self, cat_id_radius_order_map_list, loss_radius=None, radius_diff_dist=1, radius_start_val=0, cosine_emb_loss_margin=0, radius_loss_weight=0.5, similarity_loss_weight=0.5, hypsph_radius_map_list=None):
        super().__init__()
        self.cat_id_radius_order_map_list = cat_id_radius_order_map_list
        self.radius_diff_dist = radius_diff_dist
        self.radius_start_val = radius_start_val
        self.cosine_emb_loss_margin = cosine_emb_loss_margin

        if not loss_radius:
            loss_radius = {"mse": {}}
        self.loss_radius = Loss_Wrapper(loss_radius).loss
        # self.cosine_embedding_inst_discr_loss = torch.nn.CosineEmbeddingLoss(margin=cosine_emb_loss_margin, size_average=False, reduce=False, reduction=False)

        self.radius_loss_weight = radius_loss_weight
        self.similarity_loss_weight = similarity_loss_weight

        if hypsph_radius_map_list:
            self.hypsph_radius_map_list = hypsph_radius_map_list
            diff_list_tmp = [abs(hypsph_radius_map_list[i] - hypsph_radius_map_list[i+1]) for i in range(len(hypsph_radius_map_list) - 1)]
            self.mean_radius_diff = sum(diff_list_tmp) / len(diff_list_tmp)
        else:
            self.hypsph_radius_map_list = list(range(self.radius_start_val, self.radius_start_val + self.radius_diff_dist * len(self.cat_id_radius_order_map_list), self.radius_diff_dist))
            self.mean_radius_diff = self.radius_diff_dist


        radius_granularity = 10

        bin_elems = 11
        self.bin_step = radius_diff_dist / radius_granularity
        bin_roi_bound = self.bin_step * bin_elems
        bin_max_range = radius_diff_dist * 1000
        bins = np.arange(0, bin_roi_bound, self.bin_step).tolist() + [bin_max_range]
        bins = np.array(bins, dtype=np.float64)

        self.abs_radius_err_class_dict = {elem: np.zeros(bin_elems) for elem in self.cat_id_radius_order_map_list}
        self.abs_radius_err_class_dict["bins"] = bins

        self.radius_loss_item_class_dict = {elem: 0 for elem in self.cat_id_radius_order_map_list}
        self.radius_loss_item_class_dict["all"] = 0

        self.similarity_loss_item_class_dict = {elem: 0 for elem in self.cat_id_radius_order_map_list}
        self.similarity_loss_item_class_dict["all"] = 0

        self.radius_loss_counter = 0

        self.similarity_loss_counter = 0

    def forward(self, outputs, masks, annotations_data, *args, **kwargs):
        device = outputs.get_device()

        # loss_items_dict = {}

        # radius_loss = 0
        #
        # similarity_loss = 0
        # if "categories_dict" in kwargs:
        #     categories_dict = kwargs["categories_dict"]

        radius_loss = torch.tensor(0, dtype=torch.float32, device=device)

        similarity_loss = torch.tensor(0, dtype=torch.float32, device=device)

        unique_cat_ids = torch.unique(masks[:, 1, :, :])  # skip segment_id=0

        # for unique_cat_id in unique_cat_ids[1:]:
        #     loss_items_dict[f"Radius Loss - Cat ID {unique_cat_id}"] = 0

        outputs_reordered_tmp = torch.permute(outputs, (1, 0, 2, 3))
        masks_reordered_tmp = torch.permute(masks, (1, 0, 2, 3))

        # loss_radius = torch.nn.MSELoss(size_average=False, reduce=False, reduction=None)
        loss_radius = self.loss_radius
        # cosine_embedding_inst_discr_loss = self.cosine_embedding_inst_discr_loss

        radius_loss_counter = 0
        similarity_loss_counter = 0

        if self.radius_loss_weight > float_precision_thr:
            for unique_cat_id in unique_cat_ids[1:]:  # skip 0
                unique_cat_id = int(unique_cat_id.item())
                cat_id_radius_indx = self.cat_id_radius_order_map_list.index(unique_cat_id)
                radius = self.hypsph_radius_map_list[cat_id_radius_indx]
                outputs_indx_select = masks[:, 1, :, :] == unique_cat_id
                outputs_cat_id_embeddings = outputs_reordered_tmp[:, outputs_indx_select]
                # test = outputs_cat_id_embeddings[:, 0].detach().cpu().numpy()
                # test2 = np.multiply(test, test.T)
                # test3 = np.sum(test2)
                outputs_cat_id_embeddings_norm = torch.norm(outputs_cat_id_embeddings, 2, dim=0)
                # radius_sqared_loss_part = torch.full(outputs_cat_id_embeddings_norm.size(), radius*radius, device=device)
                radius_loss_part = torch.full(outputs_cat_id_embeddings_norm.size(), radius, device=device, dtype=torch.float32)
                # test_mse_loss_mean = torch.nn.MSELoss()
                # test = test_mse_loss(outputs_cat_id_embeddings_norm, radius_loss_part).detach().cpu().numpy()
                # test2 = np.mean(test)
                # test_mean = test_mse_loss_mean(outputs_cat_id_embeddings_norm, radius_loss_part).detach().cpu().numpy()
                loss_tmp = loss_radius(outputs_cat_id_embeddings_norm, radius_loss_part)
                radius_loss += loss_tmp

                abs_error = torch.abs(outputs_cat_id_embeddings_norm - radius_loss_part).detach().cpu().numpy()

                # abs_error_histogram = np.histogram(abs_error, bins=np.arange())

                np_hist, bins = np.histogram(abs_error, self.abs_radius_err_class_dict["bins"])

                self.abs_radius_err_class_dict[unique_cat_id] += np_hist

                self.radius_loss_item_class_dict[unique_cat_id] += loss_tmp.item()


                # loss_items_dict[f"Radius Loss/Part - Cat ID {unique_cat_id}"] = loss_tmp.item()
                # if categories_dict:
                #     loss_items_dict[f"Radius Loss/Part - {categories_dict[unique_cat_id].name}"] = loss_tmp.item()
                # else:
                #     loss_items_dict[f"Radius Loss/Part - Cat ID {unique_cat_id}"] = loss_tmp.item()
                # loss_item_radius_key_tmp = f"Radius Loss - Cat ID {unique_cat_id}"
                # if loss_item_radius_key_tmp not in loss_items_dict:
                #     loss_items_dict[f"Radius Loss - Cat ID {unique_cat_id}"] = loss_tmp.item()
                # else:
                #     loss_items_dict[f"Radius Loss - Cat ID {unique_cat_id}"] += loss_tmp.item()
                radius_loss_counter += outputs.shape[0]

        # test = outputs.shape[0]
        # radius_loss_counter *= outputs.shape[0] #take batch size into account for normalization

        ### instance discrimination part

        # reduce amount of masks with "isthing" part from masks - B x H x W x (segment_id, cat_id, isthing)

        batch_size = masks.shape[0]

        if self.similarity_loss_weight > float_precision_thr:
            for b in range(batch_size):

                inst_discr_masks = masks_reordered_tmp[:2, b, masks[b, 2, :, :] == True]

                unique_cat_ids = torch.unique(inst_discr_masks[1, :])

                for unique_cat_id in unique_cat_ids:
                    unique_cat_id = int(unique_cat_id.item())
                    segments_id_data = masks_reordered_tmp[:, b, masks_reordered_tmp[1, b, :, :] == unique_cat_id]

                    unique_segment_ids = torch.unique(segments_id_data[0, :])

                    segment_id_embeddings_dict = {}
                    # gather embeddings and calculate cosineembeddingloss with itself

                    for unique_segment_id in unique_segment_ids:
                        segment_id_embeddings = outputs[b, :, masks_reordered_tmp[0, b, :, :] == unique_segment_id]
                        segment_id_embeddings_dict[unique_segment_id.item()] = segment_id_embeddings

                        segment_id_embeddings = torch.div(segment_id_embeddings, torch.norm(segment_id_embeddings, 2, dim=0) + 0.000001)
                        dot_product_embeddings = torch.matmul(torch.transpose(segment_id_embeddings, 0, 1), segment_id_embeddings)

                        dot_product_embeddings = torch.triu(dot_product_embeddings)
                        # test = dot_product_embeddings.nonzero(as_tuple=True)
                        dot_product_embeddings = dot_product_embeddings[dot_product_embeddings.nonzero(as_tuple=True)]
                        dot_product_embeddings = torch.mul(dot_product_embeddings, -1)
                        dot_product_embeddings = torch.add(dot_product_embeddings, 1)
                        dot_product_embeddings_mean = torch.mean(dot_product_embeddings)
                        similarity_loss += dot_product_embeddings_mean
                        similarity_loss_counter += 1


                        self.similarity_loss_item_class_dict[unique_cat_id] += similarity_loss.item()

                        # loss_item_similarity_key_tmp = f"Similarity Loss/Part - Cat ID - {unique_cat_id} - Similar"
                        # if loss_item_similarity_key_tmp not in loss_items_dict:
                        #     loss_items_dict[loss_item_similarity_key_tmp] = similarity_loss.item()
                        # else:
                        #     loss_items_dict[loss_item_similarity_key_tmp] += similarity_loss.item()


                    for i in range(unique_segment_ids.shape[0] - 1):
                        unique_segment_id = unique_segment_ids[i]
                        for neg_unique_segment_id in unique_segment_ids[i+1:]:

                            curr_embedding = segment_id_embeddings_dict[unique_segment_id.item()]
                            neg_embedding = segment_id_embeddings_dict[neg_unique_segment_id.item()]

                            curr_embedding = torch.div(curr_embedding, torch.norm(curr_embedding, 2, dim=0) + 0.000001)
                            neg_embedding = torch.div(neg_embedding, torch.norm(neg_embedding, 2, dim=0) + 0.000001)

                            dot_product_embeddings = torch.matmul(torch.transpose(curr_embedding, 0, 1), neg_embedding)

                            dot_product_embeddings = torch.triu(dot_product_embeddings)
                            # test = dot_product_embeddings.nonzero(as_tuple=True)
                            dot_product_embeddings = dot_product_embeddings[dot_product_embeddings.nonzero(as_tuple=True)]
                            dot_product_embeddings = torch.add(dot_product_embeddings, -self.cosine_emb_loss_margin)
                            dot_product_embeddings = torch.clamp(dot_product_embeddings, min=0)
                            dot_product_embeddings_mean = torch.mean(dot_product_embeddings)

                            similarity_loss += dot_product_embeddings_mean
                            similarity_loss_counter += 1

                            self.similarity_loss_item_class_dict[unique_cat_id] += similarity_loss.item()

                            # loss_item_similarity_key_tmp = f"Similarity Loss/Part - Cat ID - {unique_cat_id} - Disimilar"
                            # if loss_item_similarity_key_tmp not in loss_items_dict:
                            #     loss_items_dict[loss_item_similarity_key_tmp] = similarity_loss.item()
                            # else:
                            #     loss_items_dict[loss_item_similarity_key_tmp] += similarity_loss.item()

        self.radius_loss_item_class_dict["all"] += radius_loss.item()
        self.similarity_loss_item_class_dict["all"] += similarity_loss.item()


        if radius_loss_counter > 0:
            radius_loss /= radius_loss_counter

            # for key in loss_items_dict.keys():
            #     if "Radius" in key:
            #         loss_items_dict[key] /= radius_loss_counter

        if similarity_loss_counter > 0:
            similarity_loss /= similarity_loss_counter

            # for key in loss_items_dict.keys():
            #     if "Similarity" in key:
            #         loss_items_dict[key] /= similarity_loss_counter

        self.radius_loss_counter += radius_loss_counter
        self.similarity_loss_counter += similarity_loss_counter

        # loss_items_dict = {"Radius Loss": radius_loss.item(),
        #                    "Similarity Loss": similarity_loss.item()}
        # loss_items_dict["Radius Loss"] = radius_loss.item()
        # loss_items_dict["Similarity Loss"] = similarity_loss.item()


        total_loss = self.radius_loss_weight * radius_loss + self.similarity_loss_weight * similarity_loss

        # return total_loss, loss_items_dict
        return total_loss

    def process_end_batch(self, *args, **kwargs):

        for key in self.radius_loss_item_class_dict:
            self.radius_loss_item_class_dict[key] /= self.radius_loss_counter

        for key in self.similarity_loss_item_class_dict:
            self.similarity_loss_item_class_dict[key] /= self.radius_loss_counter




    def log(self, logger, name, epoch, *args, **kwargs):

        if "categories" in kwargs:
            if all(x == kwargs["categories"][0] for x in kwargs["categories"]):
                categories_dict = kwargs["categories"][0]
            else:
                raise ValueError(
                    "Implementation doesnt support multiple dataset category associations!")  # conversion to unified categories should work
        else:
            categories_dict = None

        caption_name = logger.get_caption_from_name(name)

        self.abs_radius_err_class_dict["bins"][-1] =  self.abs_radius_err_class_dict["bins"][-2] + self.bin_step

        ##### radius abs err histogram
        for id in self.abs_radius_err_class_dict:
            if id == "bins":
                continue
            caption = f"{caption_name}/Abs Radius Error/Part - {categories_dict[id]['name']}"
            logger.add_text(f"{caption} - {self.abs_radius_err_class_dict[id]}", logging.INFO, epoch)
            caption_list = name + [f"Abs Radius Error/Part - {categories_dict[id]['name']}"]
            logger.add_histogram(caption_list, self.abs_radius_err_class_dict[id], self.abs_radius_err_class_dict["bins"], epoch)
        ####

        for id in self.radius_loss_item_class_dict:
            if id == "all":
                caption = f"{caption_name}/Item - Radius Loss"
                logger.add_text(f"{caption} - {self.radius_loss_item_class_dict[id]}", logging.INFO, epoch)
                caption_list = name + ["Item - Radius Loss"]
                logger.add_scalar(caption_list, self.radius_loss_item_class_dict[id], epoch)
            else:
                caption = f"{caption_name}/Item - Radius Loss/Part - {categories_dict[id]['name']}"
                logger.add_text(f"{caption} - {self.radius_loss_item_class_dict[id]}", logging.INFO, epoch)
                caption_list = name + ["Item - Radius Loss", f"Part - {categories_dict[id]['name']}"]
                logger.add_scalar(caption_list, self.radius_loss_item_class_dict[id], epoch)

        for id in self.similarity_loss_item_class_dict:
            if id == "all":
                caption = f"{caption_name}/Item - Similarity Loss"
                logger.add_text(f"{caption} - {self.similarity_loss_item_class_dict[id]}", logging.INFO, epoch)
                caption_list = name + ["Item - Similarity Loss"]
                logger.add_scalar(caption_list, self.similarity_loss_item_class_dict[id], epoch)
            else:
                caption = f"{caption_name}/Item - Similarity Loss/Part - {categories_dict[id]['name']}"
                logger.add_text(f"{caption} - {self.similarity_loss_item_class_dict[id]}", logging.INFO, epoch)
                caption_list = name + ["Item - Similarity Loss", f"Part - {categories_dict[id]['name']}"]
                logger.add_scalar(caption_list, self.similarity_loss_item_class_dict[id], epoch)

class Panoptic_spherical_contrastive_flexible_loss(nn.Module):
    def __init__(self, sphere_ct_contr_loss, loss_radius, radius, cosine_emb_loss_margin=0, sphere_ct_contr_loss_weight=1, radius_loss_weight=1, similarity_loss_weight=1):
        super().__init__()
        self.sphere_ct_contr_loss = Loss_Wrapper(sphere_ct_contr_loss).loss
        self.radius_loss = Loss_Wrapper(loss_radius).loss
        # self.outer_radius_loss = Loss_Wrapper(outer_radius_loss).loss
        self.radius = radius

        self.cosine_emb_loss_margin = cosine_emb_loss_margin

        self.sphere_ct_contr_loss_weight = sphere_ct_contr_loss_weight
        self.radius_loss_weight = radius_loss_weight
        self.similarity_loss_weight = similarity_loss_weight

        self.cat_mean_embedding_dict = {}


    def forward(self, outputs, masks, annotations_data, *args, **kwargs):
        device = outputs.get_device()

        radius_loss = torch.tensor(0, dtype=torch.float32, device=device)
        similarity_loss = torch.tensor(0, dtype=torch.float32, device=device)

        unique_cat_ids = torch.unique(masks[:, 1, :, :])  # skip segment_id=0

        outputs_reordered_tmp = torch.permute(outputs, (1, 0, 2, 3))
        masks_reordered_tmp = torch.permute(masks, (1, 0, 2, 3))

        radius_loss_counter = 0
        similarity_loss_counter = 0


        #calc/update mean radius for each class
################


##################
        if self.radius_loss_weight > float_precision_thr:
            for unique_cat_id in unique_cat_ids[1:]:  # skip 0
                unique_cat_id = int(unique_cat_id.item())

                outputs_indx_select = masks[:, 1, :, :] == unique_cat_id
                outputs_cat_id_embeddings = outputs_reordered_tmp[:, outputs_indx_select]
                # test = outputs_cat_id_embeddings[:, 0].detach().cpu().numpy()
                # test2 = np.multiply(test, test.T)
                # test3 = np.sum(test2)
                outputs_cat_id_embeddings_norm = torch.norm(outputs_cat_id_embeddings, 2, dim=0)

                cat_id_mean_embedding_local = torch.mean(outputs_cat_id_embeddings_norm, dim=1)

                self.cat_mean_embedding_dict[unique_cat_id]

                # radius_sqared_loss_part = torch.full(outputs_cat_id_embeddings_norm.size(), radius*radius, device=device)
                radius_loss_part = torch.full(outputs_cat_id_embeddings_norm.size(), self.radius, device=device,
                                              dtype=torch.float32)
                # test_mse_loss_mean = torch.nn.MSELoss()
                # test = test_mse_loss(outputs_cat_id_embeddings_norm, radius_loss_part).detach().cpu().numpy()
                # test2 = np.mean(test)
                # test_mean = test_mse_loss_mean(outputs_cat_id_embeddings_norm, radius_loss_part).detach().cpu().numpy()
                loss_tmp = self.radius_loss(outputs_cat_id_embeddings_norm, radius_loss_part)
                radius_loss += loss_tmp

                abs_error = torch.abs(outputs_cat_id_embeddings_norm - radius_loss_part).detach().cpu().numpy()

                # abs_error_histogram = np.histogram(abs_error, bins=np.arange())

                np_hist, bins = np.histogram(abs_error, self.abs_radius_err_class_dict["bins"])

                self.abs_radius_err_class_dict[unique_cat_id] += np_hist

                self.radius_loss_item_class_dict[unique_cat_id] += loss_tmp.item()

                # loss_items_dict[f"Radius Loss/Part - Cat ID {unique_cat_id}"] = loss_tmp.item()
                # if categories_dict:
                #     loss_items_dict[f"Radius Loss/Part - {categories_dict[unique_cat_id].name}"] = loss_tmp.item()
                # else:
                #     loss_items_dict[f"Radius Loss/Part - Cat ID {unique_cat_id}"] = loss_tmp.item()
                # loss_item_radius_key_tmp = f"Radius Loss - Cat ID {unique_cat_id}"
                # if loss_item_radius_key_tmp not in loss_items_dict:
                #     loss_items_dict[f"Radius Loss - Cat ID {unique_cat_id}"] = loss_tmp.item()
                # else:
                #     loss_items_dict[f"Radius Loss - Cat ID {unique_cat_id}"] += loss_tmp.item()
                radius_loss_counter += outputs.shape[0]

        # test = outputs.shape[0]
        # radius_loss_counter *= outputs.shape[0] #take batch size into account for normalization

        ### instance discrimination part

        # reduce amount of masks with "isthing" part from masks - B x H x W x (segment_id, cat_id, isthing)

        batch_size = masks.shape[0]


    def process_end_batch(self):
        pass

    def log(self, logger, name, epoch, *args, **kwargs):
        pass

class Panoptic_spherical_contrastive_flexible_hinge_loss(nn.Module):
    def __init__(self, inner_radius_loss, outer_radius_loss, radius, contr_hinge_dist, cosine_emb_loss_margin=0, radius_loss_weight=0.5, similarity_loss_weight=0.5):
        super().__init__()

        self.inner_radius_loss = Loss_Wrapper(inner_radius_loss).loss
        self.outer_radius_loss = Loss_Wrapper(outer_radius_loss).loss
        self.radius = radius

        self.cosine_emb_loss_margin = cosine_emb_loss_margin

        self.radius_loss_weight = radius_loss_weight
        self.similarity_loss_weight = similarity_loss_weight

class RadiusConditionedCrossEntropyMSE(nn.Module):
    def __init__(self):
        super().__init__()
        # def __init__(self, cat_id_radius_order_map_list, radius_diff_dist=1, radius_start_val=0, hypsph_radius_map_list=None):
        #     super().__init__()
        # self.cat_id_radius_order_map_list = cat_id_radius_order_map_list
        # self.radius_diff_dist = radius_diff_dist
        # self.radius_start_val = radius_start_val
        #
        # if hypsph_radius_map_list:
        #     self.hypsph_radius_map_list = hypsph_radius_map_list
        #     diff_list_tmp = [abs(hypsph_radius_map_list[i] - hypsph_radius_map_list[i + 1]) for i in
        #                      range(len(hypsph_radius_map_list) - 1)]
        #     self.mean_radius_diff = sum(diff_list_tmp) / len(diff_list_tmp)
        # else:
        #     self.hypsph_radius_map_list = list(range(self.radius_start_val,
        #                                              self.radius_start_val + self.radius_diff_dist * len(
        #                                                  self.cat_id_radius_order_map_list), self.radius_diff_dist))
        #     self.mean_radius_diff = self.radius_diff_dist

        self.loss_radius_cond2 = Loss_Wrapper({"reverse_huber_threshold": {}}).loss
        # self.mse_loss = torch.nn.MSELoss()

    def forward(self, inputs, targets):
        target_radius = targets[0].item()

        scaled = torch.div(inputs, target_radius)
        log_scaled = torch.mul(torch.log(scaled), -1)

        cond2_loss = self.loss_radius_cond2(inputs, targets)

        # test3 = torch.sum(inputs < target_radius)
        # test2 = self.mse_loss(inputs, targets)
        # loss = torch.mean(torch.where(inputs < target_radius, log_scaled, self.mse_loss(inputs, targets)))
        loss = torch.mean(torch.where(inputs < target_radius, log_scaled, cond2_loss))

        return loss

    def process_end_batch(self):
        pass

    def log(self, logger, name, epoch, *args, **kwargs):
        pass


class Weighted_sum(nn.Module):
    def __init__(self, loss_list, weights_list):
        super().__init__()

        self.loss_list = []
        self.weights_list = weights_list

        for loss in loss_list:
            self.loss_list.append(Loss_Wrapper(loss))


    def forward(self, outputs_dict, labels_dict):
        total_loss = 0

        for i in range(len(self.loss_list)):
            total_loss += self.weights_list * self.loss_list[i](outputs_dict, labels_dict)

        return total_loss

class Average_sum(nn.Module):
    def __init__(self, loss_list):
        super().__init__()

        self.loss_list = []

        for loss in loss_list:
            self.loss_list.append(Loss_Wrapper(loss))


    def forward(self, outputs_dict, labels_dict):
        total_loss = 0

        for i in range(len(self.loss_list)):
            total_loss += self.loss_list[i](outputs_dict, labels_dict)

        return total_loss / len(self.loss_list)

class ReverseHuberLoss(nn.Module):
    def __init__(self, factor=0.2):
        super().__init__()
        self.factor = factor

    def forward(self, output, target):
        diff = output - target
        absdiff = torch.abs(diff)
        C = self.factor * torch.max(absdiff).item()
        # test = torch.where(absdiff < C, absdiff, (absdiff * absdiff + C * C) / (2 * C))
        return torch.mean(torch.where(absdiff < C, absdiff, (diff * diff + C * C) / (2 * C)))

    def process_end_batch(self):
        pass

    def log(self, logger, name, epoch, *args, **kwargs):
        pass

class ReverseHuberLossThreshold(nn.Module):
    def __init__(self, threshold=1):
        super().__init__()
        self.threshold = threshold

    def forward(self, output, target):
        diff = output - target
        absdiff = torch.abs(diff)
        return torch.mean(torch.where(absdiff < self.threshold, absdiff, (diff * diff)))

    def process_end_batch(self):
        pass

    def log(self, logger, name, epoch, *args, **kwargs):
        pass

##################
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


class IoU(nn.Module):
    def __init__(self, threshold, reduction="mean", eps=0.00001):
        super(IoU, self).__init__()
        self.threshold = threshold
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs, labels):
        bin_outputs = (outputs > self.threshold).float()
        intersection = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) & (labels.view(labels.shape[0], -1) == 1)).sum(dim=1)
        union = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) | (labels.view(bin_outputs.shape[0], -1) == 1)).sum(dim=1)
        out = torch.true_divide(torch.add(intersection, self.eps), torch.add(union, self.eps))
        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        elif self.reduction == 'none':
            return out
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        return


class Classification_cases(nn.Module):
    def __init__(self, threshold=0.5, reduction="mean"):
        super(Classification_cases, self).__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, outputs, labels):
        return get_classification_case_rates(self.threshold, outputs, labels, self.reduction)


class Accuracy(nn.Module):
    def __init__(self, threshold=0.5):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        return (classification_case_rates.narrow(0, 0, 1) + classification_case_rates.narrow(0, 1, 1)) / classification_case_rates.sum().item()


class Precision(nn.Module):
    def __init__(self, threshold=0.5):
        super(Precision, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        return get_precision(classification_case_rates)


class Recall(nn.Module):
    def __init__(self, threshold=0.5):
        super(Recall, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        return get_recall(classification_case_rates)


class F1_Score(nn.Module):
    def __init__(self, threshold=0.5):
        super(F1_Score, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        precision = get_precision(classification_case_rates)
        recall = get_recall(classification_case_rates)
        return 2 * (precision * recall) / (precision + recall)


class False_Pos_Rate(nn.Module):
    def __init__(self, threshold=0.5):
        super(False_Pos_Rate, self).__init__()
        self.threshold = threshold

    def forward(self, classification_case_rates):
        pass

    def calc(self, classification_case_rates):
        return get_false_pos_rate(classification_case_rates)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = nn.functional.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]




def get_classification_case_rates(threshold, outputs, labels, reduction):
    bin_outputs = (outputs > threshold).float()

    true_pos = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) & (labels.view(bin_outputs.shape[0], -1) == 1)).sum(dim=1).float()
    true_neg = ((bin_outputs.view(bin_outputs.shape[0], -1) == 0) & (labels.view(bin_outputs.shape[0], -1) == 0)).sum(dim=1).float()
    false_pos = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) & (labels.view(bin_outputs.shape[0], -1) == 0)).sum(dim=1).float()
    false_neg = ((bin_outputs.view(bin_outputs.shape[0], -1) == 0) & (labels.view(bin_outputs.shape[0], -1) == 1)).sum(dim=1).float()

    if reduction == 'mean':
        true_pos = true_pos.mean()
        true_neg = true_neg.mean()
        false_pos = false_pos.mean()
        false_neg = false_neg.mean()
    elif reduction == 'sum':
        true_pos = true_pos.sum()
        true_neg = true_neg.sum()
        false_pos = false_pos.sum()
        false_neg = false_neg.sum()
    elif reduction == 'none':
        pass
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))

    return torch.stack((true_pos, true_neg, false_pos, false_neg), dim=0)


def get_precision(classification_case_rates):
    return classification_case_rates.narrow(0, 0, 1) / (
                classification_case_rates.narrow(0, 0, 1) + classification_case_rates.narrow(0, 2, 1))


def get_recall(classification_case_rates):
    return classification_case_rates.narrow(0, 0, 1) / (classification_case_rates.narrow(0, 0, 1) + classification_case_rates.narrow(0, 3, 1))


def get_false_pos_rate(classification_case_rates):
    return classification_case_rates.narrow(0, 2, 1) / (classification_case_rates.narrow(0, 2, 1) + classification_case_rates.narrow(0, 1, 1))

