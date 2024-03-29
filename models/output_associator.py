import sklearn.cluster
import sklearn.datasets
import torch
import numpy as np
import time
# import cuml
import hdbscan
from matplotlib import pyplot as plt
from panopticapi.utils import IdGenerator
from abc import abstractmethod, ABC
import cuml


class EmbeddingOutputAssociatorWrapper():
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.output_creator = self._set_creator_by_name(name, *args, **kwargs)

    def _set_creator_by_name(self, name, *args, **kwargs):
        if name == "identity":
            return IdentityAssocicator(*args, **kwargs)
        elif name == "multi_sphere_association":
            return MultiSphereAssociator(*args, **kwargs)
        elif name == "radius":
            return RadiusAssociator(*args, **kwargs)
        elif name == "multi_sphere_association_flexible":
            return MultiSphereAssociatorFlexible(*args, **kwargs)
        elif name == "nearest_class_mean_association":
            return NearestClassMeanAssociator(*args, **kwargs)
        else:
            raise NameError(f"EmbeddingOutputAssociator {name} not implemented!")

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):
        return self.output_creator.create_output_from_embeddings(outputs, dataset_category_list, annotations_data, *args, **kwargs)
    def create_association_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):
        return self.output_creator.create_association_from_embeddings(outputs, dataset_category_list, annotations_data, *args, **kwargs)

    # def accumulate_mean_embedding(self, outputs, masks, annotations_data, *args, **kwargs):
    #     self.output_creator.accumulate_mean_embedding(outputs, masks, annotations_data, *args, **kwargs)

class IdentityAssocicator():
    def __init__(self):
        pass

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):
        if all(x == dataset_category_list[0] for x in dataset_category_list):
            dataset_category = dataset_category_list[0]
        else:
            raise ValueError("Implementation doesnt support multiple dataset category associations!") # conversion to unified categories should work

        # id_gen = IdGenerator(dataset_category)

        annotations = []
        # segm_info = []

        batch_size = outputs.shape[0]

        # emb_dimensions = outputs.shape[1]

        height = outputs.shape[2]
        width = outputs.shape[3]

        final_output_mask = torch.zeros((3, batch_size, height, width), dtype=torch.uint8) # RGB image -> make configurable
        # final_single_output_mask = torch.zeros((batch_size,height, width), dtype=torch.float32)

        dataset_category_cat_id_sorted_list = sorted(list(dataset_category.keys()))

        for b in range(batch_size):
            # outputs_emb_radius_tmp = outputs_emb_radius[b, ...]

            segm_info = []

            id_gen = IdGenerator(dataset_category) # for every image unique - (reduces amount of to distinguisable colors)

            max_class = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            unique_classes = np.unique(max_class)

            for i in unique_classes:
                cat_id = dataset_category_cat_id_sorted_list[i]

                segment_id, color = id_gen.get_id_and_color(cat_id)

                area = int(np.count_nonzero(max_class[b, :, :]==i))
                if area == 0:
                    continue

                color_tensor = torch.tensor(color, dtype=torch.uint8)
                color_tensor = torch.unsqueeze(color_tensor, dim=1)
                color_tensor = color_tensor.expand(3, area)
                color_tensor_assign = color_tensor.clone()  # issues with expand? -  might unnecessary
                # test2 = final_output_mask[b, :, output_rad_mask_inst]
                # final_output_mask[b, :, outputs_rad_mask] = torch.tensor(color, dtype=torch.uint8)
                # test = max_class[b, :, :] == i
                final_output_mask[:, b, max_class[b, :, :] == i] = color_tensor_assign

                # # bbox computation for a segment
                # hor = torch.count_nonzero(outputs_rad_range, axis=0)
                # hor_idx = np.nonzero(hor)[0]
                # x = hor_idx[0]
                # width = hor_idx[-1] - x + 1
                # vert = np.sum(mask, axis=1)
                # vert_idx = np.nonzero(vert)[0]
                # y = vert_idx[0]
                # height = vert_idx[-1] - y + 1
                # bbox = [x, y, width, height]
                # bbox = [x, y, width, height];
                # bbox = [int(value) for value in bbox]
                #

                segm_info.append({"id": int(segment_id),
                                  "category_id": int(cat_id),
                                  "area": area})
                                  # "area": area,
                                  # "bbox": bbox,
                                  # "iscrowd": is_crowd})


            annotations.append({'image_id': annotations_data[b]["image_id"],
                                # 'file_name': file_name,
                                "segments_info": segm_info})

        final_output_mask = torch.permute(final_output_mask, (1, 0, 2, 3))
        return final_output_mask, annotations

class MultiSphereAssociator():
    def __init__(self, cat_id_radius_order_map_list, radius_diff_dist, radius_start_val, hypsph_radius_map_list=None, radius_association_margin=0.5, instance_clustering_method=None):
        self.cat_id_radius_order_map_list = cat_id_radius_order_map_list
        self.radius_diff_dist = radius_diff_dist
        self.radius_start_val = radius_start_val
        self.radius_association_margin = radius_association_margin

        instance_cluster_method_name = list(instance_clustering_method.keys())[0]
        instance_clustering_method_config = instance_clustering_method[instance_cluster_method_name]

        if hypsph_radius_map_list:
            self.hypsph_radius_map_list = hypsph_radius_map_list
        else:
            self.hypsph_radius_map_list = list(np.arange(self.radius_start_val, self.radius_start_val + self.radius_diff_dist * len(self.cat_id_radius_order_map_list), self.radius_diff_dist))

        self.instance_clustering_method = ClusteringWrapper(instance_cluster_method_name, **instance_clustering_method_config)
        #

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):


        if all(x == dataset_category_list[0] for x in dataset_category_list):
            dataset_category = dataset_category_list[0]
        else:
            raise ValueError("Implementation doesnt support multiple dataset category associations!") # conversion to unified categories should work

        # id_gen = IdGenerator(dataset_category)

        annotations = []
        # segm_info = []

        batch_size = outputs.shape[0]

        # emb_dimensions = outputs.shape[1]

        height = outputs.shape[2]
        width = outputs.shape[3]

        final_output_mask = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8) # RGB image -> make configurable
        # final_single_output_mask = torch.zeros((batch_size,height, width), dtype=torch.float32)

        # calc radius of embeddings

        outputs_emb_radius = torch.norm(outputs, 2, dim=1)
        # ## TESTDATA!!
        # # device = outputs_emb_radius.get_device()
        # outputs_emb_radius = torch.rand(outputs_emb_radius.shape, device=device)
        #
        # outputs_emb_radius = torch.mul(outputs_emb_radius, 22)
        # for k in range(len(annotations_data)):
        #     if annotations_data[k]["image_id"] == 'frankfurt_000001_038418':
        #         print("Test")
        # ## TESTDATA!!
        for b in range(batch_size):
            outputs_emb_radius_tmp = outputs_emb_radius[b, ...]

            segm_info = []

            id_gen = IdGenerator(dataset_category) # for every image unique - (reduces amount of to distinguisable colors)

            for i in range(len(self.cat_id_radius_order_map_list)):
                cat_id = self.cat_id_radius_order_map_list[i]
                cat_id_radius = self.hypsph_radius_map_list[i]

                radius_lw_bd = max(0, cat_id_radius - abs(self.radius_association_margin))
                radius_up_bd = cat_id_radius + abs(self.radius_association_margin)


                outputs_rad_lw_bd_indx = outputs_emb_radius_tmp > radius_lw_bd
                outputs_rad_up_bd_indx = outputs_emb_radius_tmp < radius_up_bd
                outputs_rad_mask = torch.logical_and(outputs_rad_lw_bd_indx, outputs_rad_up_bd_indx)

                area = torch.count_nonzero(outputs_rad_mask)
                if area.item() == 0:
                    continue

                # mask = original_format == el
                if not dataset_category[cat_id]["isthing"]:
                    segment_id, color = id_gen.get_id_and_color(cat_id)

                    # final_single_output_mask[b, outputs_rad_mask] = segment_id

                    # mask = final_output_mask[b, :, outputs_rad_mask]
                    # area = torch.count_nonzero(outputs_rad_mask)
                    # if area.item() == 0:
                    #     continue

                    color_tensor = torch.tensor(color, dtype=torch.uint8)
                    color_tensor = torch.unsqueeze(color_tensor, dim=1)
                    color_tensor = color_tensor.expand(3, area)
                    color_tensor_assign = color_tensor.clone() # dues to issues with expand? -  might unnecessary
                    # test2 = final_output_mask[b, :, outputs_rad_mask]
                    # final_output_mask[b, :, outputs_rad_mask] = torch.tensor(color, dtype=torch.uint8)
                    final_output_mask[b, :, outputs_rad_mask] = color_tensor_assign

                    # area = torch.count_nonzero(outputs_rad_mask)  # segment area computation

                    # # bbox computation for a segment
                    # hor = torch.count_nonzero(outputs_rad_range, axis=0)
                    # hor_idx = np.nonzero(hor)[0]
                    # x = hor_idx[0]
                    # width = hor_idx[-1] - x + 1
                    # vert = np.sum(mask, axis=1)
                    # vert_idx = np.nonzero(vert)[0]
                    # y = vert_idx[0]
                    # height = vert_idx[-1] - y + 1
                    # bbox = [x, y, width, height]
                    # bbox = [x, y, width, height];
                    # bbox = [int(value) for value in bbox]
                    #
                    area = int(area.item())

                    segm_info.append({"id": int(segment_id),
                                      "category_id": int(cat_id),
                                      "area": area})
                                      # "area": area,
                                      # "bbox": bbox,
                                      # "iscrowd": is_crowd})

                else:
                    cat_id_indices = outputs_rad_mask.nonzero()
                    outputs_embeddings = outputs[b, :, cat_id_indices[:, 0], cat_id_indices[:, 1]]

                    outputs_embeddings = outputs_embeddings.cpu().detach().numpy()
                    clustered_embedding_indices = self.instance_clustering_method.apply_clustering(np.transpose(outputs_embeddings))

                    cluster_ids = np.unique(clustered_embedding_indices)

                    # remove noisy samples (labelled with -1)
                    if cluster_ids[0] == -1 and cluster_ids.shape[0] > 1:
                        cluster_ids = cluster_ids[1:]

                    for j in cluster_ids:
                        segment_id, color = id_gen.get_id_and_color(cat_id)

                        # final_single_output_mask[b, output_rad_mask_inst] = segment_id

                        instance_indices = cat_id_indices[clustered_embedding_indices == j]

                        output_rad_mask_inst = torch.zeros(outputs_rad_mask.shape, dtype=torch.bool)
                        output_rad_mask_inst[instance_indices[:, 0], instance_indices[:, 1]] = True

                        area = instance_indices.shape[0]

                        color_tensor = torch.tensor(color, dtype=torch.uint8)
                        color_tensor = torch.unsqueeze(color_tensor, dim=1)
                        color_tensor = color_tensor.expand(3, area)
                        color_tensor_assign = color_tensor.clone()  # issues with expand? -  might unnecessary
                        # test2 = final_output_mask[b, :, output_rad_mask_inst]
                        # final_output_mask[b, :, outputs_rad_mask] = torch.tensor(color, dtype=torch.uint8)
                        final_output_mask[b, :, output_rad_mask_inst] = color_tensor_assign

                        # area = torch.count_nonzero(outputs_rad_mask)  # segment area computation

                        # # bbox computation for a segment
                        # hor = torch.count_nonzero(outputs_rad_range, axis=0)
                        # hor_idx = np.nonzero(hor)[0]
                        # x = hor_idx[0]
                        # width = hor_idx[-1] - x + 1
                        # vert = np.sum(mask, axis=1)
                        # vert_idx = np.nonzero(vert)[0]
                        # y = vert_idx[0]
                        # height = vert_idx[-1] - y + 1
                        # bbox = [x, y, width, height]
                        # bbox = [x, y, width, height];
                        # bbox = [int(value) for value in bbox]
                        #
                        # area = int(area.item())

                        # test = np.unique(final_output_mask[b])

                        segm_info.append({"id": int(segment_id),
                                          "category_id": int(cat_id),
                                          "area": int(area)})
                        # "area": area,
                        # "bbox": bbox,
                        # "iscrowd": is_crowd})

            # test_segment_id_info = sorted([foo["id"] for foo in segm_info])
            # test_segment_id_mask = np.zeros(final_output_mask.shape, dtype=np.uint32)
            # final_output_mask_float = final_output_mask.detach().cpu().numpy().astype(np.uint32)
            # test_segment_id_mask = final_output_mask_float[b, 0, ...] + 256 * final_output_mask_float[b, 1, ...] + 256 * 256 * final_output_mask_float[b, 2, ...]
            # test2 = np.unique(test_segment_id_mask).tolist()
            # test_3 = test2[1:]
            #
            # if test_segment_id_info != test_3:
            #     print("FAIL")

            annotations.append({'image_id': annotations_data[b]["image_id"],
                                # 'file_name': file_name,
                                "segments_info": segm_info})

        return final_output_mask, annotations
        # return final_output_mask, final_single_output_mask, annotations

class MultiSphereAssociatorFlexible():
    def __init__(self, radius, radius_association_margin=0.5, instance_clustering_method=None):
        self.radius = radius
        self.radius_association_margin = radius_association_margin

        instance_cluster_method_name = list(instance_clustering_method.keys())[0]
        instance_clustering_method_config = instance_clustering_method[instance_cluster_method_name]

        self.instance_clustering_method = ClusteringWrapper(instance_cluster_method_name, **instance_clustering_method_config)

        # self.cls_mean_embeddings = {}
        #
        # self.cls_mean_embeddings_init = False

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):
        if all(x == dataset_category_list[0] for x in dataset_category_list):
            dataset_category = dataset_category_list[0]
        else:
            raise ValueError(
                "Implementation doesnt support multiple dataset category associations!")  # conversion to unified categories should work

        # id_gen = IdGenerator(dataset_category)

        annotations = []
        # segm_info = []

        batch_size = outputs.shape[0]

        # emb_dimensions = outputs.shape[1]

        height = outputs.shape[2]
        width = outputs.shape[3]

        no_spatial_embedds = height * width

        final_output_mask = torch.zeros((batch_size, 3, height, width),
                                        dtype=torch.uint8)  # RGB image -> make configurable


        radius_lw_bd = max(0, self.radius - abs(self.radius_association_margin))
        radius_up_bd = self.radius + abs(self.radius_association_margin)

        embedding_handler = kwargs["embedding_handler"]
        cls_mean_embeddings = embedding_handler.embedding_storage.cls_mean_embeddings

        for b in range(batch_size):
            # outputs_emb_radius = torch.norm(outputs, 2, dim=1)
            # outputs_emb_radius_tmp = outputs_emb_radius[b, ...]

            segm_info = []

            id_gen = IdGenerator(
                dataset_category)  # for every image unique - (reduces amount of to distinguisable colors)

            occupied_pixel_output_mask = torch.zeros((height, width),
                                        dtype=torch.bool, device=outputs.device)

            for cat_id in cls_mean_embeddings.keys():

                cat_id_mean_emb = cls_mean_embeddings[cat_id]

                # cat_id_mean_emb_output_dims = cat_id_mean_emb.repeat(1, no_spatial_embedds).view(cat_id_mean_emb.shape[0], height, width)
                cat_id_mean_emb_output_dims = cat_id_mean_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, height, width)

                dist_mean_emb2output_emb = torch.norm(torch.sub(outputs[b], cat_id_mean_emb_output_dims), 2, dim=0)

                # ######
                # ####
                # mean_emb_test = cat_id_mean_emb.detach().cpu().numpy()
                # mean_emb_test_radius = torch.norm(cat_id_mean_emb, 2, dim=0).detach().cpu().numpy()
                # rad_dist_test = dist_mean_emb2output_emb.detach().cpu().numpy()
                # #######
                # ######

                outputs_rad_lw_bd_indx = dist_mean_emb2output_emb > radius_lw_bd
                outputs_rad_up_bd_indx = dist_mean_emb2output_emb < radius_up_bd
                outputs_rad_mask = torch.logical_and(outputs_rad_lw_bd_indx, outputs_rad_up_bd_indx)

                outputs_rad_mask = torch.logical_and(outputs_rad_mask, torch.logical_not(occupied_pixel_output_mask))

                occupied_pixel_output_mask = torch.logical_or(outputs_rad_mask, occupied_pixel_output_mask)

                area = torch.count_nonzero(outputs_rad_mask)
                if area.item() == 0:
                    continue

                # mask = original_format == el
                if not dataset_category[cat_id]["isthing"]:
                    segment_id, color = id_gen.get_id_and_color(cat_id)


                    color_tensor = torch.tensor(color, dtype=torch.uint8)
                    color_tensor = torch.unsqueeze(color_tensor, dim=1)
                    color_tensor = color_tensor.expand(3, area)
                    color_tensor_assign = color_tensor.clone()  # dues to issues with expand? -  might unnecessary
                    # test2 = final_output_mask[b, :, outputs_rad_mask]
                    # final_output_mask[b, :, outputs_rad_mask] = torch.tensor(color, dtype=torch.uint8)
                    final_output_mask[b, :, outputs_rad_mask] = color_tensor_assign

                    # area = torch.count_nonzero(outputs_rad_mask)  # segment area computation

                    # # bbox computation for a segment
                    # hor = torch.count_nonzero(outputs_rad_range, axis=0)
                    # hor_idx = np.nonzero(hor)[0]
                    # x = hor_idx[0]
                    # width = hor_idx[-1] - x + 1
                    # vert = np.sum(mask, axis=1)
                    # vert_idx = np.nonzero(vert)[0]
                    # y = vert_idx[0]
                    # height = vert_idx[-1] - y + 1
                    # bbox = [x, y, width, height]
                    # bbox = [x, y, width, height];
                    # bbox = [int(value) for value in bbox]
                    #
                    area = int(area.item())

                    segm_info.append({"id": int(segment_id),
                                      "category_id": int(cat_id),
                                      "area": area})
                    # "area": area,
                    # "bbox": bbox,
                    # "iscrowd": is_crowd})

                else:
                    cat_id_indices = outputs_rad_mask.nonzero()
                    outputs_embeddings = outputs[b, :, cat_id_indices[:, 0], cat_id_indices[:, 1]]

                    outputs_embeddings = outputs_embeddings.cpu().detach().numpy()
                    clustered_embedding_indices = self.instance_clustering_method.apply_clustering(
                        np.transpose(outputs_embeddings))

                    cluster_ids = np.unique(clustered_embedding_indices)

                    # remove noisy samples (labelled with -1)
                    if cluster_ids[0] == -1 and cluster_ids.shape[0] > 1:
                        cluster_ids = cluster_ids[1:]

                    for j in cluster_ids:
                        segment_id, color = id_gen.get_id_and_color(cat_id)

                        # final_single_output_mask[b, output_rad_mask_inst] = segment_id

                        instance_indices = cat_id_indices[clustered_embedding_indices == j]

                        output_rad_mask_inst = torch.zeros(outputs_rad_mask.shape, dtype=torch.bool)
                        output_rad_mask_inst[instance_indices[:, 0], instance_indices[:, 1]] = True

                        area = instance_indices.shape[0]

                        color_tensor = torch.tensor(color, dtype=torch.uint8)
                        color_tensor = torch.unsqueeze(color_tensor, dim=1)
                        color_tensor = color_tensor.expand(3, area)
                        color_tensor_assign = color_tensor.clone()  # issues with expand? -  might unnecessary
                        # test2 = final_output_mask[b, :, output_rad_mask_inst]
                        # final_output_mask[b, :, outputs_rad_mask] = torch.tensor(color, dtype=torch.uint8)
                        final_output_mask[b, :, output_rad_mask_inst] = color_tensor_assign

                        # area = torch.count_nonzero(outputs_rad_mask)  # segment area computation

                        # # bbox computation for a segment
                        # hor = torch.count_nonzero(outputs_rad_range, axis=0)
                        # hor_idx = np.nonzero(hor)[0]
                        # x = hor_idx[0]
                        # width = hor_idx[-1] - x + 1
                        # vert = np.sum(mask, axis=1)
                        # vert_idx = np.nonzero(vert)[0]
                        # y = vert_idx[0]
                        # height = vert_idx[-1] - y + 1
                        # bbox = [x, y, width, height]
                        # bbox = [x, y, width, height];
                        # bbox = [int(value) for value in bbox]
                        #
                        # area = int(area.item())

                        # test = np.unique(final_output_mask[b])

                        segm_info.append({"id": int(segment_id),
                                          "category_id": int(cat_id),
                                          "area": int(area)})
                        # "area": area,
                        # "bbox": bbox,
                        # "iscrowd": is_crowd})

            # test_segment_id_info = sorted([foo["id"] for foo in segm_info])
            # test_segment_id_mask = np.zeros(final_output_mask.shape, dtype=np.uint32)
            # final_output_mask_float = final_output_mask.detach().cpu().numpy().astype(np.uint32)
            # test_segment_id_mask = final_output_mask_float[b, 0, ...] + 256 * final_output_mask_float[b, 1, ...] + 256 * 256 * final_output_mask_float[b, 2, ...]
            # test2 = np.unique(test_segment_id_mask).tolist()
            # test_3 = test2[1:]
            #
            # if test_segment_id_info != test_3:
            #     print("FAIL")

            annotations.append({'image_id': annotations_data[b]["image_id"],
                                # 'file_name': file_name,
                                "segments_info": segm_info})
        # ##############
        # from panopticapi.utils import get_traceback, rgb2id
        # from matplotlib
        #
        # final_output_mask_test = rgb2id(np.moveaxis(final_output_mask.detach().cpu().numpy()[0], 0, 2))
        #
        # test = np.unique(final_output_mask_test, return_counts=True)
        #
        # ######################


        return final_output_mask, annotations

    # def accumulate_mean_embedding(self, outputs, masks, annotations_data, *args, **kwargs):
    #
    #     unique_cat_ids = torch.unique(masks[:, 1, :, :])  # skip segment_id=0
    #
    #     outputs_reordered_tmp = torch.permute(outputs, (1, 0, 2, 3))
    #     # masks_reordered_tmp = torch.permute(masks, (1, 0, 2, 3))
    #
    #
    #     ##################
    #     for unique_cat_id in unique_cat_ids[1:]:  # skip 0
    #         unique_cat_id = int(unique_cat_id.item())
    #
    #         outputs_indx_select = masks[:, 1, :, :] == unique_cat_id
    #         outputs_cat_id_embeddings = outputs_reordered_tmp[:, outputs_indx_select]
    #         # test = outputs_cat_id_embeddings[:, 0].detach().cpu().numpy()
    #         # test2 = np.multiply(test, test.T)
    #         # test3 = np.sum(test2)
    #         outputs_cat_id_embeddings_mean = torch.mean(outputs_cat_id_embeddings, dim=1)
    #
    #         if unique_cat_id not in self.cls_mean_embeddings.keys():
    #             self.cls_mean_embeddings[unique_cat_id] = outputs_cat_id_embeddings_mean
    #         else:
    #             self.cls_mean_embeddings[unique_cat_id] = (outputs_cat_id_embeddings_mean + self.cls_mean_embeddings[unique_cat_id]) / 2
    #
    #
    #
    #
    #
    # def get_cls_mean_embeddings(self):
    #     return self.cls_mean_embeddings
    #
    # def set_cls_mean_embeddings(self, new_cls_mean_embeddings):
    #     self.cls_mean_embeddings = new_cls_mean_embeddings

class RadiusAssociator():
    def __init__(self, mean_origin=False):
        self.mean_origin = mean_origin

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):
        raise NameError("Not implemented yet!")

    def create_association_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):
        return torch.unsqueeze(torch.norm(outputs, 2, dim=1), 1)

class NearestClassMeanAssociator():
    def __init__(self, instance_clustering_method=None):
        instance_cluster_method_name = list(instance_clustering_method.keys())[0]
        instance_clustering_method_config = instance_clustering_method[instance_cluster_method_name]

        self.instance_clustering_method = ClusteringWrapper(instance_cluster_method_name, **instance_clustering_method_config)


    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data, *args, **kwargs):
        if all(x == dataset_category_list[0] for x in dataset_category_list):
            dataset_category = dataset_category_list[0]
        else:
            raise ValueError(
                "Implementation doesnt support multiple dataset category associations!")  # conversion to unified categories should work

        # id_gen = IdGenerator(dataset_category)

        annotations = []
        # segm_info = []

        batch_size = outputs.shape[0]

        # emb_dimensions = outputs.shape[1]

        height = outputs.shape[2]
        width = outputs.shape[3]

        no_spatial_embedds = height * width

        final_output_mask = torch.zeros((batch_size, 3, height, width),
                                        dtype=torch.uint8)  # RGB image -> make configurable


        embedding_handler = kwargs["embedding_handler"]
        cls_mean_embeddings = embedding_handler.embedding_storage.cls_mean_embeddings

        mean_emb_cat_id_order_list = [tmp_cat_id for tmp_cat_id in cls_mean_embeddings.keys()]

        # mean_emb_tensor = torch.stack([cls_mean_embeddings[tmp_cat_id].repeat(1, no_spatial_embedds).view(cls_mean_embeddings[tmp_cat_id].shape[0], height, width) for tmp_cat_id in mean_emb_cat_id_order_list])
        mean_emb_tensor = torch.stack([cls_mean_embeddings[tmp_cat_id].unsqueeze(-1).unsqueeze(-1).expand(-1, height, width) for tmp_cat_id in mean_emb_cat_id_order_list])

        #####
        # mean_embedding_db7_ref = cls_mean_embeddings[7].detach().cpu().numpy()
        # mean_embedding_db7_single = cls_mean_embeddings[7].repeat(1, no_spatial_embedds).view(cls_mean_embeddings[7].shape[0], height, width).detach().cpu().numpy()
        # mean_embedding_db7_new = cls_mean_embeddings[7].repeat(1, width, height).detach().cpu().numpy()
        # mean_embedding_db7_exp = cls_mean_embeddings[7].unsqueeze(-1).unsqueeze(-1).expand(-1, width, height).detach().cpu().numpy()
        #
        # mean_embedding_db7 = cls_mean_embeddings[7].detach().cpu().numpy()
        # mean_embedding_db22 = cls_mean_embeddings[22].detach().cpu().numpy()
        # mean_embedding_db32 = cls_mean_embeddings[32].detach().cpu().numpy()
        # #
        # mean_emb_tensor_db = mean_emb_tensor.detach().cpu().numpy()

        #####

        for b in range(batch_size):
            # outputs_emb_radius = torch.norm(outputs, 2, dim=1)
            # outputs_emb_radius_tmp = outputs_emb_radius[b, ...]

            segm_info = []

            id_gen = IdGenerator(
                dataset_category)  # for every image unique - (reduces amount of to distinguisable colors)

            # occupied_pixel_output_mask = torch.zeros((height, width),
            #                             dtype=torch.bool, device=outputs.device)

            outputs_batch_indx_stack = torch.stack([outputs[b] for k in range(len(cls_mean_embeddings))])

            dist_mean_emb2output_emb = torch.norm(torch.sub(outputs_batch_indx_stack, mean_emb_tensor), dim=1)
            # mean_emb_tensor_test = mean_emb_tensor.detach().cpu().numpy()
            # outputs_batch_indx_test = outputs_batch_indx_stack.detach().cpu().numpy()
            # dist_test = dist_mean_emb2output_emb.detach().cpu().numpy()
            min_dist_arg = torch.argmin(dist_mean_emb2output_emb, dim=0)

            unique_indices = torch.unique(min_dist_arg)

            for unique_index in unique_indices:

                cat_id = mean_emb_cat_id_order_list[unique_index.item()]

                # cat_id_mean_emb = cls_mean_embeddings[cat_id]
                #
                # cat_id_mean_emb_output_dims = cat_id_mean_emb.repeat(1, no_spatial_embedds).view(cat_id_mean_emb.shape[0], height, width)
                #
                # dist_mean_emb2output_emb = torch.norm(torch.sub(outputs[b], cat_id_mean_emb_output_dims), 1, dim=0)

                # ######
                # ####
                # mean_emb_test = cat_id_mean_emb.detach().cpu().numpy()
                # mean_emb_test_radius = torch.norm(cat_id_mean_emb, 2, dim=0).detach().cpu().numpy()
                # rad_dist_test = dist_mean_emb2output_emb.detach().cpu().numpy()
                # #######
                # ######

                # outputs_rad_lw_bd_indx = dist_mean_emb2output_emb > radius_lw_bd
                # outputs_rad_up_bd_indx = dist_mean_emb2output_emb < radius_up_bd
                # outputs_rad_mask = torch.logical_and(outputs_rad_lw_bd_indx, outputs_rad_up_bd_indx)

                outputs_rad_mask = min_dist_arg == unique_index

                # outputs_rad_mask = torch.logical_and(outputs_rad_mask, torch.logical_not(occupied_pixel_output_mask))

                # occupied_pixel_output_mask = torch.logical_or(outputs_rad_mask, occupied_pixel_output_mask)

                area = torch.count_nonzero(outputs_rad_mask)
                if area.item() == 0:
                    continue

                # mask = original_format == el
                if not dataset_category[cat_id]["isthing"]:
                    segment_id, color = id_gen.get_id_and_color(cat_id)


                    color_tensor = torch.tensor(color, dtype=torch.uint8)
                    color_tensor = torch.unsqueeze(color_tensor, dim=1)
                    color_tensor = color_tensor.expand(3, area)
                    color_tensor_assign = color_tensor.clone()  # dues to issues with expand? -  might unnecessary
                    # test2 = final_output_mask[b, :, outputs_rad_mask]
                    # final_output_mask[b, :, outputs_rad_mask] = torch.tensor(color, dtype=torch.uint8)
                    final_output_mask[b, :, outputs_rad_mask] = color_tensor_assign

                    # area = torch.count_nonzero(outputs_rad_mask)  # segment area computation

                    # # bbox computation for a segment
                    # hor = torch.count_nonzero(outputs_rad_range, axis=0)
                    # hor_idx = np.nonzero(hor)[0]
                    # x = hor_idx[0]
                    # width = hor_idx[-1] - x + 1
                    # vert = np.sum(mask, axis=1)
                    # vert_idx = np.nonzero(vert)[0]
                    # y = vert_idx[0]
                    # height = vert_idx[-1] - y + 1
                    # bbox = [x, y, width, height]
                    # bbox = [x, y, width, height];
                    # bbox = [int(value) for value in bbox]
                    #
                    area = int(area.item())

                    segm_info.append({"id": int(segment_id),
                                      "category_id": int(cat_id),
                                      "area": area})
                    # "area": area,
                    # "bbox": bbox,
                    # "iscrowd": is_crowd})

                else:
                    cat_id_indices = outputs_rad_mask.nonzero()
                    outputs_embeddings = outputs[b, :, cat_id_indices[:, 0], cat_id_indices[:, 1]]

                    outputs_embeddings = outputs_embeddings.cpu().detach().numpy()
                    clustered_embedding_indices = self.instance_clustering_method.apply_clustering(
                        np.transpose(outputs_embeddings))

                    cluster_ids = np.unique(clustered_embedding_indices)

                    # remove noisy samples (labelled with -1)
                    if cluster_ids[0] == -1 and cluster_ids.shape[0] > 1:
                        cluster_ids = cluster_ids[1:]

                    for j in cluster_ids:
                        segment_id, color = id_gen.get_id_and_color(cat_id)

                        # final_single_output_mask[b, output_rad_mask_inst] = segment_id

                        instance_indices = cat_id_indices[clustered_embedding_indices == j]

                        output_rad_mask_inst = torch.zeros(outputs_rad_mask.shape, dtype=torch.bool)
                        output_rad_mask_inst[instance_indices[:, 0], instance_indices[:, 1]] = True

                        area = instance_indices.shape[0]

                        color_tensor = torch.tensor(color, dtype=torch.uint8)
                        color_tensor = torch.unsqueeze(color_tensor, dim=1)
                        color_tensor = color_tensor.expand(3, area)
                        color_tensor_assign = color_tensor.clone()  # issues with expand? -  might unnecessary
                        # test2 = final_output_mask[b, :, output_rad_mask_inst]
                        # final_output_mask[b, :, outputs_rad_mask] = torch.tensor(color, dtype=torch.uint8)
                        final_output_mask[b, :, output_rad_mask_inst] = color_tensor_assign

                        # area = torch.count_nonzero(outputs_rad_mask)  # segment area computation

                        # # bbox computation for a segment
                        # hor = torch.count_nonzero(outputs_rad_range, axis=0)
                        # hor_idx = np.nonzero(hor)[0]
                        # x = hor_idx[0]
                        # width = hor_idx[-1] - x + 1
                        # vert = np.sum(mask, axis=1)
                        # vert_idx = np.nonzero(vert)[0]
                        # y = vert_idx[0]
                        # height = vert_idx[-1] - y + 1
                        # bbox = [x, y, width, height]
                        # bbox = [x, y, width, height];
                        # bbox = [int(value) for value in bbox]
                        #
                        # area = int(area.item())

                        # test = np.unique(final_output_mask[b])

                        segm_info.append({"id": int(segment_id),
                                          "category_id": int(cat_id),
                                          "area": int(area)})
                        # "area": area,
                        # "bbox": bbox,
                        # "iscrowd": is_crowd})

            # test_segment_id_info = sorted([foo["id"] for foo in segm_info])
            # test_segment_id_mask = np.zeros(final_output_mask.shape, dtype=np.uint32)
            # final_output_mask_float = final_output_mask.detach().cpu().numpy().astype(np.uint32)
            # test_segment_id_mask = final_output_mask_float[b, 0, ...] + 256 * final_output_mask_float[b, 1, ...] + 256 * 256 * final_output_mask_float[b, 2, ...]
            # test2 = np.unique(test_segment_id_mask).tolist()
            # test_3 = test2[1:]
            #
            # if test_segment_id_info != test_3:
            #     print("FAIL")

            annotations.append({'image_id': annotations_data[b]["image_id"],
                                # 'file_name': file_name,
                                "segments_info": segm_info})
        # ##############
        # from panopticapi.utils import get_traceback, rgb2id
        # from matplotlib
        #
        # final_output_mask_test = rgb2id(np.moveaxis(final_output_mask.detach().cpu().numpy()[0], 0, 2))
        #
        # test = np.unique(final_output_mask_test, return_counts=True)
        #
        # ######################


        return final_output_mask, annotations


class ClusteringWrapper():
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.clusterer = self._set_clusterer_by_name(name, *args, **kwargs)

    # def _set_clusterer_by_name(self, name, *args, **kwargs):
    #     if name == "identity":
    #         return IdentityClusterer(**kwargs)
    #     if name == "dbscan":
    #         return sklearn.cluster.DBSCAN(**kwargs)
    #     if name == "mean_shift":
    #         return sklearn.cluster.MeanShift(**kwargs)
    #     if name == "optics":
    #         return sklearn.cluster.OPTICS(**kwargs)
    #     if name == "hierarchical":
    #         raise ValueError("Not implemented yet!")
    def _set_clusterer_by_name(self, name, *args, **kwargs):
        if name == "identity":
            return IdentityClusterer(**kwargs)
        if name == "dbscan":
            return DBSCANClusterer(kwargs)
        if name == "hdbscan":
            return HDBSCANClusterer(kwargs)
        if name == "mean_shift":
            return MeanShiftClusterer(kwargs)
        if name == "optics":
            return OpticsClusterer(kwargs)
        if name == "hierarchical":
            raise ValueError("Not implemented yet!")
        if name == "cuml_dbscan":
            return CUMLDBSCAN(kwargs)
        # if name == "cuml_kmeans":
        #     return CUMLKmeans
        if name == "cuml_hdbscan":
            return CUMLHDBSCAN(kwargs)
        if name == "cuml_agglomerative_clustering":
            return CUMLAgglomerativeClustering(kwargs)
        else:
            return NameError(f"Clusterer {name} not implemented yet!")

    # def apply_clustering(self, embeddings):
    #     """
    #
    #     Args:
    #         embeddings: embeddings of shape (n_samples, m_features)
    #
    #     Returns:
    #         indices of clustered embeddings
    #     """
    #
    #     # embeddings = np.array([[1, 1], [1, 2], [2, 1], [5, 5], [6, 5], [3, 1], [2, 1], [6, 6],
    #     #                            [1.6, 2.2], [3.4, 7.2], [0.4, 0.6], [0.5, 0.5], [1.5, 1.5], [6.5, 6.5],[6, 5], [5, 6]])
    #     #
    #     # embeddings = sklearn.datasets.make_circles(n_samples=300, factor=0.5, noise=0.05)[0]
    #
    #     clustering = self.clusterer.fit(embeddings)
    #
    #     # for i in np.unique(clustering.labels_):
    #     #
    #     #     plt.scatter(embeddings[clustering.labels_ == i, 0], embeddings[clustering.labels_ == i, 1], label=i)
    #     # plt.legend()
    #     # plt.show()
    #
    #     return clustering.labels_
    def apply_clustering(self, embeddings):
        """

        Args:
            embeddings: embeddings of shape (n_samples, m_features)

        Returns:
            indices of clustered embeddings
        """

        clustering_labels = self.clusterer.apply_clustering(embeddings)

        return clustering_labels

class BaseClusterer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_clustering(self, embeddings):
        " abstract method"

class DBSCANClusterer(ABC):
    def __init__(self, config_dict):
        self.clusterer = sklearn.cluster.DBSCAN(**config_dict)

    def apply_clustering(self, embeddings):
        if embeddings.shape[0] > self.clusterer.min_samples:
            clustering = self.clusterer.fit(embeddings)
            # print(f"Clustering took: {time.time() - start} seconds!")
            return clustering.labels_
        else:
            # print(f"Clustering took: {time.time() - start} seconds!")
            return np.zeros(embeddings.shape[0])

class HDBSCANClusterer(ABC):
    def __init__(self, config_dict):
        # self.clusterer = hdbscan.HDBSCAN(min_cluster_size=2, max_cluster_size=45, allow_single_cluster=True, **config_dict)
        self.clusterer = hdbscan.HDBSCAN(**config_dict)

    def apply_clustering(self, embeddings):
        if embeddings.shape[0] > self.clusterer.min_cluster_size:
            clustering = self.clusterer.fit(embeddings)

            return clustering.labels_
        else:
            # print(f"Clustering took: {time.time() - start} seconds!")
            return np.zeros(embeddings.shape[0])

class MeanShiftClusterer(ABC):
    def __init__(self, config_dict):
        self.clusterer = sklearn.cluster.MeanShift(**config_dict)

    def apply_clustering(self, embeddings):
        clustering = self.clusterer.fit(embeddings)

        return clustering.labels_


class OpticsClusterer(ABC):
    def __init__(self, config_dict):
        self.clusterer = sklearn.cluster.OPTICS(**config_dict)

    def apply_clustering(self, embeddings):
        # start = time.time()
        if embeddings.shape[0] > self.clusterer.min_cluster_size:
            clustering = self.clusterer.fit(embeddings)
            # print(f"Clustering took: {time.time() - start} seconds!")
            return clustering.labels_
        else:
            # print(f"Clustering took: {time.time() - start} seconds!")
            return np.zeros(embeddings.shape[0])

class IdentityClusterer(BaseClusterer):
    def __init__(self):
        pass

    def apply_clustering(self, embeddings):
        return np.zeros(embeddings.shape[0])

class CUMLDBSCAN(BaseClusterer):
    def __init__(self, config_dict):
        self.clusterer = cuml.DBSCAN(**config_dict)

    def apply_clustering(self, embeddings):
        if embeddings.shape[0] > self.clusterer.min_samples:
            clustering = self.clusterer.fit(embeddings)
            # print(f"Clustering took: {time.time() - start} seconds!")
            return clustering.labels_
        else:
            # print(f"Clustering took: {time.time() - start} seconds!")
            return np.zeros(embeddings.shape[0])

class CUMLHDBSCAN(BaseClusterer):
    def __init__(self, config_dict):
        self.clusterer = cuml.HDBSCAN(**config_dict)

    def apply_clustering(self, embeddings):
        if embeddings.shape[0] > self.clusterer.min_samples:
            clustering = self.clusterer.fit(embeddings)
            # print(f"Clustering took: {time.time() - start} seconds!")
            return clustering.labels_
        else:
            # print(f"Clustering took: {time.time() - start} seconds!")
            return np.zeros(embeddings.shape[0])

class CUMLAgglomerativeClustering(BaseClusterer):
    def __init__(self, config_dict):
        self.clusterer = cuml.AgglomerativeClustering(**config_dict)

    def apply_clustering(self, embeddings):
        if embeddings.shape[0] > self.clusterer.min_samples:
            clustering = self.clusterer.fit(embeddings)
            # print(f"Clustering took: {time.time() - start} seconds!")
            return clustering.labels_
        else:
            # print(f"Clustering took: {time.time() - start} seconds!")
            return np.zeros(embeddings.shape[0])

class Visualizer():
    def __init__(self):
        pass

