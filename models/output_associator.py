import sklearn.cluster
import torch
from panopticapi.utils import IdGenerator

class EmbeddingOutputAssociatorWrapper():
    def __init__(self, name, *args, **kwargs):
        self.output_creator = self._set_creator_by_name(name, *args, **kwargs)

    def _set_creator_by_name(self, name, *args, **kwargs):
        if name == "multi_sphere_association":
            return MultiSphereAssociator(*args, **kwargs)

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data):
        return self.output_creator.create_output_from_embeddings(outputs, dataset_category_list, annotations_data)


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
            self.hypsph_radius_map_list = list(range(self.radius_start_val, self.radius_start_val + self.radius_diff_dist * len(self.cat_id_radius_order_map_list), self.radius_diff_dist))

        self.instance_clustering_method = ClusteringWrapper(instance_cluster_method_name, **instance_clustering_method_config)
        #

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data):


        if all(x == dataset_category_list[0] for x in dataset_category_list):
            dataset_category = dataset_category_list[0]
        else:
            raise ValueError("Implementation doesnt support multiple dataset category associations!") # conversion to unified categories should work

        id_gen = IdGenerator(dataset_category)

        annotations = []
        segm_info = []

        batch_size = outputs.shape[0]

        emb_dimensions = outputs.shape[1]

        height = outputs.shape[2]
        width = outputs.shape[3]

        final_output_mask = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8) # RGB image -> make configurable

        # calc radius of embeddings

        outputs_emb_radius = torch.norm(outputs, 2, dim=1)

        for b in range(batch_size):
            outputs_emb_radius_tmp = outputs_emb_radius[b, ...]
            for i in range(len(self.cat_id_radius_order_map_list)):
                cat_id = self.cat_id_radius_order_map_list[i]
                cat_id_radius = self.hypsph_radius_map_list[i]

                radius_lw_bd = max(0, cat_id_radius - abs(self.radius_association_margin))
                radius_up_bd = cat_id_radius + abs(self.radius_association_margin)


                outputs_rad_lw_bd_indx = outputs_emb_radius_tmp > radius_lw_bd
                outputs_rad_up_bd_indx = outputs_emb_radius_tmp < radius_up_bd
                outputs_rad_mask = torch.logical_and(outputs_rad_lw_bd_indx, outputs_rad_up_bd_indx)

                # mask = original_format == el
                segment_id, color = id_gen.get_id_and_color(cat_id)

                mask = final_output_mask[b, :, outputs_rad_mask]
                area = torch.count_nonzero(outputs_rad_mask)
                if area.item() == 0:
                    continue

                color_tensor = torch.tensor(color, dtype=torch.uint8)
                color_tensor = torch.unsqueeze(color_tensor, dim=1)
                color_tensor = color_tensor.expand(3, area)
                color_tensor_assign = color_tensor.clone() # dues to issues with expand? -  might unnecessary
                test2 = final_output_mask[b, :, outputs_rad_mask]
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
                area = int(area.item())

                segm_info.append({"id": int(segment_id),
                                  "category_id": int(cat_id),
                                  "area": area})
                                  # "area": area,
                                  # "bbox": bbox,
                                  # "iscrowd": is_crowd})


        annotations.append({'image_id': annotations_data[b],
                            # 'file_name': file_name,
                            "segments_info": segm_info})

            # outputs_cat_id_association = outputs[outputs_rad_range]


class ClusteringWrapper():
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.clusterer = self._set_clusterer_by_name(name, *args, **kwargs)

    def _set_clusterer_by_name(self, name, *args, **kwargs):
        if name == "dbscan":
            return sklearn.cluster.DBSCAN(**kwargs)
        if name == "mean_shift":
            return sklearn.cluster.MeanShift(**kwargs)
        if name == "optics":
            return sklearn.cluster.OPTICS(**kwargs)
        if name == "hierarchical":
            raise ValueError("Not implemented yet!")

    def apply_clustering(self):
        raise NameError("Not implemented yet!")
