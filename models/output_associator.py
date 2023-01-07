import sklearn.cluster

class EmbeddingOutputAssociatorWrapper():
    def __init__(self, name, *args, **kwargs):
        self.output_creator = self._set_creator_by_name(name, *args, **kwargs)

    def _set_creator_by_name(self, name, *args, **kwargs):
        if name == "multi_sphere_association":
            return MultiSphereAssociator(*args, **kwargs)

    def create_output_from_embeddings(self, output_dict):
        return self.output_creator.create_output_from_embeddings(output_dict)


class MultiSphereAssociator():
    def __init__(self, cat_id_radius_order_map_list, radius_diff_dist, radius_start_val, soft_association_margin, instance_clustering_method):
        self.cat_id_radius_order_map_list = cat_id_radius_order_map_list
        self.radius_diff_dist = radius_diff_dist
        self.radius_start_val = radius_start_val
        self.soft_association_margin = soft_association_margin

        instance_cluster_method_name = list(instance_clustering_method.keys())[0]
        instance_clustering_method_config = instance_clustering_method[instance_cluster_method_name]

        self.instance_clustering_method = ClusteringWrapper(instance_cluster_method_name, **instance_clustering_method_config)
        #

    def create_output_from_embeddings(self, output_dict):
        raise NameError("Not implemented yet!")


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
