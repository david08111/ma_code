import torch
import numpy as np


class EmbeddingHandler():
    def __init__(self, embedding_storage, embedding_sampler, dataset_category_dict, emb_dimensions, device):
        self.embedding_storage = EmbeddingStorageWrapper(embedding_storage)
        self.embedding_sampler = EmbeddingSamplerWrapper()

        # if all(x == dataset_category_list[0] for x in dataset_category_list):
        #     dataset_category = dataset_category_list[0]
        # else:
        #     raise ValueError(
        #         "Implementation doesnt support multiple dataset category associations!")  # conversion to unified categories should work

        self.dataset_categories = dataset_category_dict

        self.cls_mean_embeddings = {}

        for cat_id in self.dataset_categories.keys():
            self.cls_mean_embeddings[cat_id] = torch.zeros((emb_dimensions), device=device)

    def accumulate_mean_embedding(self, outputs, masks, *args, **kwargs):

        unique_cat_ids = torch.unique(masks[:, 1, :, :])  # skip segment_id=0

        outputs_reordered_tmp = torch.permute(outputs, (1, 0, 2, 3))
        # masks_reordered_tmp = torch.permute(masks, (1, 0, 2, 3))


        ##################
        for unique_cat_id in unique_cat_ids[1:]:  # skip 0
            unique_cat_id = int(unique_cat_id.item())

            outputs_indx_select = masks[:, 1, :, :] == unique_cat_id
            outputs_cat_id_embeddings = outputs_reordered_tmp[:, outputs_indx_select]
            # test = outputs_cat_id_embeddings[:, 0].detach().cpu().numpy()
            # test2 = np.multiply(test, test.T)
            # test3 = np.sum(test2)
            outputs_cat_id_embeddings_mean = torch.mean(outputs_cat_id_embeddings, dim=1)


            self.cls_mean_embeddings[unique_cat_id] = (outputs_cat_id_embeddings_mean + self.cls_mean_embeddings[unique_cat_id]) / 2


    def get_cls_mean_embeddings(self):
        return self.cls_mean_embeddings

    def get_mean_embedding(self, cat_id):
        return self.cls_mean_embeddings[cat_id]

    def set_cls_mean_embeddings(self, new_cls_mean_embeddings):
        self.cls_mean_embeddings = new_cls_mean_embeddings

    def set_mean_embeddings(self, cat_id, new_cls_mean_embedding):
        self.cls_mean_embeddings[cat_id] = new_cls_mean_embedding

class EmbeddingStorageWrapper():
    def __init__(self, emb_storage_config):
        self.embedding_storage_type = list(emb_storage_config.keys())[0]
        self.embedding_storage_config = dict(emb_storage_config[self.embedding_storage_type])

        self.embedding_storage = self.set_embedding_storage(self.embedding_storage_type, emb_storage_config)

    def set_embedding_storage(self, name, emb_storage_config):
        if name == "memory_bank":
            return MemoryBank(**emb_storage_config)
        if name == "queue":
            return Queue(**emb_storage_config)
        else:
            raise NameError(f"EmbeddingStorage {name} not implemented!")

    def sample_embeddings(self):
        return self.embedding_storage.sample_embeddings()

class MemoryBank():
    def __init__(self, k):
        self.k

    def sample_embeddings(self):
        pass

    def store_embeddings(self):
        pass

class Queue():
    def __init__(self):
        pass

    def sample_embeddings(self):
        pass

    def store_embeddings(self):
        pass

class EmbeddingSamplerWrapper():
    def __init__(self, name, emb_sampler_config):
        self.embedding_sampler_type = list(emb_sampler_config.keys())[0]
        self.embedding_sampler_config = dict(emb_sampler_config[self.embedding_sampler_type])

    def set_embedding_sampler(self, name, sampler_config):
        if name == "random_sampler":
            return RandomSampler(**sampler_config)
        else:
            raise

    def sample_embeddings(self):
        pass

class RandomSampler():
    def __init__(self):
        pass

    def sample_embeddings(self):
        pass
