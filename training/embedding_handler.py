import torch
import numpy as np
import random


class EmbeddingHandler():
    def __init__(self, embedding_storage_config, embedding_sampler_config, dataset_category_dict, emb_dimensions, device, *args, **kwargs):

        # if all(x == dataset_category_list[0] for x in dataset_category_list):
        #     dataset_category = dataset_category_list[0]
        # else:
        #     raise ValueError(
        #         "Implementation doesnt support multiple dataset category associations!")  # conversion to unified categories should work

        self.dataset_categories = dataset_category_dict

        embedding_storage_config_tmp = dict(embedding_storage_config)

        embedding_storage_type = list(embedding_storage_config_tmp.keys())[0]

        embedding_storage_config_tmp[embedding_storage_type]["dataset_categories"] = self.dataset_categories
        embedding_storage_config_tmp[embedding_storage_type]["embedding_dims"] = emb_dimensions


        self.embedding_storage = EmbeddingStorageWrapper(embedding_storage_config_tmp)

        self.embedding_sampler = EmbeddingSamplerWrapper(embedding_sampler_config)

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

    def store_embeddings(self, embeddings):
        self.embedding_storage.store_embeddings(embeddings)

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        self.embedding_sampler(batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds)

class EmbeddingStorageWrapper():
    def __init__(self, emb_storage_config):
        self.embedding_storage_type = list(emb_storage_config.keys())[0]
        self.embedding_storage_config = dict(emb_storage_config[self.embedding_storage_type])

        self.embedding_storage_config["cat_id2indx_map"] = self.set_cat_id2indx_map(self.embedding_storage_config.pop("dataset_categories"))

        self.embedding_storage = self.set_embedding_storage(self.embedding_storage_type, self.embedding_storage_config)

    def set_embedding_storage(self, name, emb_storage_config):
        if name == "memory_bank":
            return MemoryBank(**emb_storage_config)
        if name == "queue":
            return Queue(**emb_storage_config)
        else:
            raise NameError(f"EmbeddingStorage {name} not implemented!")

    def set_cat_id2indx_map(self, dataset_category):
        cat_id2indx_map = {}
        indx_counter = 0
        for cat_id in dataset_category.keys():
            cat_id2indx_map[cat_id] = indx_counter
            indx_counter += 1

        return cat_id2indx_map

    # def sample_embeddings(self):
    #     return self.embedding_storage.sample_embeddings()

    def store_embeddings(self, embeddings):
        self.embedding_storage.store_embeddings(embeddings)

    def get_storage_size(self):
        return self.embedding_storage.get_storage_size()


class MemoryBank():
    def __init__(self, num_embeddings, embedding_dims, cat_id2indx_map, *args, **kwargs):
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims

        self.cat_id2indx_map = cat_id2indx_map


        self.num_categories = len(list(self.cat_id2indx_map.keys()))
        self.num_embeds_per_cat = int(np.ceil(self.num_embeddings / self.num_categories))

        self.storage = torch.zeros((self.num_categories, self.num_embeds_per_cat, self.embedding_dims))

        self.cat_id_curr_indx = torch.zeros(self.num_categories)

    def get_storage(self):
        return self.storage

    def get_newest_elem_indx(self):
        return self.cat_id_curr_indx

    def store_embeddings(self, new_embeddings_dict):
        """

        Args:
            embeddings: dict with cat_id as keys containing torch tensors with shape (n, embed_dims)

        Returns:

        """

        for cat_id in new_embeddings_dict:
            num_new_elems = new_embeddings_dict[cat_id].shape[0]
            if num_new_elems + self.cat_id_curr_indx[cat_id] >= self.num_embeds_per_cat:
                num_residual_elems = num_new_elems + self.cat_id_curr_indx[cat_id] - self.num_embeds_per_cat
                self.storage[self.cat_id2indx_map[cat_id], self.cat_id_curr_indx[cat_id]:] = new_embeddings_dict[cat_id][:-num_residual_elems]

                self.storage[self.cat_id2indx_map[cat_id], :num_residual_elems] = new_embeddings_dict[cat_id][-num_residual_elems:]

                self.cat_id_curr_indx[cat_id] = num_residual_elems
            else:
                self.storage[self.cat_id2indx_map[cat_id], self.cat_id_curr_indx[cat_id]:(num_new_elems + self.cat_id_curr_indx[cat_id])] = new_embeddings_dict[cat_id]
                self.cat_id_curr_indx[cat_id] += num_new_elems

    def get_storage_size(self):
        return self.storage

class Queue():
    def __init__(self, num_embeddings, embedding_dims, cat_id2indx_map, *args, **kwargs):
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims

        self.cat_id2indx_map = cat_id2indx_map

        self.storage = torch.zeros((self.num_embeddings, self.embedding_dims))

    def get_storage(self):
        return self.storage

    def store_embeddings(self, embeddings):
        pass

class EmbeddingSamplerWrapper():
    def __init__(self, emb_sampler_config):
        self.embedding_sampler_type = list(emb_sampler_config.keys())[0]
        self.embedding_sampler_config = dict(emb_sampler_config[self.embedding_sampler_type])

        self.embedding_sampler = self.set_embedding_sampler(self.embedding_sampler_type, self.embedding_sampler_config)

    def set_embedding_sampler(self, name, sampler_config):
        if name == "random_sampler":
            return RandomSampler(**sampler_config)
        elif name == "mean_sampler":
            return MeanSampler(**sampler_config)
        elif name == "img_sampler":
            return ImgSampler(**sampler_config)
        elif name == "batch_sampler":
            return BatchSampler(**sampler_config)
        elif name == "storage_sampler":
            return StorageSampler(**sampler_config)
        elif name == "combined_sampler":
            return CombinedSampler(**sampler_config)
        else:
            raise

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        return self.embedding_sampler.sample_embeddings(batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds)

class RandomSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):

        embedding_dims = batch_embeds[0][0].shape[0]
        # for negatives
        perc_embedding_storage_samples = random.random()

        num_embedding_storage_samples = np.round(num_neg_embeds * perc_embedding_storage_samples)

        num_batch_samples = num_neg_embeds - num_embedding_storage_samples

        ##
        pos_embeds_indx_list = random.sample(range(batch_embeds[cat_id][batch_index].shape[0]), k=num_pos_embeds)
        pos_embeds = batch_embeds[cat_id][batch_index][:, pos_embeds_indx_list]

        #

        neg_embeds = torch.zeros(embedding_dims, num_neg_embeds)

        neg_embeds[:, :num_embedding_storage_samples] = self._sample_pos_embeddings_from_storage(embedding_storage)



        return pos_embeds, neg_embeds


    def _sample_pos_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        embeds_storage_indx_list = random.sample(range(batch_embeds[cat_id][batch_index].shape[0]), k=num_pos_embeds)

    def _sample_neg_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        embeds_storage_indx_list = random.sample(range(batch_embeds[cat_id][batch_index].shape[0]), k=num_pos_embeds)

class MeanSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

    def _sample_neg_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

class ImgSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

    def _sample_neg_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

class BatchSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

    def _sample_neg_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

class StorageSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

    def _sample_neg_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

class ImgRegionSampler():
    def __init__(self):
        pass
    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass

    def _sample_neg_embeddings_from_storage(self, embedding_storage, num_embedding_storage_samples, cat_id):
        pass


class CombinedSampler():
    def __init__(self, sampler_list_config):

        self.sampler_list = self.set_sampler_list(sampler_list_config)

    def set_sampler_list(self, sampler_list_config):
        for sampler_config in sampler_list_config:
            self.sampler_list.append(EmbeddingSamplerWrapper(sampler_config))

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass