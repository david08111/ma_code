import torch
import numpy as np
import random


class EmbeddingHandler():
    def __init__(self, embedding_storage_config, embedding_sampler_config, storage_step_update_sample_size, dataset_category_dict, emb_dimensions, device, *args, **kwargs):

        # if all(x == dataset_category_list[0] for x in dataset_category_list):
        #     dataset_category = dataset_category_list[0]
        # else:
        #     raise ValueError(
        #         "Implementation doesnt support multiple dataset category associations!")  # conversion to unified categories should work

        self.dataset_categories = dataset_category_dict

        self.storage_step_update_sample_size = storage_step_update_sample_size

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
        self.embedding_sampler.sample_embeddings(batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds)

    def step_sample_embeddings2store(self, outputs, masks):
        sampled_embeddings = self.embedding_sampler.sample_embeddings2store(outputs, masks, self.storage_step_update_sample_size)
        self.embedding_storage.store_embeddings(sampled_embeddings)

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

    def get_storage_elems(self, cat_id, elem_indx_list):
        return self.embedding_storage.get_storage_elems(cat_id, elem_indx_list)

    def get_storage_size(self):
        return self.embedding_storage.get_storage_size()

    def get_storage(self):
        return self.embedding_storage.get_storage()

    def get_size(self):
        return self.embedding_storage.get_size()

    def get_num_categories(self):
        return self.embedding_storage.get_num_categories()

    def get_cat_id2indx_map(self):
        return self.embedding_storage.get_cat_id2indx_map()


class MemoryBank():
    """
        Stores embeddings in a Ringbuffer and gives directly access to specific indices
    """
    def __init__(self, num_embeddings, embedding_dims, cat_id2indx_map, *args, **kwargs):
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims

        self.cat_id2indx_map = cat_id2indx_map


        self.num_categories = len(list(self.cat_id2indx_map.keys()))
        self.num_embeds_per_cat = int(np.ceil(self.num_embeddings / self.num_categories))

        # self.storage = torch.range(0, self.num_embeds_per_cat * self.num_categories - 1)

        # self.storage = self.storage.repeat(self.num_categories, 1)
        # self.storage = self.storage.repeat(1, self.embedding_dims)
        # self.storage = self.storage.repeat(self.num_categories, 1, 1)
        # test = self.storage.view(self.num_categories, self.num_embeds_per_cat, self.embedding_dims).numpy()
        self.storage = torch.zeros((self.num_categories, self.num_embeds_per_cat, self.embedding_dims))

        self.cat_id_curr_indx = np.zeros(self.num_categories, dtype=np.int32)

    def get_storage(self):
        return self.storage

    def get_storage_elems(self, cat_id, elem_indx_list):
        cat_id_storage_indx = self.cat_id2indx_map[cat_id]
        test = self.storage.numpy()
        return self.storage[cat_id_storage_indx, elem_indx_list]

    def get_newest_elem_indx(self):
        return self.cat_id_curr_indx

    def get_num_categories(self):
        return self.num_categories

    def store_embeddings(self, new_embeddings_dict):
        """

        Args:
            embeddings: dict with cat_id as keys containing torch tensors with shape (n, embed_dims)

        Returns:

        """

        for cat_id in new_embeddings_dict:
            num_new_elems = new_embeddings_dict[cat_id].shape[0]
            cat_id_indx = self.cat_id2indx_map[cat_id]
            if num_new_elems + self.cat_id_curr_indx[cat_id_indx] >= self.num_embeds_per_cat:
                num_residual_elems = num_new_elems + self.cat_id_curr_indx[cat_id_indx] - self.num_embeds_per_cat
                self.storage[cat_id_indx, self.cat_id_curr_indx[cat_id_indx]:] = new_embeddings_dict[cat_id][:-num_residual_elems]

                self.storage[cat_id_indx, :num_residual_elems] = new_embeddings_dict[cat_id][-num_residual_elems:]

                self.cat_id_curr_indx[cat_id_indx] = num_residual_elems
            else:
                self.storage[cat_id_indx, self.cat_id_curr_indx[cat_id_indx]:(num_new_elems + self.cat_id_curr_indx[cat_id_indx])] = new_embeddings_dict[cat_id]
                # test = self.storage.numpy()
                self.cat_id_curr_indx[cat_id_indx] += num_new_elems

    def get_size(self):
        return self.storage.shape

    def get_cat_id2indx_map(self):
        return self.cat_id2indx_map

class Queue():
    """
        Stores embeddings in a Ringbuffer but gives only access to the last few embeddings (ignores specified index list)
    """
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

    def get_storage_elems(self, cat_id, elem_indx_list):
        cat_id_storage_indx = self.cat_id2indx_map[cat_id]

        return self.storage[cat_id_storage_indx, elem_indx_list]

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
                self.storage[self.cat_id2indx_map[cat_id], self.cat_id_curr_indx[cat_id]:] = new_embeddings_dict[
                                                                                                 cat_id][
                                                                                             :-num_residual_elems]

                self.storage[self.cat_id2indx_map[cat_id], :num_residual_elems] = new_embeddings_dict[cat_id][
                                                                                  -num_residual_elems:]

                self.cat_id_curr_indx[cat_id] = num_residual_elems
            else:
                self.storage[self.cat_id2indx_map[cat_id],
                self.cat_id_curr_indx[cat_id]:(num_new_elems + self.cat_id_curr_indx[cat_id])] = new_embeddings_dict[
                    cat_id]
                self.cat_id_curr_indx[cat_id] += num_new_elems

    def get_size(self):
        return self.storage.shape

    def get_num_categories(self):
        return self.num_categories

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

    def sample_embeddings2store(self, output_embeddings, masks, num_embeddings_per_cat):
        return self.embedding_sampler.sample_embeddings2store(output_embeddings, masks, num_embeddings_per_cat)

class RandomSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        batch_embeds_first_key = list(batch_embeds.keys())[0]
        embedding_dims = batch_embeds[batch_embeds_first_key][0].shape[0]
        batch_size = len(batch_embeds[batch_embeds_first_key].keys())
        num_categories = embedding_storage.get_num_categories()
        # for negatives
        perc_embedding_storage_samples = random.random()

        # num_embedding_storage_samples_neg = int(np.round(num_neg_embeds * perc_embedding_storage_samples))
        #
        # num_batch_samples_neg = num_neg_embeds - num_embedding_storage_samples_neg

        #
        num_embedding_storage_samples_pos = int(np.round(num_pos_embeds * perc_embedding_storage_samples))

        num_batch_samples_pos = num_pos_embeds - num_embedding_storage_samples_pos

        ##
        pos_embeds = torch.zeros(embedding_dims, num_pos_embeds)
        # pos_embeds_indx_list = random.sample(range(batch_embeds[cat_id][batch_index].shape[0]), k=num_batch_samples_pos)
        # pos_embeds_batch_indx_list = random.sample(batch_size, k=num_batch_samples_pos)

        # pos_embeds[:num_batch_samples_pos] = batch_embeds[cat_id][batch_index][:, pos_embeds_indx_list]
        if num_batch_samples_pos:
            pos_embeds_indx_list = [random.sample(range(batch_size), k=num_batch_samples_pos), []]
            for k in range(len(pos_embeds_indx_list[0])):
                pos_embeds_indx_list[1].append(random.sample(range(batch_embeds[cat_id][pos_embeds_indx_list[0][k]].shape[1]), k=1))
            for i in range(len(pos_embeds_indx_list[0])):
                # test1 = batch_embeds[cat_id][pos_embeds_indx_list[0][i]][:, pos_embeds_indx_list[1][i]][:, 0]
                # test2 = pos_embeds[:, i]
                pos_embeds[:, i] = batch_embeds[cat_id][pos_embeds_indx_list[0][i]][:, pos_embeds_indx_list[1][i]][:, 0]

        if num_embedding_storage_samples_pos:
            # pos_embeds_indx_list_storage = random.sample(range(embedding_storage.get_size()[1]), k=num_embedding_storage_samples_pos)
            # pos_embeds[num_batch_samples_pos:] = embedding_storage.get_storage_elems(cat_id, pos_embeds_indx_list_storage)
            # test1 = self._sample_embeddings_from_storage(cat_id, embedding_storage, num_embedding_storage_samples_pos)
            # test2 = pos_embeds[:, num_batch_samples_pos:]
            pos_embeds[:, num_batch_samples_pos:] = self._sample_embeddings_from_storage(cat_id, embedding_storage, num_embedding_storage_samples_pos)
            #

        num_embeddings_per_cat = int(np.round(num_neg_embeds / num_categories))

        num_neg_embeds_per_cat_storage = int(np.round(num_embeddings_per_cat * perc_embedding_storage_samples))

        num_neg_embeds_per_cat_batch = num_embeddings_per_cat - num_neg_embeds_per_cat_storage

        # num_neg_embeds_per_cat_storage = int(np.ceil(num_embedding_storage_samples_neg / num_categories))
        # num_neg_embeds_per_cat_batch = num_categories - num_neg_embeds_per_cat_storage

        num_neg_embeds_per_cat = num_neg_embeds_per_cat_batch + num_neg_embeds_per_cat_storage

        num_neg_embeds_residual = num_neg_embeds - num_neg_embeds_per_cat * num_categories

        neg_embeds = torch.zeros(embedding_dims, num_neg_embeds)

        indx_counter = 0
        for neg_cat_id in embedding_storage.get_cat_id2indx_map():
            if neg_cat_id == cat_id:
                continue
            # neg_embeds[:, num_neg_embeds_per_cat_storage*indx_counter:num_neg_embeds_per_cat_storage*(indx_counter + 1)] = self._sample_embeddings_from_storage(embedding_storage, num_neg_embeds_per_cat_storage, neg_cat_id)
            # indx_counter += 1
            # if indx_counter == 17:
            #     break
            neg_embeds[:, num_neg_embeds_per_cat * indx_counter:num_neg_embeds_per_cat * indx_counter + num_neg_embeds_per_cat_storage] = self._sample_embeddings_from_storage(
                neg_cat_id, embedding_storage, num_neg_embeds_per_cat_storage)

            # neg_embeds_indx_list = random.sample(range(batch_embeds[neg_cat_id][batch_index].shape[0]),
            #                                      k=num_neg_embeds_per_cat_batch)
            # neg_embeds[:, num_neg_embeds_per_cat * indx_counter + num_neg_embeds_per_cat_storage:num_neg_embeds_per_cat * (indx_counter + 1)] = batch_embeds[neg_cat_id][batch_index][:, neg_embeds_indx_list]

            # neg_embeds_indx_list = [random.sample(range(batch_size), k=num_neg_embeds_per_cat_batch),
            #                         random.sample(range(batch_embeds[neg_cat_id][batch_index].shape[0]),
            #                                       k=num_neg_embeds_per_cat_batch)]

            neg_embeds_indx_list = [[random.choice(range(batch_size)) for foo in range(num_neg_embeds_per_cat_batch)], []]
            for k in range(len(neg_embeds_indx_list[0])):
                neg_embeds_indx_list[1].append(
                    random.sample(range(batch_embeds[neg_cat_id][neg_embeds_indx_list[0][k]].shape[1]), k=1))

            for i in range(len(neg_embeds_indx_list[0])):
                neg_embeds[:, num_neg_embeds_per_cat * indx_counter + num_neg_embeds_per_cat_storage + i] = batch_embeds[neg_cat_id][neg_embeds_indx_list[0][i]][:, neg_embeds_indx_list[1][i]][:, 0]

            indx_counter += 1
            if indx_counter == 17:
                break

        residual_neg_cat_id = random.sample(list(embedding_storage.get_cat_id2indx_map().keys()), 1)
        for k in range(num_neg_embeds_residual):
            neg_embeds_indx_list_residual = [random.sample(range(batch_size), k=num_neg_embeds_residual), []]
            for l in range(len(neg_embeds_indx_list_residual[0])):
                neg_embeds_indx_list_residual[1].append(
                    random.sample(range(batch_embeds[residual_neg_cat_id][neg_embeds_indx_list_residual[0][l]].shape[1]), k=1))

        return pos_embeds, neg_embeds


    def _sample_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        embeds_storage_indx_list = random.sample(range(embedding_storage.get_size()[1]), k=num_embedding_storage_samples)

        return embedding_storage.get_storage_elems(cat_id, embeds_storage_indx_list).T

    def sample_embeddings2store(self, output_embeddings, masks, num_embeddings_per_cat):

        unique_cat_ids = torch.unique(masks[:, 1, :, :])

        # batch_size = output_embeddings.shape[0]


        outputs_reordered_tmp = torch.permute(output_embeddings, (1, 0, 2, 3))

        sampled_embeddings = {}

        for unique_cat_id in unique_cat_ids[1:]:  # skip 0
            unique_cat_id = int(unique_cat_id.item())

            outputs_indx_select = masks[:, 1, :, :] == unique_cat_id
            outputs_cat_id_embeddings = outputs_reordered_tmp[:, outputs_indx_select]


            embeds_indx_list = [random.choice(range(outputs_cat_id_embeddings.shape[1])) for foo in range(num_embeddings_per_cat)]

            sampled_embeddings[unique_cat_id] = outputs_cat_id_embeddings[:, embeds_indx_list].T


        return sampled_embeddings


class MeanSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

    def _sample_neg_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

class ImgSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

    def _sample_neg_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

class BatchSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

    def _sample_neg_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

class StorageSampler():
    def __init__(self):
        # self.k = k
        pass

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

    def _sample_neg_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

class ImgRegionSampler():
    def __init__(self):
        pass
    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

    def _sample_neg_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

class DensityBasedSampler():
    def __init__(self):
        pass
    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass

    def _sample_pos_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass

    def _sample_neg_embeddings_from_storage(self, cat_id, embedding_storage, num_embedding_storage_samples):
        pass


class CombinedSampler():
    def __init__(self, sampler_list_config):

        self.sampler_list = self.set_sampler_list(sampler_list_config)

    def set_sampler_list(self, sampler_list_config):
        for sampler_config in sampler_list_config:
            self.sampler_list.append(EmbeddingSamplerWrapper(sampler_config))

    def sample_embeddings(self, batch_embeds, embedding_storage, batch_index, cat_id, num_pos_embeds, num_neg_embeds):
        pass