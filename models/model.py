import torch
import torch.nn as nn
import torchvision
from utils import Builder
import torchvision.models as torch_models
import segmentation_models_pytorch as segmentation_models_pt

from .output_associator import EmbeddingOutputAssociatorWrapper
from .model_associater import ModelAssociater

class Model(nn.Module):
    def __init__(self, general_config, *args, **kwargs):
        super().__init__()

        self.model_configs = general_config["model"]

        self.model_architecture_config = dict(self.model_configs["model_architecture"])
        self.model_architecture_embedding_dims = self.model_architecture_config["embeddings_dims"]
        self.model_architecture_origin = self.model_architecture_config.pop("model_architecture_origin")
        self.model_architecture_name = self.model_architecture_config.pop("model_architecture_name")

        self.output_creator_list = []
        for elem in self.model_configs["output_creation"]:

            output_creator_config = dict(elem)
            output_creator_name = list(output_creator_config.keys())[0]
            output_creator_config = output_creator_config[output_creator_name]
            self.output_creator_list.append(EmbeddingOutputAssociatorWrapper(output_creator_name, **output_creator_config))

        # output_creator_config = dict(self.model_configs["output_creation"])
        # output_creator_name = list(output_creator_config.keys())[0]
        # output_creator_config = output_creator_config[output_creator_name]
        # self.output_creator = EmbeddingOutputAssociatorWrapper(output_creator_name, **output_creator_config)

        self.model = ModelAssociater.get_model_by_name(self.model_architecture_origin, self.model_architecture_name, self.model_architecture_config)

        # self.model = self._process_architecture_config(self.model_configs["architecture_config"])


    def forward(self, input):
        output_item_dict = {}

        output = self.model.forward(input)

        return output, output_item_dict

    def create_output_from_embeddings(self, outputs, dataset_category_list, annotations_data):

        return self.output_creator_list[0].create_output_from_embeddings(outputs, dataset_category_list, annotations_data)


    def create_auxiliary_output_from_embeddings(self, outputs, dataset_category_list, annotations_data):
        output_list = []
        for elem in self.output_creator_list[1:]:
            output_list.append([elem.create_association_from_embeddings(outputs, dataset_category_list, annotations_data), elem.name])
        return output_list

    # def accumulate_mean_embedding(self, outputs, masks, annotations_data, *args, **kwargs):
    #     self.output_creator_list[0].accumulate_mean_embedding(outputs, masks, annotations_data, *args, **kwargs)

