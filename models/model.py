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
        self.model_architecture_origin = self.model_architecture_config.pop("model_architecture_origin")
        self.model_architecture_name = self.model_architecture_config.pop("model_architecture_name")

        output_creator_config = dict(self.model_configs["output_creation"])
        output_creator_name = list(output_creator_config.keys())[0]
        output_creator_config = output_creator_config[output_creator_name]
        self.output_creater = EmbeddingOutputAssociatorWrapper(output_creator_name, **output_creator_config)

        self.model = ModelAssociater.get_model_by_name(self.model_architecture_origin, self.model_architecture_name, self.model_architecture_config)

        # self.model = self._process_architecture_config(self.model_configs["architecture_config"])


    def forward(self, input):
        output_dict = {}

        output_dict["final_pixel_embeddings"] = self.model.forward(input)

        return output_dict

    def create_output_from_embeddings(self, output_dict):

        return self.output_creater.create_output_from_embeddings(output_dict)





