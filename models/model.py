import torch
import torch.nn as nn
import torchvision
from utils import Builder
import torchvision.models as torch_models
import segmentation_models_pytorch as segmentation_models_pt

from .model_associater import ModelAssociater

class Model(nn.Module):
    def __init__(self, general_config, *args, **kwargs):
        super().__init__()

        self.model_configs = general_config["model"]

        self.model_architecture_config = dict(self.model_configs["model_architecture"])
        self.model_architecture_origin = self.model_architecture_config.pop("model_architecture_origin")
        self.model_architecture_name = self.model_architecture_config.pop("model_architecture_name")

        self.model = ModelAssociater.get_model_by_name(self.model_architecture_origin, self.model_architecture_name, self.model_architecture_config)

        # self.model = self._process_architecture_config(self.model_configs["architecture_config"])


    def forward(self, input):
        output_dict = {}

        output_dict["final_pixel_embeddings"] = self.model.forward(input)

        return output_dict

