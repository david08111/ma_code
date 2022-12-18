import torch
import torch.nn as nn
import torchvision
from .efficientnet import EfficientNet
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

        pass

        # self.model = self._process_architecture_config(self.model_configs["architecture_config"])


    def forward(self, input):
        x = self.model.forward(input)

        # if "refine_lw_net" in self.net_name:
        #     x = torch.nn.functional.interpolate(x[:, :, :, 0], (x.shape[0], self.img_size, self.img_size))

        return x

