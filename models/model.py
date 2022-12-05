import torch
import torch.nn as nn
from .efficientnet import EfficientNet
from .builder import Builder


# class Model(nn.Module):
#     def __init__(self, net, channels, classes, img_size, architecture_config):
#         super().__init__()
#         self.channels = channels
#         self.img_size = img_size
#         self.classes = classes
#         self.net_name = net
#
#         if "efficientnet" in net:
#             self.model = EfficientNet.from_pretrained(model_name=net, num_classes=classes, image_size=img_size, advprop=True)
class Model(nn.Module):
    def __init__(self, architecture_config, channels, classes, img_size, architecture_config):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        self.classes = classes
        self.net_name = net

        if "efficientnet" in net:
            self.model = EfficientNet.from_pretrained(model_name=net, num_classes=classes, image_size=img_size, advprop=True)


    def _process_architecture_config(self, config_dict):
        pass

    def forward(self, input):
        x = self.model.forward(input)

        # if "refine_lw_net" in self.net_name:
        #     x = torch.nn.functional.interpolate(x[:, :, :, 0], (x.shape[0], self.img_size, self.img_size))

        return x

    def create_final_outputs(self, outputs, focal_length, net_input_img_size):
        # relcam_depth = outputs[:, 0].clone()
        # outputs[:, 0] = outputs[:, 0] * focal_length
        final_outputs = outputs.clone()
        final_outputs[:, 1] = (outputs[:, 1] - 0.5) * 180
        final_outputs[:, 2] = outputs[:, 2] * net_input_img_size
        final_outputs[:, 3] = outputs[:, 3] * net_input_img_size
        final_outputs[:, 4] = outputs[:, 4] * net_input_img_size * outputs[:, 0]
        final_outputs[:, 5] = outputs[:, 5] * net_input_img_size * outputs[:, 0]
        final_outputs[:, 6] = outputs[:, 6] * net_input_img_size * outputs[:, 0]
        # outputs[:, 0] = outputs[:, 0] * focal_length
        final_outputs[:, 0] = outputs[:, 0] * focal_length # CHANGE BACK TO ABOVE VARIANT DUE TO RELDEPTH


        return final_outputs

        # @staticmethod
    # def closure(model, data_dict: dict, optimizers: dict, criterions={}, metrics={}, fold=0, **kwargs):
    #     pass

