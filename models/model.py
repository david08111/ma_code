import torch
import torch.nn as nn
import torchvision
from .efficientnet import EfficientNet
from utils import Builder


class Model(nn.Module):
    def __init__(self, general_config, *args, **kwargs):
        super().__init__()

        self.model_configs = general_config["model"]
        self.model = self._process_architecture_config(self.model_configs["architecture_config"])


    def _get_backbone_from_architecture(self, model):
        # model_list = list(model.modules())
        model_list = list(model.children())

        # model_list_new = model_list[:-1]

        feature_backbone_end_module_indx = len(model_list)
        for i in range(len(model_list) - 1, 0, -1):
            if isinstance(model_list[i], nn.AdaptiveAvgPool2d) or isinstance(model_list[i], nn.Linear):
                feature_backbone_end_module_indx = i

        ## catch cases where linear layers in last sequential elem:
        if feature_backbone_end_module_indx == len(model_list):
            feature_backbone_end_module_indx -= 1

        model_list_new = model_list[:feature_backbone_end_module_indx]

        return nn.Sequential(*model_list_new)

    def _get_block_module(self, name, block_settings, block_config):
        torchvision_model_dict = torchvision.models.__dict__
        # if "efficientnet" in name:
        #     if block_settings["pre_trained"]:
        #         return EfficientNet.from_pretrained(model_name=name, advprop=True, **block_config)
        #     else:
        #         return EfficientNet.from_name(model_name=name, **block_config)
        test = torchvision_model_dict["efficientnet_b0"]()
        test2 = torchvision_model_dict["resnet50"]()
        test3 = torchvision_model_dict["inception_v3"]()
        test4 = torchvision_model_dict["densenet161"]()
        test5 = torchvision_model_dict["mnasnet1_0"]()
        test6 = torchvision_model_dict["regnet_y_128gf"]()
        test7 = torchvision_model_dict["resnext101_32x8d"]()
        test8 = torchvision_model_dict["wide_resnet101_2"]()
        test9 = torchvision_model_dict["shufflenet_v2_x1_5"]()
        test10 = torchvision_model_dict["squeezenet1_1"]()
        test11 = torchvision_model_dict["vgg16_bn"]()

        from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50

        test12 = fcn_resnet50()
        test13 = deeplabv3_resnet50()

        test_new = self._get_backbone_from_architecture(test)
        test_new2 = self._get_backbone_from_architecture(test2)
        test_new3 = self._get_backbone_from_architecture(test3)
        test_new4 = self._get_backbone_from_architecture(test4)
        test_new5 = self._get_backbone_from_architecture(test5)
        test_new6 = self._get_backbone_from_architecture(test6)
        test_new7 = self._get_backbone_from_architecture(test7)
        test_new8 = self._get_backbone_from_architecture(test8)
        test_new9 = self._get_backbone_from_architecture(test9)
        test_new10 = self._get_backbone_from_architecture(test10)
        test_new11 = self._get_backbone_from_architecture(test11)


        if name in torchvision_model_dict.keys():
            return self._get_backbone_from_architecture(torchvision_model_dict[name](block_config))
        else:
            raise ValueError("Model Block Module " + name + " not implemented yet!")

    def _process_architecture_config(self, architecture_config):

        default_module_settings = {
            "pre_trained": False
        }

        architecture_module_list = []

        torch_nn_builder = Builder(torch.nn.__dict__)

        for elem in architecture_config["architecture"]["modules"]:
            test = list(elem.items())
            module_name = list(elem.keys())[0]

            if module_name == "block_module":
                block_settings = elem[module_name].pop("settings") if "settings" in list(
                    elem[module_name].keys()) else default_module_settings
                block_module_name = list(elem[module_name].keys())[0]
                architecture_module_list.append(self._get_block_module(block_module_name, block_settings, elem[module_name][block_module_name]))
            elif module_name == "native_module":
                for native_item in elem[module_name]:
                    # module = native_item
                    assert len(native_item) == 1
                    name, kwargs = list(native_item.items())[0]
                    if kwargs is None:
                        kwargs = {}
                    args = kwargs.pop("args", [])
                    architecture_module_list.append(torch_nn_builder(name, *args, **kwargs))

            else:
                raise ValueError("Module name " + module_name + " not existent!")

        return nn.Sequential(*architecture_module_list)

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

