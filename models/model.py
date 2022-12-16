import torch
import torch.nn as nn
import torchvision
from .efficientnet import EfficientNet
from utils import Builder
import torchvision.models as torch_models
import segmentation_models_pytorch as segmentation_models_pt

torchvision_segm_model_map_dict = {"deeplabv3_mobilenet_v3_large": torch_models.segmentation.deeplabv3_mobilenet_v3_large,
                                "deeplabv3_resnet50": torch_models.segmentation.deeplabv3_resnet50,
                                "deeplabv3_resnet101": torch_models.segmentation.deeplabv3_resnet101,
                                "fcn_resnet50": torch_models.segmentation.fcn_resnet50,
                                "fcn_resnet101": torch_models.segmentation.fcn_resnet101,
                               "lraspp_mobilenet_v3_large": torch_models.segmentation.lraspp_mobilenet_v3_large}
torchvision_segm_model_weightscls_map_dict = {"deeplabv3_mobilenet_v3_large": torch_models.MobileNet_V3_Large_Weights,
                                "deeplabv3_resnet50": torch_models.ResNet50_Weights,
                                "deeplabv3_resnet101": torch_models.ResNet101_Weights,
                                "fcn_resnet50": torch_models.ResNet50_Weights,
                                "fcn_resnet101": torch_models.ResNet101_Weights,
                               "lraspp_mobilenet_v3_large": torch_models.MobileNet_V3_Large_Weights}

class Model(nn.Module):
    def __init__(self, general_config, *args, **kwargs):
        super().__init__()

        self.model_configs = general_config["model"]

        self.model_architecture_config = dict(self.model_configs["model_architecture"])
        self.model_architecture_name = self.model_architecture_config.pop("model_architecture_name")

        self.model = self._get_model_by_name(self.model_architecture_name, self.model_architecture_config)

        pass

        # self.model = self._process_architecture_config(self.model_configs["architecture_config"])

    def _get_model_by_name(self, model_name, model_architecture_config):

        torchvision_model_dict = torchvision.models.__dict__.keys()

        model_architecture_config_tmp = dict(model_architecture_config)
        if model_name in torchvision_model_dict:   # pointless?
            return self._get_torchvision_backbone_by_name(model_name)
        elif model_name in torchvision_segm_model_map_dict.keys():
            if model_architecture_config_tmp.pop("pretrained"):
                model_architecture_config_tmp["weights_backbone"] = torchvision_segm_model_weightscls_map_dict[model_name]
            model_architecture_config_tmp["num_classes"] = model_architecture_config_tmp.pop("embedding_dims")

            return torchvision_segm_model_map_dict[model_name](**model_architecture_config_tmp)


    def _get_torchvision_backbone_by_name(self, model_name):
        torchvision_model_dict = torchvision.models.__dict__
        model = torchvision_model_dict[model_name]()

        model = self._remove_linear_layer(model)

        return model

    def _remove_linear_layer(self, model):
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



        # test12 = fcn_resnet50()
        # test13 = deeplabv3_resnet50()

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
            return self._get_torchvision_backbone_by_name(torchvision_model_dict[name](block_config))
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

