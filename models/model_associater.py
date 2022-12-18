import torch
import torch.nn as nn
import torchvision
from utils import Builder
import torchvision.models as torch_models
import segmentation_models_pytorch as segm_models_pt
# from segmentation_models_pytorch import *


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

segmentation_models_pytorch_map_dict = {"unet": segm_models_pt.Unet,
                                        "unet++": segm_models_pt.UnetPlusPlus,
                                        "manet": segm_models_pt.MAnet,
                                        "deeplabv3": segm_models_pt.DeepLabV3,
                                        "deeplabv3+": segm_models_pt.DeepLabV3Plus,
                                        "linknet": segm_models_pt.Linknet,
                                        "fpn": segm_models_pt.FPN,
                                        "pan": segm_models_pt.PAN,
                                        "pspnet": segm_models_pt.PSPNet
}



class ModelAssociater():
    def __int__(self):
        pass

    @staticmethod
    def get_model_by_name(model_origin, model_name, model_architecture_config):

        torchvision_model_dict = torchvision.models.__dict__.keys()

        model_architecture_config_tmp = dict(model_architecture_config)
        if model_origin == "torchvision":
            if model_name in torchvision_model_dict:   # pointless?
                return ModelAssociater._get_torchvision_backbone_by_name(model_name, model_architecture_config)
            elif model_name in torchvision_segm_model_map_dict.keys():
                if model_architecture_config_tmp.pop("pretrained"):
                    model_architecture_config_tmp["weights_backbone"] = torchvision_segm_model_weightscls_map_dict[model_name]
                model_architecture_config_tmp["num_classes"] = model_architecture_config_tmp.pop("embedding_dims")

                return torchvision_segm_model_map_dict[model_name](**model_architecture_config_tmp)
        elif model_origin == "segmentation_models_pytorch":
            """
            Reference from https://github.com/david08111/segmentation_models.pytorch
            
            Possible Config args:
                encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
                    to extract features of different spatial resolution
                encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
                    two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
                    with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
                    Default is 5
                encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
                    other pretrained weights (see table with available weights for each encoder_name)
                decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
                    Length of the list should be the same as **encoder_depth**
                decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
                    is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
                    Available options are **True, False, "inplace"**
                decoder_attention_type: Attention module used in decoder of the model. Available options are
                    **None** and **scse** (https://arxiv.org/abs/1808.08127).
                in_channels: A number of input channels for the model, default is 3 (RGB images)
                classes: A number of classes for output mask (or you can think as a number of channels of output mask)
                activation: An activation function to apply after the final convolution layer.
                    Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                        **callable** and **None**.
                    Default is **None**
                aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
                    on top of encoder if **aux_params** is not **None** (default). Supported params:
                        - classes (int): A number of classes
                        - pooling (str): One of "max", "avg". Default is "avg"
                        - dropout (float): Dropout factor in [0, 1)
                        - activation (str): An activation function to apply "sigmoid"/"softmax"
                            (could be **None** to return logits)
            """
            model_name_split_list = model_name.split("_")
            model_name_encoder = model_name_split_list[1]
            model_name_decoder = model_name_split_list[0]

            model_architecture_config_tmp["encoder_name"] = model_name_encoder
            model_architecture_config_tmp["classes"] = model_architecture_config_tmp.pop("embedding_dims")
            if model_architecture_config_tmp.pop("pretrained"):
                model_architecture_config_tmp["encoder_weights"] = "imagenet"

            if model_name_decoder in segmentation_models_pytorch_map_dict.keys():
                return segmentation_models_pytorch_map_dict[model_name_decoder](**model_architecture_config_tmp)

        # elif model_origin == "contrastive_seg":
        #     configer_dict = {}
        #     configer = contrast_seg.Configer(config_dict=configer_dict)
        #     return contrast_seg.ModelManager(configer)


    @staticmethod
    def _get_torchvision_backbone_by_name(model_name, model_architecture_config):
        torchvision_model_dict = torchvision.models.__dict__
        model = torchvision_model_dict[model_name](**model_architecture_config)

        model = ModelAssociater._remove_linear_layer(model)

        return model

    @staticmethod
    def _remove_linear_layer(model):
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

    @staticmethod
    def _get_block_module(name, block_settings, block_config):
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

        test_new = ModelAssociater._get_backbone_from_architecture(test)
        test_new2 = ModelAssociater._get_backbone_from_architecture(test2)
        test_new3 = ModelAssociater._get_backbone_from_architecture(test3)
        test_new4 = ModelAssociater._get_backbone_from_architecture(test4)
        test_new5 = ModelAssociater._get_backbone_from_architecture(test5)
        test_new6 = ModelAssociater._get_backbone_from_architecture(test6)
        test_new7 = ModelAssociater._get_backbone_from_architecture(test7)
        test_new8 = ModelAssociater._get_backbone_from_architecture(test8)
        test_new9 = ModelAssociater._get_backbone_from_architecture(test9)
        test_new10 = ModelAssociater._get_backbone_from_architecture(test10)
        test_new11 = ModelAssociater._get_backbone_from_architecture(test11)


        if name in torchvision_model_dict.keys():
            return ModelAssociater._get_torchvision_backbone_by_name(torchvision_model_dict[name](block_config))
        else:
            raise ValueError("Model Block Module " + name + " not implemented yet!")

    @staticmethod
    def _process_architecture_config(architecture_config):

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
                architecture_module_list.append(ModelAssociater._get_block_module(block_module_name, block_settings, elem[module_name][block_module_name]))
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
