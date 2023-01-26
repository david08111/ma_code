## Deep Learning Panoptic Segmentation
This project contains the tools to process a dataset and train/evaluate different deep learning models for Panoptic Segmentation with contrastive embedding learning.

### Installation
Make sure to have miniconda3 installed on your device and having cloned the submodules completely
To be able to run the project scripts one has to install the conda environment and set it up as interpreter environment before using scripts etc.

To install the environment type 
`conda env create -f /PATH_TO_REPOSITORY/environment.yml`

Then install the proper torch, torchvision and torchaudio version using pip - (switch to full version in environment.yml or pytorch 2.0) 

`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`

Finally, install the submodules as packages with
`pip install -e PATH_TO_SUBMODULE`

### Training
Before training the network all the hyperparameters have to be set in the general config file and all its predecessors. To start the training use `train_net.py -c CONFIG_PATH` 

#### Config parameter

| Config Parameter | Possible values | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
|-----------------------------------|-------------|----------------|
|model_architecture_name| u_net, u_net++, MAnet , Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+ | dpn68, (dpn68b), (dpn92), dpn98, (dpn107), dpn131, vgg11, vgg13, vgg16, vgg19, vgg11_bn vgg13_bn, vgg16_bn, vgg19_bn, efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7, ineceptionresnetv2, densenet121, densenet161, densenet169, densenet201, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, (resnext101_32x16d), (resnext101_32x32d), (resnext101_32x48d), senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d    |
| ema_net                            | resnet50, resnet101, resnet152                                                                                                                                                                                       

Following architectures can be trained
| General architecture             | Variants                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| u_net|  u_net, u_net++, MAnet , Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3+ | dpn68, (dpn68b), (dpn92), dpn98, (dpn107), dpn131, vgg11, vgg13, vgg16, vgg19, vgg11_bn vgg13_bn, vgg16_bn, vgg19_bn, efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7, ineceptionresnetv2, densenet121, densenet161, densenet169, densenet201, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, (resnext101_32x16d), (resnext101_32x32d), (resnext101_32x48d), senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d    |
| ema_net                            | resnet50, resnet101, resnet152                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| refine_net                         | resnet101                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| refine_lw_net            | resnet50, resnet101, resnet152, mobilenetv2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| gcn                                | resnet152                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| seg_net                            | vgg19_bn                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| e_net                              | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| deeplabv3, deeplabv3+              | resnet50, resnet101, resnet152, mobilenetv2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

### Evaluation
For evaluation purposes several scripts can be ran.

DEPRECATED!
For inference use `predict -d INPUT_DATA_PATH -s OUTPUT_PATH -w PATH_TO_WEIGHT_FILE -t THRESHOLD_VALUE -c CONFIG_PATH [-cp] [-v]` to let the model predict according outputs and binarize them with the specified threshold.

(To be implemented:)
To calculate evaluation metrics use `evaluate -d INPUT_PATH -s SAVE_PATH -w PATH_TO_WEIGHT_FILE -c CONFIG_PATH`
