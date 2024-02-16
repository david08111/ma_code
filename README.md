## Panoptic Segmentation with Pixel-wise Embeddings using Supervised Contrastive Learning
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
|model_architecture_name| see architecture listing   | sets architecture to train
|model_architecture_origin |   segmentation_models_pytorch   |     sets model architecture model_architecture_origin
|pretrained| True/False| Whether to use a possibly available pretrained model
|embedding_dims| [0, 1000]| sets the output dimensionality
|channels| 3| sets the number of input channels 
|output_creation| nearest_class_mean_association, multi_sphere_association, identity, radius| sets the method to create the segmentation output from the plain embedding_dims
|instance_clustering_method| identity, hdbscan, optics, dbscan, mean_shift| sets the clustering method used to distinguish instance embedding clusters
|optimizer|see pytorch optimizers| sets the optimizer used for training
|scheduler|see pytorch schedulers|sets the scheduler used for training
|[training] save_path|(system path) |sets the path to save model versions during training                           
|save_freq|[1, 1000000]|epoch frequency the models are saved and evaluated
|num_epochs|[1,10000000]|amount of epochs to train
|train_loss/val_loss|hierarchical_cluster_batch_contrast, hierarchical_cluster_mean_contrast, hierarchical_cluster_mean_mov_avg_update_contrast, hierarchical_cluster_all_embeds_contrast, hierarchical_cluster_batch_contrast_panoptic, hierarchical_cluster_mean_contrast_panoptic, hierarchical_cluster_mean_mov_avg_update_contrast_panoptic, spherical_contrast_panoptic, spherical_contrast_panoptic_means, spherical_contrast_panoptic_all_embeds, metric_learning_sem_segm, arcface_loss, cosface_loss, lifted_structure_loss, generalized_lifted_structure_loss, intra_pair_variance_loss, large_margin_softmax_loss, nca_loss, normalized_softmax_loss, proxy_anchor_loss, proxy_nca_loss, soft_triple_loss, sphere_face_loss, sub_center_arcface_loss, reverse_huber, reverse_huber_threshold, radius_cross_entropy_mse, sem_segm_cross_entropy, mse, l1, mse_hinged_pos, mse_hinged_neg, l1_hinged_pos, l1_hinged_neg|sets the loss used to train the model     
|datasets_file_path|(path to dataset settings file)| sets the path to the dataset setting file
|batch_size| [0, 1000000]| sets the batch batch_size
|num_workers| [0,num of cpu threads]| sets the data loading worker threads 
|load_ram|True/False|sets whether the dataset should be loaded into the RAM before the training spherical_contrast_panoptic_means
|load_orig_size|True/False|Whether the original image size is loaded or not - has to be set accordingly to the augmentation config file when images are cropped as part of the augmentation
|augmentations_file_path|(path to augmentation config file)|path to the augmentation config file
|[logging] save_path|(path to logging files)| sets path where logging savings are to be model_architecture_origin
|num_log_img|[0,1000]|amount of images to be saved with every logging procedure
|log_graph|True/False|whether to log the computational graph of the model
|wandb_config|project, entity| sets wandb-related settings for logging
|sampler|nthstep, range, cluster, voxelgrid, maxnum|sets the sampler which is used to sample embeddings from the logged images for further visualizations

#### Architecture models
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

For inference use `predict.py -d INPUT_DATA_PATH -s OUTPUT_PATH -w PATH_TO_WEIGHT_FILE -c CONFIG_PATH [-cp] [-v]` to let the model predict segmentation outputs.

To calculate evaluation metrics use `evaluate.py --gt_json_file GT_JSON_INPUT_PATH --pred_json_file PRED_JSON_INPUT_PATH --gt_folder GT_IMG_FOLDER_INPUT_PATH --pred_folder PRED_IMG_FOLDER_INPUT_PATH -s SAVE_PATH -n OUTPUT_NAME`

For inference + evaluation use `predict_and_evaluate.py -d INPUT_DATA_PATH -s OUTPUT_PATH -w PATH_TO_WEIGHT_FILE -c CONFIG_PATH [-cp] [-v]` to let the model predict segmentation outputs and save evaluation metrics to a file.

To compare and visualize the evaluation metric files from different runs use `./dataset/visualize_compare_eval_runs.py -n NAME_OF_THE_PLOT -y Y_AXIS_LABEL (repeat[-l LABEL_OF_EVAL_RUN -d EVAL_RUN_FILE_PATH]) -s OUTPUT_PATH [-t DATASET_TYPE] [-o SORT_NUMERIC_LABELS] `
