## Deep Learning Panoptic Segmentation
This project contains the tools to process a dataset and train/evaluate different deep learning models for Panoptic Segmentation.

### Installation
Make sure to have miniconda3 installed on your device.
To be able to run the project scripts one has to install the conda environment and set it up as interpreter environment before using scripts etc.

To install the environment type 
`conda env create -f /PATH_TO_REPOSITORY/environment.yml`

### Training
Before training the network all the hyperparameters have to be set in the general config file and all its predecessors. To start the training use `train_net -c CONFIG_PATH` 

#### Config parameter
To be added

### Evaluation
For evaluation purposes several scripts can be ran.

DEPRECATED!
For inference use `predict -d INPUT_DATA_PATH -s OUTPUT_PATH -w PATH_TO_WEIGHT_FILE -t THRESHOLD_VALUE -c CONFIG_PATH [-cp] [-v]` to let the model predict according outputs and binarize them with the specified threshold.

(To be implemented:)
To calculate evaluation metrics use `evaluate -d INPUT_PATH -s SAVE_PATH -w PATH_TO_WEIGHT_FILE -c CONFIG_PATH`
