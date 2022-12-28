## Deep Learning Panoptic Segmentation
This project contains the tools to process a dataset and train/evaluate different deep learning models for Panoptic Segmentation.

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
To be added

### Evaluation
For evaluation purposes several scripts can be ran.

DEPRECATED!
For inference use `predict -d INPUT_DATA_PATH -s OUTPUT_PATH -w PATH_TO_WEIGHT_FILE -t THRESHOLD_VALUE -c CONFIG_PATH [-cp] [-v]` to let the model predict according outputs and binarize them with the specified threshold.

(To be implemented:)
To calculate evaluation metrics use `evaluate -d INPUT_PATH -s SAVE_PATH -w PATH_TO_WEIGHT_FILE -c CONFIG_PATH`
