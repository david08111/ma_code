import logging
import os
from torch.utils.tensorboard import SummaryWriter

class TrainLogger(): # Wrapper for Logging to txt + TensorBoard + Wandb
    def __init__(self, name, save_path, ):
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        logging.basicConfig(os.path.join(save_path, name + ".log"), filemode='w', encoding='utf-8', level=logging.DEBUG)
        self.tb_logger = SummaryWriter(os.path.join(save_path, name + "_tb"), filename_suffix=".log")

    def add_text(self, text, level):
        logging.log(level, text)

    def add_scalar(self, name, value, epoch):
        self.tb_logger.add_scalar(name, value, epoch)

    def add_image(self, name, img, epoch):
        self.tb_logger.add_image(name, img, epoch, dataformats="NHWC")

    def flush(self):
        self.tb_logger.flush()
