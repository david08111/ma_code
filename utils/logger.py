import logging
import os
from torch.utils.tensorboard import SummaryWriter

class TrainLogger(): # Wrapper for Logging to txt + TensorBoard + Wandb

    last_epoch_img = 0
    num_log_img_counter = 0

    def __init__(self, name, save_path, img_log_freq=10, num_log_img=20):
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        self.img_log_freq = img_log_freq
        self.num_log_img = num_log_img

        logging.basicConfig(filename=os.path.join(save_path, name + ".log"), format='Time - %(asctime)s - %(levelname)s - %(message)s', filemode='w', encoding='utf-8', level=logging.INFO, force=True)

        # self.logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # logging.setFormatter(self.logging_formatter)
        self.tb_logger = SummaryWriter(os.path.join(save_path, name + "_tb"), filename_suffix=".log")


    def add_text(self, text, level, epoch):
        log_msg = f"Epoch {epoch} - {text}"
        logging.log(level, log_msg)

    def add_scalar(self, name, value, epoch):
        log_msg = f"Epoch {epoch} - {name} - {value}"
        logging.log(logging.INFO, log_msg)
        self.tb_logger.add_scalar(name, value, epoch)


    def add_image(self, name, img, epoch):
        if self.last_epoch_img != epoch:
            self.last_epoch_img = epoch
            self.num_log_img_counter = 0

        if epoch % self.img_log_freq == 0 and self.num_log_img_counter < self.num_log_img:
            log_msg = f"Epoch {epoch} - IMG {name} LOGGED"
            logging.log(logging.DEBUG, log_msg)
            self.tb_logger.add_images(name, img, epoch, dataformats="NHWC")
            self.num_log_img_counter += 1
            # self.tb_logger.add_image

    def add_embedding(self, name, metadata, embeddings, epoch):
        self.tb_logger.add_embedding(mat=embeddings, metadata=metadata, tag=name, global_step=epoch)

    def add_figure(self):
        raise NameError("Not implemented yet!")

    def add_graph(self, model, inputs):
        self.tb_logger.add_graph(model, input_to_model=inputs)

    def add_hyperparams(self):
        raise NameError("Not implemented yet!")

    def flush(self):
        self.tb_logger.flush()

    def close(self):
        self.tb_logger.close()
