from .optimizer import Optimizer_Wrapper
from .scheduler import Scheduler_Wrapper
from .loss import Metrics_Wrapper
from utils import TrainLogger
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import sys
import logging
import cv2

## implement SWA

class Net_trainer():

    def __init__(self, net, save_path, save_freq, metrics_calc_freq, num_epochs, optim_config, scheduler_config, best_eval_mode, device, criterions, **kwargs):
        self.max_epoch = num_epochs
        self.save_freq = save_freq
        self.metrics_calc_freq = metrics_calc_freq
        self.save_path = save_path
        self.optimizer = Optimizer_Wrapper(net, optim_config)
        self.scheduler = Scheduler_Wrapper(scheduler_config, self.optimizer)
        self.criterions = criterions

        self.start_epoch = 0
        self.best_eval_mode = best_eval_mode

        if "min" in best_eval_mode:
            self.best_loss_score = float('inf')
        if "max" in best_eval_mode:
            self.best_loss_score = 0

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.hyperparams_dict = self.create_hyperparams_dict(kwargs["config_dict"])

        self.train_setup(net, device, kwargs["config_dict"])


    def train_setup(self, net, device, config_dict):
        latest_state_path = None

        if os.path.isdir(self.save_path):

            files = [x for x in os.listdir(self.save_path)
                     if os.path.isfile(os.path.join(self.save_path, x))
                     and "chkpt" in x
                     and not x.endswith("_best.pth")]

            if files:
                latest_epoch = max([int(x.rsplit("_", 1)[-1].rsplit(".", 1)[0])
                                    for x in files])
                for file in files:
                    if (str(latest_epoch) + ".pth") in file:
                        latest_state_path = os.path.join(self.save_path, file)
                # sub_file_name = files[0].rsplit(str(0))[0]

                self.start_epoch = latest_epoch + 1

                # latest_state_path = os.path.join(self.save_path,
                #                                  sub_file_name + str(latest_epoch) + ".pth")
        else:
            os.makedirs(self.save_path, exist_ok=True)
            # first_run = True
        if latest_state_path:
            # state = torch.load(os.path.abspath(latest_state_path))
            self.load_checkpoint(net, latest_state_path)

            # try:
            #     net.model.load_state_dict(state["state_dict"]["model"])
            # except KeyError:
            #     try:
            #         net.model.load_state_dict(state["model"])
            #     except KeyError:
            #         net.model.load_state_dict(state)

        if "logging" in config_dict:
            self.train_logger = TrainLogger(**config_dict["logging"], img_log_freq=self.metrics_calc_freq, hyperparams_dict=self.hyperparams_dict)

        # if first_run:
        #     net.apply(weights_init)


    # def weights_init(self, m):
    #
    #

    def create_hyperparams_dict(self, config_dict):
        # import wandb
        # wandb.config(config_dict)

        # return {
        #     "model_architecture_name": config_dict["model"]["model_architecture"]["model_architecture_name"],
        #     "pretrained": config_dict["model"]["model_architecture"]["pretrained"],
        #     "embedding_dims": config_dict["model"]["model_architecture"]["embedding_dims"],
        #     "AMP": config_dict["model"]["AMP"], # does not do anything
        #     "model_architecture_name": config_dict["model"]["model_architecture"]["model_architecture_name"],
        # }
        return config_dict

    def save_checkpoint(self, net, optimizer, scheduler, epoch):
        check_pt = {
            "epoch": epoch,
            "model": net.model.state_dict(),
            "optimizer": optimizer.optimizer.state_dict(),
            "scheduler": scheduler.scheduler.state_dict(),
            "best_loss": self.best_loss_score
        }
        torch.save(check_pt, os.path.join(self.save_path, net.model_architecture_name + "_chkpt_" + str(epoch) + ".pth"))

    def save_model(self, net, best=False):
        if best:
            torch.save(net.model.state_dict(), os.path.join(self.save_path, net.model_architecture_name + "_chkpt_best.pth"))
        else:
            torch.save(net.model.state_dict(), os.path.join(self.save_path, net.model_architecture_name + ".pth"))

    def load_checkpoint(self, net, path):
        loaded_check_pt = torch.load(path)
        # self.start_epoch = loaded_check_pt["epoch"]
        net.model.load_state_dict(loaded_check_pt["model"])
        self.optimizer.optimizer.load_state_dict(loaded_check_pt["optimizer"])
        self.scheduler.scheduler.load_state_dict(loaded_check_pt["scheduler"])
        self.best_loss_score = loaded_check_pt["best_loss"]

    def load_model(self, net, path):
        net.model.load_state_dict(torch.load(path))

    def set_categories(self, data):
        self.dataset_category_dict = {}
        for key in data:
            self.dataset_category_dict[key] = []
            for dataset_cls in data[key].dataset.dataset_cls_list:
                self.dataset_category_dict[key].append(dataset_cls.categories_id)
                # break # see comment below

        # current unification assumption (all datasets same category dict mapping)


    def train_step(self, epoch, net, device, data):

        loss_sum = 0
        loss_items_sum = {}
        # metrics_sum = {}

        for key in self.criterions["criterion_metrics"].keys():
            self.criterions["criterion_metrics"][key].metric.reset()

        # net.model.to(device)
        net.model.train()

        for batch_id, datam in enumerate(tqdm(data["train_loader"], desc="Train_batch", file=sys.stdout)):
            # if batch_id < 700:
            #     continue

            self.optimizer.optimizer.zero_grad()

            # [inputs, masks, segments_id_data, annotations_data] = datam
            [inputs, masks, annotations_data] = datam


            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            # segments_id_data = segments_id_data.to(device, non_blocking=True)
            # labels = labels.narrow(3, 0, 1).contiguous()
            # inputs = inputs.permute((0, 3, 2, 1))

            outputs, output_items = net(inputs)


            # inp = inputs.cpu().numpy()[0, :, :, :]
            # inp = np.moveaxis(inp, 0, -1)
            # inp = inp.astype(np.uint8)
            # plt.imshow(inp)
            # plt.show()
            # lab = masks.cpu().numpy()[0, :, :]
            # plt.imshow(lab)
            # plt.show()

            loss, loss_items = self.criterions["criterion_train"].loss(outputs, masks, annotations_data)

            loss.backward()

            self.optimizer.optimizer.step()


            if epoch % self.metrics_calc_freq == 0:
                final_outputs, final_output_segmentation_data = net.create_output_from_embeddings(outputs, self.dataset_category_dict["train_loader"], annotations_data)
                for key in self.criterions["criterion_metrics"].keys():
                    self.criterions["criterion_metrics"][key].metric(final_outputs, masks, final_output_segmentation_data, annotations_data, categories=self.dataset_category_dict["train_loader"])

                if self.train_logger:
                    self.train_logger.add_image_and_mask(["Train", "Panoptic Masks"], inputs, masks, annotations_data,
                                                         final_outputs,
                                                         final_output_segmentation_data, epoch,
                                                         self.dataset_category_dict["train_loader"])


            loss_sum += loss.item()

            for key in loss_items.keys():
                if key not in loss_items_sum.keys():
                    loss_items_sum[key] = loss_items[key]
                else:
                    loss_items_sum[key] += loss_items[key]


            if self.train_logger:
                self.train_logger.add_text(f"STEP {batch_id}", logging.INFO, epoch)
                # self.train_logger.add_image_and_mask("Panoptic Masks", inputs, masks, annotations_data, final_outputs,
                #                                      final_output_segmentation_data, epoch,
                #                                      self.dataset_category_dict["train_loader"])
                # self.train_logger.add_image("TEST IMAGE", test_output_masks, final_output_segmentation_data, epoch)

                self.train_logger.add_embeddings(["Train", "Output-Embeddings"], output_embeddings=outputs, data_pts_names=masks, annotations_data=annotations_data, epoch=epoch)
                # self.train_logger.flush()

        loss_sum /= len(data["train_loader"])
        for key in loss_items_sum.keys():
            loss_items_sum[key] /= len(data["train_loader"])

        # for metric in metrics_sum:
        #     metrics_sum[metric] /= len(data["train_loader"])

        if epoch % self.metrics_calc_freq == 0:
            for key in self.criterions["criterion_metrics"].keys():
                self.criterions["criterion_metrics"][key].metric.process_end_batch(categories=self.dataset_category_dict["train_loader"])


        if self.train_logger:
            self.train_logger.add_text(f"Finalizing Train Step", logging.INFO, epoch)
            self.train_logger.add_text(f"Train Loss {loss_sum}", logging.INFO, epoch)
            self.train_logger.add_text(f"Learning Rate {self.optimizer.optimizer.param_groups[0]['lr']}", logging.INFO, epoch)
            self.train_logger.add_scalar(["Train", "Loss", self.criterions['criterion_train'].loss_type], loss_sum, epoch)
            self.train_logger.add_scalar("Learning Rate", self.optimizer.optimizer.param_groups[0]['lr'], epoch)

            for key in loss_items_sum.keys():
                self.train_logger.add_text(f"Train Loss Item {key} - {loss_items_sum[key]}", logging.INFO, epoch)
                self.train_logger.add_scalar(["Train", "Loss", f"Item - {key}"], loss_items_sum[key],
                                             epoch)


            if epoch % self.metrics_calc_freq == 0:
                for key in self.criterions["criterion_metrics"].keys():
                    self.criterions["criterion_metrics"][key].metric.log_metric(self.train_logger, ["Train", "Metric"], epoch, categories=self.dataset_category_dict["train_loader"])

            # metrics stuff

            self.train_logger.flush()


    def val(self, epoch, net, device, data):

        loss_sum = 0
        loss_items_sum = {}

        for key in self.criterions["criterion_metrics"].keys():
            self.criterions["criterion_metrics"][key].metric.reset()

        self.train_logger.reset_img_emb_counter()

        net.model.eval()
        with torch.no_grad():

            for batch_id, datam in enumerate(tqdm(data["val_loader"], desc="Val_batch", file=sys.stdout)):

                [inputs, masks, annotations_data] = datam

                inputs = inputs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                outputs, output_items = net(inputs)

                # test = outputs.cpu().numpy()

                # plt.imshow(inputs.cpu().numpy()[0, :, :, :])
                # plt.show()
                # plt.imshow(outputs.detach().cpu().numpy()[0, :, :, 0])
                # plt.show()
                # plt.imshow(labels.cpu().numpy()[0, :, :, 0])
                # plt.show()

                loss, loss_items = self.criterions["criterion_val"].loss(outputs, masks, annotations_data)

                if epoch % self.metrics_calc_freq == 0:
                    final_outputs, final_output_segmentation_data = net.create_output_from_embeddings(outputs,
                                                                                                      self.dataset_category_dict[
                                                                                                          "val_loader"],
                                                                                                      annotations_data)
                    for key in self.criterions["criterion_metrics"].keys():
                        self.criterions["criterion_metrics"][key].metric(final_outputs, masks,
                                                                         final_output_segmentation_data,
                                                                         annotations_data,
                                                                         categories=self.dataset_category_dict[
                                                                             "val_loader"])
                    if self.train_logger:
                        self.train_logger.add_image_and_mask(["Val", "Panoptic Masks"], inputs, masks, annotations_data,
                                                             final_outputs,
                                                             final_output_segmentation_data, epoch,
                                                             self.dataset_category_dict["val_loader"])

                loss_sum += loss.item()

                for key in loss_items.keys():
                    if key not in loss_items_sum.keys():
                        loss_items_sum[key] = loss_items[key]
                    else:
                        loss_items_sum[key] += loss_items[key]

                if self.train_logger:
                    self.train_logger.add_text(f"STEP {batch_id}", logging.INFO, epoch)
                    # self.train_logger.add_image_and_mask("Panoptic Masks", inputs, masks, annotations_data,
                    #                                      final_outputs,
                    #                                      final_output_segmentation_data, epoch,
                    #                                      self.dataset_category_dict["val_loader"])
                    # self.train_logger.add_scalar("TEST SCALAR!", 69, epoch)
                    # self.train_logger.add_image("TEST IMAGE", test_output_masks, final_output_segmentation_data, epoch)
                    # self.train_logger.add_graph(net.model, inputs)

                    self.train_logger.add_embeddings(["Val", "Output-Embeddings"], output_embeddings=outputs,
                                                     data_pts_names=masks, annotations_data=annotations_data,
                                                     epoch=epoch)
                    # self.train_logger.flush()

        loss_sum /= len(data["val_loader"])
        for key in loss_items_sum.keys():
            loss_items_sum[key] /= len(data["val_loader"])

        # for metric in metrics_sum:
        #     metrics_sum[metric] /= len(data["train_loader"])


        self.scheduler.scheduler.step(loss_sum)

        if epoch % self.metrics_calc_freq == 0:
            for key in self.criterions["criterion_metrics"].keys():
                self.criterions["criterion_metrics"][key].metric.process_end_batch(categories=self.dataset_category_dict["val_loader"])


        if self.train_logger:
            self.train_logger.add_text(f"Finalizing Val Step", logging.INFO, epoch)
            self.train_logger.add_text(f"Val Loss - {loss_sum}", logging.INFO, epoch)
            # self.train_logger.add_text(f"Learning Rate {self.optimizer.optimizer.param_groups[0]['lr']}", logging.INFO,
            #                            epoch)
            self.train_logger.add_scalar(["Val", "Loss", self.criterions['criterion_val'].loss_type], loss_sum, epoch)
            # self.train_logger.add_scalar(f"Learning Rate", self.optimizer.optimizer.param_groups[0]['lr'], epoch)
            for key in loss_items_sum.keys():
                self.train_logger.add_text(f"Val Loss Item {key} - {loss_items_sum[key]}", logging.INFO, epoch)
                self.train_logger.add_scalar(["Val", "Loss", f"Item - {key}"], loss_items_sum[key],
                                             epoch)

            # self.train_logger.add_graph(net.model, inputs)

            if epoch % self.metrics_calc_freq == 0:
                for key in self.criterions["criterion_metrics"].keys():
                    self.criterions["criterion_metrics"][key].metric.log_metric(self.train_logger, ["Val", "Metric"], epoch, categories=self.dataset_category_dict["val_loader"])

            # metrics stuff

            self.train_logger.flush()

        if "min" in self.best_eval_mode:
            if loss_sum < self.best_loss_score:
                self.best_loss_score = loss_sum
                self.save_model(net, best=True)
                # print("best_loss_score: " + str(loss_sum))
                tqdm.write("best_loss_score: " + str(loss_sum))
                self.train_logger.add_text(f"Best Validation Loss Score: {loss_sum}", logging.INFO, epoch)
        if "max" in self.best_eval_mode:
            if loss_sum > self.best_loss_score:
                self.best_loss_score = loss_sum
                self.save_model(net, best=True)
                # print("best_loss_score: " + str(loss_sum))
                tqdm.write("best_loss_score: " + str(loss_sum))
                self.train_logger.add_text(f"Best Validation Loss Score: {loss_sum}", logging.INFO, epoch)

        self.train_logger.add_graph(net.model, inputs)

    # def epoch_finish(self):
    #
    #     self.scheduler.scheduler.step()
    #


    def train(self, net, device, data):

        self.set_categories(data)
        # # # # remove !
        # ######
        # self.start_epoch -= 1
        # #########
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            tqdm.write("Epoch " + str(epoch) + ":")
            tqdm.write("-" * 50)
            # print("\nEpoch " + str(epoch) + ":")
            # print("Epoch {}:".format(epoch))
            # print("-" * 50)
            self.train_step(epoch, net, device, data)
            self.val(epoch, net, device, data)
            # self.epoch_finish()

            # print("-" * 50)
            tqdm.write("-" * 50)

            if epoch % self.save_freq == 0:
                self.save_checkpoint(net, self.optimizer, self.scheduler, epoch)



        self.train_logger.close()