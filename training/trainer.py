from .optimizer import Optimizer_Wrapper
from .scheduler import Scheduler_Wrapper
from .loss import Metrics_Wrapper
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import sys
import cv2

## implement SWA

class Net_trainer():

    log_num_img = 5
    log_img_save_freq = 20

    def __init__(self, net, save_path, save_freq, num_epochs, optim_config, scheduler_config, best_eval_mode, device, criterions, **kwargs):
        self.max_epoch = num_epochs
        self.save_freq = save_freq
        self.save_path = save_path
        self.optimizer = Optimizer_Wrapper(net, optim_config)
        self.scheduler = Scheduler_Wrapper(scheduler_config, self.optimizer)
        self.criterions = criterions


        metric_threshold_step_sizes = []
        metric_num_thresholds = []
        # for metric in self.criterions["criterion_metrics"]:
        #     if "num_thresholds" in self.criterions["criterion_metrics"][metric].metric_config.keys():
        #         metric_num_thresholds.append(self.criterions["criterion_metrics"][metric].metric_config["num_thresholds"])
        #
        # if all(num_threshold == metric_num_thresholds[0] for num_threshold in metric_num_thresholds):
        #     common_num_threshold = metric_num_thresholds[0]
        # else:
        #     common_num_threshold = metric_num_thresholds[0]
        #     print("Different num thresholds for classification detected. Num threshold: " + common_num_threshold + " is used")
        #
        # for metric in self.criterions["criterion_metrics"]:
        #     if "accuracy" in self.criterions["criterion_metrics"][metric].metric_type or "precision" in self.criterions["criterion_metrics"][metric].metric_type or "recall" in self.criterions["criterion_metrics"][metric].metric_type or "f1_score" in self.criterions["criterion_metrics"][metric].metric_type or "false_pos_rate" in self.criterions["criterion_metrics"][metric].metric_type:
        #         self.criterions["criterion_metrics"]["metric_additional"] = { "threshold_" + str(threshold): Metrics_Wrapper({"metric_type": "class_cases",
        #                                                                                                                  "threshold": threshold}) for threshold in np.arange(0, 1, 1 / common_num_threshold)}
        #         break

        self.start_epoch = 0
        self.best_eval_mode = best_eval_mode

        if "min" in best_eval_mode:
            self.best_loss_score = float('inf')
        if "max" in best_eval_mode:
            self.best_loss_score = 0

        # torch.backends.cudnn.enabled = False

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

        if config_dict["logging"]["enable"]:
            if config_dict["logging"]["save_path"]:
                if not os.path.isdir(config_dict["logging"]["save_path"]):
                    os.makedirs(config_dict["logging"]["save_path"], exist_ok=True)
                self.log_writer = SummaryWriter(config_dict["logging"]["save_path"], filename_suffix=".log")


        # if first_run:
        #     net.apply(weights_init)


    # def weights_init(self, m):
    #
    #

    def save_checkpoint(self, net, optimizer, scheduler, epoch):
        check_pt = {
            "epoch": epoch,
            "model": net.model.state_dict(),
            "optimizer": optimizer.optimizer.state_dict(),
            "scheduler": scheduler.scheduler.state_dict(),
            "best_loss": self.best_loss_score
        }
        torch.save(check_pt, os.path.join(self.save_path, net.net_name + "_chkpt_" + str(epoch) + ".pth"))

    def save_model(self, net, best=False):
        if best:
            torch.save(net.model.state_dict(), os.path.join(self.save_path, net.net_name + "_chkpt_best.pth"))
        else:
            torch.save(net.model.state_dict(), os.path.join(self.save_path, net.name + ".pth"))

    def load_checkpoint(self, net, path):
        loaded_check_pt = torch.load(path)
        # self.start_epoch = loaded_check_pt["epoch"]
        net.model.load_state_dict(loaded_check_pt["model"])
        self.optimizer.optimizer.load_state_dict(loaded_check_pt["optimizer"])
        self.scheduler.scheduler.load_state_dict(loaded_check_pt["scheduler"])
        self.best_loss_score = loaded_check_pt["best_loss"]

    def load_model(self, net, path):
        net.model.load_state_dict(torch.load(path))

    def train_step(self, epoch, net, device, data):

        log_img_counter = 0

        loss_sum = 0
        # metrics_sum = {}
        # eval_metrics = {}
        # for metric in self.criterions["criterion_metrics"]:
        #     if metric != "metric_additional":
        #         if self.criterions["criterion_metrics"][metric].metric_type != "accuracy" and self.criterions["criterion_metrics"][metric].metric_type != "precision" and self.criterions["criterion_metrics"][metric].metric_type != "recall" and self.criterions["criterion_metrics"][metric].metric_type != "f1_score" and self.criterions["criterion_metrics"][metric].metric_type != "false_pos_rate":
        #             metrics_sum[metric] = 0
        #     else:
        #         if epoch % Net_trainer.log_img_save_freq == 0:
        #             for threshold in self.criterions["criterion_metrics"][metric]:
        #                 metrics_sum[threshold] = 0
        # net.model.to(device)
        net.model.train()

        for batch_id, datam in enumerate(tqdm(data["train_loader"], desc="Train_batch", file=sys.stdout)):

            self.optimizer.optimizer.zero_grad()

            [inputs, masks, segments_id_data, annotations_data] = datam

            # inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
            # labels = torch.tensor(labels, device=device, dtype=torch.float32)[:, :, :, 0:1]

            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            segments_id_data = segments_id_data.to(device, non_blocking=True)
            # labels = labels.narrow(3, 0, 1).contiguous()
            # inputs = inputs.permute((0, 3, 2, 1))

            outputs = net.model(inputs)

            # test = outputs.detach().cpu().numpy()
            # test2 = labels.detach().cpu().numpy()

            # inp = inputs.cpu().numpy()[0, :, :, :]
            # inp = np.moveaxis(inp, 0, -1)
            # inp = inp.astype(np.uint8)
            # plt.imshow(inp)
            # plt.show()
            # lab = masks.cpu().numpy()[0, :, :]
            # plt.imshow(lab)
            # plt.show()

            loss = self.criterions["criterion_train"].loss(outputs, masks, segments_id_data)

            # final_outputs = net.create_final_outputs(outputs, self.cam_config["fx"], inputs.shape[2])

            # if loss.get_device() is not device:
            #     pass
            # print("labels:")
            # print(labels)
            # print("outputs:")
            # print(outputs)
            # print("loss:")
            # print(loss)
            loss.backward()

            self.optimizer.optimizer.step()

            final_outputs = net.create_output_from_embeddings(outputs)

            loss_sum += loss.item()

            # for metric in metrics_sum:
            #     # if metric != "accuracy" and metric != "precision" and metric != "recall" and metric != "f1_score" and metric != "false_pos_rate":
            #     if "threshold" not in metric:
            #         metrics_sum[metric] += self.criterions["criterion_metrics"][metric].metric(outputs, labels, final_outputs=final_outputs, crop_pos_list=crop_pos_list, cam_config=self.cam_config,
            #                                              cam_extr_list=cam_extr_list, input_img_size=inputs.shape[2])
            #     else:
            #         if epoch % Net_trainer.log_img_save_freq == 0:
            #             metrics_sum[metric] += self.criterions["criterion_metrics"]["metric_additional"][metric].metric(outputs, labels)

            if self.log_writer:
                if epoch % Net_trainer.log_img_save_freq == 0 and log_img_counter < Net_trainer.log_num_img:
                    # visualized_img = visualize_3d_bboxes(inputs.detach().cpu().numpy(), final_outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()[:, 1:], crop_pos_list, self.cam_config,
                    #                                      cam_extr_list, img_list_full)
                    # visualized_img *= 255
                    # for k in range(visualized_img.shape[0]):
                    #     visualized_img[k] = cv2.cvtColor(visualized_img[k], cv2.COLOR_BGR2RGB)
                    visualized_img = None # WIP
                    self.log_writer.add_image("img_train_epoch_" + str(epoch) + "_num_" + str(log_img_counter) + "_input",
                                              visualized_img, epoch, dataformats="NHWC")
                    self.log_writer.flush()
                    log_img_counter += 1

        loss_sum /= len(data["train_loader"])

        # for metric in metrics_sum:
        #     metrics_sum[metric] /= len(data["train_loader"])

        # if self.scheduler.scheduler_type == "reduce_on_plateau":
        #     self.scheduler.scheduler.step(loss_sum)
        # else:
        #     self.scheduler.scheduler.step()



        if self.log_writer:
            self.log_writer.add_scalar("train_loss_" + self.criterions["criterion_train"].loss_type, loss_sum, epoch)
            self.log_writer.add_scalar("learning_rate_" + self.scheduler.scheduler_type, self.optimizer.optimizer.param_groups[0]["lr"], epoch)

            # for metric in self.criterions["criterion_metrics"]:
            #     if metric != "metric_additional":
            #         if self.criterions["criterion_metrics"][metric].metric_type != "accuracy" and self.criterions["criterion_metrics"][metric].metric_type != "precision" and self.criterions["criterion_metrics"][metric].metric_type != "recall" and self.criterions["criterion_metrics"][metric].metric_type != "f1_score" and self.criterions["criterion_metrics"][metric].metric_type != "false_pos_rate" and metric != "metric_additional":
            #             self.log_writer.add_scalar("train_metric_" + self.criterions["criterion_metrics"][metric].metric_type, metrics_sum[metric], epoch)
            #         else:
            #             if epoch % Net_trainer.log_img_save_freq == 0:
            #                 eval_metrics[self.criterions["criterion_metrics"][metric].metric_type] = {self.criterions["criterion_metrics"]["metric_additional"][threshold].metric_config["threshold"]: Metrics_Wrapper({"metric_type": self.criterions["criterion_metrics"][metric].metric_type, "threshold": self.criterions["criterion_metrics"]["metric_additional"][threshold].metric_config["threshold"]}).metric.calc(metrics_sum[threshold]) for threshold in self.criterions["criterion_metrics"]["metric_additional"]}

            # if eval_metrics:
            #     for metric in eval_metrics:
            #         sorted_metric_list = sorted(eval_metrics[metric].items())
            #         plot_metric_x, plot_metric_y = zip(*sorted_metric_list)
            #
            #         fig = plt.figure()
            #         ax = fig.add_subplot()
            #         ax.set_title(metric)
            #         ax.set_xlabel("thresholds")
            #         ax.set_ylabel(metric)
            #         ax.plot(plot_metric_x, plot_metric_y)
            #         self.log_writer.add_figure(metric + "_train_epoch_" + str(epoch), fig, epoch)
            #
            #     if "precision" in eval_metrics.keys() and "recall" in eval_metrics.keys():
            #         fig_pr = plot_pr_curve(eval_metrics["precision"], eval_metrics["recall"])
            #         self.log_writer.add_figure("pr_curve_train_epoch_" + str(epoch), fig_pr, epoch)
            #
            #     if "recall" in eval_metrics.keys() and "false_pos_rate" in eval_metrics.keys():
            #         fig_roc = plot_roc_curve(eval_metrics["recall"], eval_metrics["false_pos_rate"])
            #         self.log_writer.add_figure("roc_curve_train_epoch_" + str(epoch), fig_roc, epoch)


            # self.log_writer.add_pr_curve_raw()
            # self.log_writer.add_pr_curve("pr_curve_train", )
            self.log_writer.flush()


    def val(self, epoch, net, device, data):

        log_img_counter = 0

        loss_sum = 0
        # metrics_sum = {}
        # eval_metrics = {}
        # for metric in self.criterions["criterion_metrics"]:
        #     if metric != "metric_additional":
        #         if self.criterions["criterion_metrics"][metric].metric_type != "accuracy" and \
        #                 self.criterions["criterion_metrics"][metric].metric_type != "precision" and \
        #                 self.criterions["criterion_metrics"][metric].metric_type != "recall" and \
        #                 self.criterions["criterion_metrics"][metric].metric_type != "f1_score" and \
        #                 self.criterions["criterion_metrics"][metric].metric_type != "false_pos_rate":
        #             metrics_sum[metric] = 0
        #     else:
        #         if epoch % Net_trainer.log_img_save_freq == 0:
        #             for threshold in self.criterions["criterion_metrics"][metric]:
        #                 metrics_sum[threshold] = 0
        net.model.eval()
        with torch.no_grad():

            for batch_id, datam in enumerate(tqdm(data["val_loader"], desc="Val_batch", file=sys.stdout)):



                [inputs, labels, input_name, label_name, crop_pos_list, cam_extr_list, img_list_full] = datam

                # inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
                # labels = torch.tensor(labels, device=device, dtype=torch.float32)[:, :, :, 0:1]

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # labels = labels.narrow(3, 0, 1)
                inputs = inputs.permute((0, 3, 2, 1))

                outputs = net.model(inputs)

                final_outputs = net.create_final_outputs(outputs, self.cam_config["fx"], inputs.shape[2])

                # test = outputs.cpu().numpy()

                # plt.imshow(inputs.cpu().numpy()[0, :, :, :])
                # plt.show()
                # plt.imshow(outputs.detach().cpu().numpy()[0, :, :, 0])
                # plt.show()
                # plt.imshow(labels.cpu().numpy()[0, :, :, 0])
                # plt.show()

                loss = self.criterions["criterion_train"].loss(outputs, labels)

                loss_sum += loss.item()

                # for metric in metrics_sum:
                #     # if metric != "accuracy" and metric != "precision" and metric != "recall" and metric != "f1_score" and metric != "false_pos_rate":
                #     if "threshold" not in metric:
                #         metrics_sum[metric] += self.criterions["criterion_metrics"][metric].metric(outputs, labels, final_outputs=final_outputs, crop_pos_list=crop_pos_list, cam_config=self.cam_config,
                #                                          cam_extr_list=cam_extr_list, input_img_size=inputs.shape[2])
                #     else:
                #         if epoch % Net_trainer.log_img_save_freq == 0:
                #             metrics_sum[metric] += self.criterions["criterion_metrics"]["metric_additional"][metric].metric(
                #                 outputs, labels)

                if self.log_writer:
                    if epoch % Net_trainer.log_img_save_freq == 0 and log_img_counter < Net_trainer.log_num_img:
                        # visualized_img = visualize_3d_bboxes(inputs.detach().cpu().numpy(), final_outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()[:, 1:], crop_pos_list, self.cam_config,
                        #                                      cam_extr_list, img_list_full)
                        # visualized_img *= 255

                        visualized_img = None # WIP
                        self.log_writer.add_image("img_val_epoch_" + str(epoch) + "_num_" + str(log_img_counter) + "_input",
                                                  visualized_img, epoch, dataformats="NHWC")
                        self.log_writer.flush()
                        log_img_counter += 1

        loss_sum /= len(data["val_loader"])

        # for metric in metrics_sum:
        #     metrics_sum[metric] /= len(data["train_loader"])

        if self.scheduler.scheduler_type == "reduce_on_plateau":
            self.scheduler.scheduler.step(loss_sum)
        else:
            self.scheduler.scheduler.step()

        if self.log_writer:
            self.log_writer.add_scalar("val_loss_" + self.criterions["criterion_val"].loss_type, loss_sum, epoch)
            # for metric in self.criterions["criterion_metrics"]:
            #     if metric != "metric_additional":
            #         if self.criterions["criterion_metrics"][metric].metric_type != "accuracy" and \
            #                 self.criterions["criterion_metrics"][metric].metric_type != "precision" and \
            #                 self.criterions["criterion_metrics"][metric].metric_type != "recall" and \
            #                 self.criterions["criterion_metrics"][metric].metric_type != "f1_score" and \
            #                 self.criterions["criterion_metrics"][
            #                     metric].metric_type != "false_pos_rate" and metric != "metric_additional":
            #             self.log_writer.add_scalar(
            #                 "val_metric_" + self.criterions["criterion_metrics"][metric].metric_type,
            #                 metrics_sum[metric], epoch)
            #         else:
            #             if epoch % Net_trainer.log_img_save_freq == 0:
            #                 eval_metrics[self.criterions["criterion_metrics"][metric].metric_type] = {
            #                     self.criterions["criterion_metrics"]["metric_additional"][threshold].metric_config[
            #                         "threshold"]: Metrics_Wrapper(
            #                         {"metric_type": self.criterions["criterion_metrics"][metric].metric_type, "threshold":
            #                             self.criterions["criterion_metrics"]["metric_additional"][threshold].metric_config[
            #                                 "threshold"]}).metric.calc(metrics_sum[threshold]) for threshold in
            #                     self.criterions["criterion_metrics"]["metric_additional"]}

            # if eval_metrics:
            #     for metric in eval_metrics:
            #         sorted_metric_list = sorted(eval_metrics[metric].items())
            #         plot_metric_x, plot_metric_y = zip(*sorted_metric_list)
            #
            #         fig = plt.figure()
            #         ax = fig.add_subplot()
            #         ax.set_title(metric)
            #         ax.set_xlabel("thresholds")
            #         ax.set_ylabel(metric)
            #         ax.plot(plot_metric_x, plot_metric_y)
            #         self.log_writer.add_figure(metric + "_val_epoch_" + str(epoch), fig, epoch)
            #
            #     if "precision" in eval_metrics.keys() and "recall" in eval_metrics.keys():
            #         fig_pr = plot_pr_curve(eval_metrics["precision"], eval_metrics["recall"])
            #         self.log_writer.add_figure("pr_curve_val_epoch" + str(epoch), fig_pr, epoch)
            #
            #     if "recall" in eval_metrics.keys() and "false_pos_rate" in eval_metrics.keys():
            #         fig_roc = plot_roc_curve(eval_metrics["recall"], eval_metrics["false_pos_rate"])
            #         self.log_writer.add_figure("roc_curve_val_epoch" + str(epoch), fig_roc, epoch)

            self.log_writer.flush()

        if "min" in self.best_eval_mode:
            if loss_sum < self.best_loss_score:
                self.best_loss_score = loss_sum
                self.save_model(net, best=True)
                # print("best_loss_score: " + str(loss_sum))
                tqdm.write("best_loss_score: " + str(loss_sum))
        if "max" in self.best_eval_mode:
            if loss_sum > self.best_loss_score:
                self.best_loss_score = loss_sum
                self.save_model(net, best=True)
                # print("best_loss_score: " + str(loss_sum))
                tqdm.write("best_loss_score: " + str(loss_sum))

    # def epoch_finish(self):
    #
    #     self.scheduler.scheduler.step()
    #


    def train(self, net, device, data):


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



        self.log_writer.close()