import argparse
import torch
from torch.utils.data import DataLoader
from utils import Config
import os
import time
from tqdm import tqdm
import cv2
import numpy as np
from matplotlib import pyplot as plt
from data_handling import DataHandler
from utils import Evaluation_Logger
from training import Metrics_Wrapper, Loss_Wrapper


def evaluate(in_path, out_path, weight_file, config_file):
    config = Config()
    config_dict = config(os.path.abspath(config_file))

    eval_logger = Evaluation_Logger(out_path)

    net = Model(config_dict["network"]["architecture"], config_dict["network"]["in_channels"],
                config_dict["network"]["classes"], config_dict["data"]["img_size"], config_dict["network"]["architecture_config"])

    net.eval()

    total_net_params = sum(p.numel() for p in net.model.parameters())

    state = torch.load(os.path.abspath(weight_file))

    try:
        net.model.load_state_dict(state["state_dict"]["model"])
    except KeyError:
        try:
            net.model.load_state_dict(state["model"])
        except KeyError:
            net.model.load_state_dict(state)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net.to(device)

    # data = {
    #     "pred_loader": DataLoader(
    #         dataset=DataHandler(in_path, config_dict["data"]["img_size"],
    #                             config_dict["data"]["img_size"], config_dict["network"]["in_channels"], device,
    #                             rotation=0, translation=0, scaling=0, hor_flip=False, ver_flip=False,
    #                             config_data=config_dict["data"]),
    #         batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
    #         pin_memory=True)
    # }
    data = {
        "pred_loader": DataLoader(
            dataset=DataHandler(in_path, config_dict["data"]["img_size"],
                                config_dict["data"]["img_size"], config_dict["network"]["in_channels"], device,
                                rotation=0, translation=0, scaling=0, hor_flip=False, ver_flip=False,
                                config_data=config_dict["data"]),
            batch_size=1, shuffle=False, num_workers=config_dict["data"]["num_workers"], drop_last=False,
            pin_memory=True)
    }

    criterions = {
        "criterion_metrics": {key: Metrics_Wrapper(config_dict["data"]["metrics"][key]) for key in
                              config_dict["data"]["metrics"]}
    }

    metric_threshold_step_sizes = []
    metric_num_thresholds = []
    for metric in criterions["criterion_metrics"]:
        if "num_thresholds" in criterions["criterion_metrics"][metric].metric_config.keys():
            metric_num_thresholds.append(criterions["criterion_metrics"][metric].metric_config["num_thresholds"])

    if all(num_threshold == metric_num_thresholds[0] for num_threshold in metric_num_thresholds):
        common_num_threshold = metric_num_thresholds[0]
    else:
        common_num_threshold = metric_num_thresholds[0]
        print(
            "Different num thresholds for classification detected. Num threshold: " + common_num_threshold + " is used")

    for metric in criterions["criterion_metrics"]:
        if "accuracy" in criterions["criterion_metrics"][metric].metric_type or "precision" in \
                criterions["criterion_metrics"][metric].metric_type or "recall" in \
                criterions["criterion_metrics"][metric].metric_type or "f1_score" in \
                criterions["criterion_metrics"][metric].metric_type or "false_pos_rate" in \
                criterions["criterion_metrics"][metric].metric_type:
            criterions["criterion_metrics"]["metric_additional"] = {
                "threshold_" + str(threshold): Metrics_Wrapper({"metric_type": "class_cases",
                                                                "threshold": threshold}) for threshold in
                np.arange(0, 1, 1 / common_num_threshold)}
            break

    metrics_sum = {}
    eval_metrics = {}
    for metric in criterions["criterion_metrics"]:
        if metric != "metric_additional":
            if criterions["criterion_metrics"][metric].metric_type != "accuracy" and \
                    criterions["criterion_metrics"][metric].metric_type != "precision" and \
                    criterions["criterion_metrics"][metric].metric_type != "recall" and \
                    criterions["criterion_metrics"][metric].metric_type != "f1_score" and \
                    criterions["criterion_metrics"][metric].metric_type != "false_pos_rate":
                metrics_sum[metric] = 0
        else:
            for threshold in criterions["criterion_metrics"][metric]:
                metrics_sum[threshold] = 0

    predict_time_net_sum = 0

    with torch.no_grad():

        for batch_id, datam in enumerate(tqdm(data["pred_loader"], desc="Predict")):

            [inputs, labels, input_file_name, label_file_name] = datam

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = labels.narrow(3, 0, 1).contiguous()

            start_pred_time_net = time.time()
            outputs = net.model(inputs)
            end_pred_time_net = time.time()

            predict_time_net_sum += (end_pred_time_net - start_pred_time_net)


            for metric in metrics_sum:
                # if metric != "accuracy" and metric != "precision" and metric != "recall" and metric != "f1_score" and metric != "false_pos_rate":
                if "threshold" not in metric:
                    metrics_sum[metric] += criterions["criterion_metrics"][metric].metric(outputs, labels)
                else:
                    metrics_sum[metric] += criterions["criterion_metrics"]["metric_additional"][metric].metric(outputs, labels)


    predict_time_net_sum /= batch_id

    for metric in metrics_sum:
        metrics_sum[metric] /= len(data["train_loader"])

    for metric in criterions["criterion_metrics"]:
        if metric != "metric_additional":
            if criterions["criterion_metrics"][metric].metric_type != "accuracy" and \
                    criterions["criterion_metrics"][metric].metric_type != "precision" and \
                    criterions["criterion_metrics"][metric].metric_type != "recall" and \
                    criterions["criterion_metrics"][metric].metric_type != "f1_score" and \
                    criterions["criterion_metrics"][
                        metric].metric_type != "false_pos_rate" and metric != "metric_additional":
                eval_logger.add_item(criterions["criterion_metrics"][metric].metric_type,
                                           metrics_sum[metric])
            else:
                eval_metrics[criterions["criterion_metrics"][metric].metric_type] = {
                    criterions["criterion_metrics"]["metric_additional"][threshold].metric_config[
                        "threshold"]: Metrics_Wrapper(
                        {"metric_type": criterions["criterion_metrics"][metric].metric_type, "threshold":
                            criterions["criterion_metrics"]["metric_additional"][threshold].metric_config[
                                "threshold"]}).metric.calc(metrics_sum[threshold]) for threshold in
                    criterions["criterion_metrics"]["metric_additional"]}

    if eval_metrics:
        for metric in eval_metrics:
            sorted_metric_list = sorted(eval_metrics[metric].items())
            plot_metric_x, plot_metric_y = zip(*sorted_metric_list)
            eval_logger.add_item(metric + "_plot", {"threshold": plot_metric_x, metric: plot_metric_y})

        if "precision" in eval_metrics.keys() and "recall" in eval_metrics.keys():
            sorted_precision_list = sorted(eval_metrics["precision"].items())
            thresholds, precision = zip(*sorted_precision_list)

            sorted_recall_list = sorted(eval_metrics["recall"].items())
            thresholds, recall = zip(*sorted_recall_list)

            eval_logger.add_item("pr_curve", {"recall": recall, "precision": precision})

        if "recall" in eval_metrics.keys() and "false_pos_rate" in eval_metrics.keys():
            sorted_tpr_list = sorted(eval_metrics["recall"].items())
            thresholds, true_pos_rate = zip(*sorted_tpr_list)

            sorted_fpr_list = sorted(eval_metrics["false_pos_rate"].items())
            thresholds, false_pos_rate = zip(*sorted_fpr_list)

            eval_logger.add_item("roc_curve", {"false_pos_rate": false_pos_rate, "true_pos_rate": true_pos_rate})

    eval_logger.add_item("model_param_size", total_net_params)
    eval_logger.add_item("avg_pred_time", predict_time_net_sum)

    eval_logger.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--in_path_pred", type=str, help="Input Prediction Data Dir")
    parser.add_argument("-ǵ", "--in_path_gt", type=str, help="Input Groundtruth Data Dir")
    parser.add_argument("-s", "--out_path", default="./outputs", type=str,
                        help="Output Data Dir")


    args = parser.parse_args()

    evaluate(args.in_path, args.out_path, args.weight_file, args.config_file)