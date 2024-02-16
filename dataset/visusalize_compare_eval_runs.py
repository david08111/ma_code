import argparse
import json
import seaborn
import pandas as pd
import numpy as np
import copy
import os
from matplotlib import pyplot as plt
from utils.logger import flatten_dict

cityscapes_categories_dict = {7: {'id': 7, 'name': 'road', 'color': [128, 64, 128], 'supercategory': 'flat', 'isthing': 0},
 8: {'id': 8, 'name': 'sidewalk', 'color': [244, 35, 232], 'supercategory': 'flat', 'isthing': 0},
 11: {'id': 11, 'name': 'building', 'color': [70, 70, 70], 'supercategory': 'construction', 'isthing': 0},
 12: {'id': 12, 'name': 'wall', 'color': [102, 102, 156], 'supercategory': 'construction', 'isthing': 0},
 13: {'id': 13, 'name': 'fence', 'color': [190, 153, 153], 'supercategory': 'construction', 'isthing': 0},
 17: {'id': 17, 'name': 'pole', 'color': [153, 153, 153], 'supercategory': 'object', 'isthing': 0},
 19: {'id': 19, 'name': 'traffic light', 'color': [250, 170, 30], 'supercategory': 'object', 'isthing': 0},
 20: {'id': 20, 'name': 'traffic sign', 'color': [220, 220, 0], 'supercategory': 'object', 'isthing': 0},
 21: {'id': 21, 'name': 'vegetation', 'color': [107, 142, 35], 'supercategory': 'nature', 'isthing': 0},
 22: {'id': 22, 'name': 'terrain', 'color': [152, 251, 152], 'supercategory': 'nature', 'isthing': 0},
 23: {'id': 23, 'name': 'sky', 'color': [70, 130, 180], 'supercategory': 'sky', 'isthing': 0},
 24: {'id': 24, 'name': 'person', 'color': [220, 20, 60], 'supercategory': 'human', 'isthing': 1},
 25: {'id': 25, 'name': 'rider', 'color': [255, 0, 0], 'supercategory': 'human', 'isthing': 1},
 26: {'id': 26, 'name': 'car', 'color': [0, 0, 142], 'supercategory': 'vehicle', 'isthing': 1},
 27: {'id': 27, 'name': 'truck', 'color': [0, 0, 70], 'supercategory': 'vehicle', 'isthing': 1},
 28: {'id': 28, 'name': 'bus', 'color': [0, 60, 100], 'supercategory': 'vehicle', 'isthing': 1},
 31: {'id': 31, 'name': 'train', 'color': [0, 80, 100], 'supercategory': 'vehicle', 'isthing': 1},
 32: {'id': 32, 'name': 'motorcycle', 'color': [0, 0, 230], 'supercategory': 'vehicle', 'isthing': 1},
 33: {'id': 33, 'name': 'bicycle', 'color': [119, 11, 32], 'supercategory': 'vehicle', 'isthing': 1}}

def convert_metrics_dict(metrics_dict):
    test_categories_dict = cityscapes_categories_dict
    metrics_dict_tmp = copy.deepcopy(metrics_dict)
    for metric in metrics_dict_tmp:
        if metric == "panoptic_quality":
            metrics_dict_pq_tmp = metrics_dict.pop(metric)

            metrics_dict["panoptic_quality"] = {}
            metrics_dict["recognition_quality"] = {}
            metrics_dict["segmentation_quality"] = {}

            for key in metrics_dict_pq_tmp:
                if not key == "Class":
                    for metric_name in metrics_dict_pq_tmp[key]:
                        if metric_name == "pq":
                            metrics_dict["panoptic_quality"][key] = metrics_dict_pq_tmp[key][metric_name]
                        if metric_name == "sq":
                            metrics_dict["segmentation_quality"][key] = metrics_dict_pq_tmp[key][metric_name]
                        if metric_name == "rq":
                            metrics_dict["recognition_quality"][key] = metrics_dict_pq_tmp[key][metric_name]

            for cat_id in metrics_dict_pq_tmp["Class"]:
                for metric_name in metrics_dict_pq_tmp["Class"][cat_id]:
                    if metric_name == "pq":
                        # test = test_categories_dict[cat_id]
                        # cat_id = str(cat_id)
                        # test2 = metrics_dict_pq_tmp[key][cat_id]
                        metrics_dict["panoptic_quality"][cityscapes_categories_dict[int(cat_id)]["name"].capitalize()] = metrics_dict_pq_tmp["Class"][cat_id][metric_name]
                    if metric_name == "sq":
                        metrics_dict["segmentation_quality"][cityscapes_categories_dict[int(cat_id)]["name"].capitalize()] = metrics_dict_pq_tmp["Class"][cat_id][metric_name]
                    if metric_name == "rq":
                        metrics_dict["recognition_quality"][cityscapes_categories_dict[int(cat_id)]["name"].capitalize()] = metrics_dict_pq_tmp["Class"][cat_id][metric_name]

        if metric == "silhouette_score":

            for key in metrics_dict_tmp[metric]:
                if key == "Class":
                    metrics_class_dict_tmp = metrics_dict[metric].pop(key)
                    for cat_id in metrics_class_dict_tmp:
                        metrics_dict[metric][cityscapes_categories_dict[int(cat_id)]["name"].capitalize()] = metrics_class_dict_tmp[cat_id]

                if key == "Things_Semantic":
                    metrics_class_dict_tmp = metrics_dict[metric].pop(key)
                    for cat_id in metrics_class_dict_tmp:
                        metrics_dict[metric]["Things - Semantic - " + cityscapes_categories_dict[int(cat_id)]["name"].capitalize()] = \
                        metrics_class_dict_tmp[cat_id]

                if key == "Things_Instance":
                    metrics_class_dict_tmp = metrics_dict[metric].pop(key)
                    for cat_id in metrics_class_dict_tmp:
                        metrics_dict[metric]["Things - Instance - " + cityscapes_categories_dict[int(cat_id)]["name"].capitalize()] = \
                        metrics_class_dict_tmp[cat_id]


# def visusalize_compare_eval_runs(run_comp_name, label_list, eval_file_path_list, save_path, categories_dict, xlabel, ylabel):
def visusalize_compare_eval_runs(run_comp_name, ylabel_set, label_list, eval_file_path_list, save_path, categories_dict, sort_numeric_labels):
    metrics_dict = {}

    # seaborn.set_theme()
    # seaborn.set_style("whitegrid")
    # # seaborn.set_context("talk")
    # seaborn.set_context("notebook", font_scale=1.75, rc={"lines.linewidth": 2.5, 'font.family':'Helvetica'})

    # df[run_comp_name] = label_list
    if not ylabel_set:
        ylabel = run_comp_name.split(" ")[0]
    else:
        ylabel = ylabel_set

    if sort_numeric_labels:
        label_list_tmp = [float(elem) for elem in label_list]
        label_list_np = np.array(label_list_tmp)
        label_list_sort_indices = np.argsort(label_list_np)

        eval_file_path_list_tmp = list(eval_file_path_list)
        eval_file_path_list = []
        for i in range(label_list_sort_indices.shape[0]):
            eval_file_path_list.append(eval_file_path_list_tmp[label_list_sort_indices[i]])

        labels_list_tmp = list(label_list)
        label_list = []
        for i in range(label_list_sort_indices.shape[0]):
            label_list.append(labels_list_tmp[label_list_sort_indices[i]])



    for i in range(len(eval_file_path_list)):
        eval_file_path = eval_file_path_list[i]
        if not os.path.isfile(eval_file_path):
            # test = os.listdir(eval_file_path)
            if "metrics.json" in os.listdir(eval_file_path):
                eval_file_path = os.path.join(eval_file_path, "metrics.json")
        with open(eval_file_path, 'r') as f:
            data_metrics_dict = json.load(f)

            convert_metrics_dict(data_metrics_dict)

            for metric in data_metrics_dict:
                flattened_dict = flatten_dict(data_metrics_dict[metric], separator=" - ")



                if metric not in metrics_dict:
                    metrics_dict[metric] = {}

                for key in flattened_dict:
                    if key not in metrics_dict[metric]:
                        metrics_dict[metric][key] = {}

                    metrics_dict[metric][key][label_list[i]] = flattened_dict[key]

            # for metric in data_metrics_dict:
            #     if metric not in metrics_dict:
            #         metrics_dict[metric] = {}
            #
            #     for key in data_metrics_dict[metric]:
            #         metrics_dict_tmp = data_metrics_dict[metric][key]
            #         if key == "Class":
            #             metrics_dict_tmp = data_metrics_dict[metric]["Class"]
            #         else:
            #             if key not in metrics_dict[metric]:
            #                 metrics_dict[metric][key] = {}
            #         for final_key_value in metrics_dict_tmp:
            #             if final_key_value not in metrics_dict[metric][key]:
            #                 metrics_dict[metric][key][final_key_value] = []
            #             metrics_dict[metric][key][final_key_value].append(metrics_dict_tmp[final_key_value])
            #     for key in data_metrics_dict[metric]:
            #         metrics_dict_tmp = data_metrics_dict[metric][key]
            #         if key == "Class":
            #             metrics_dict_tmp = data_metrics_dict[metric]["Class"]
            #         else:
            #             if key not in metrics_dict[metric]:
            #                 metrics_dict[metric][key] = {}
            #         for final_key_value in metrics_dict_tmp:
            #             if final_key_value not in metrics_dict[metric][key]:
            #                 metrics_dict[metric][key][final_key_value] = []
            #             metrics_dict[metric][key][final_key_value].append(metrics_dict_tmp[final_key_value])

    df_dict = {}

    for metric in metrics_dict:
        df = pd.DataFrame(dtype=float)
        for key in metrics_dict[metric]:
            if " - n" in key:
                continue
            df[key] = metrics_dict[metric][key]
            # df.loc[key] = metrics_dict[key]

        df_dict[metric] = df

    # test_frame = df.append(metrics_dict, ignore_index=True)
    #
    # print(test_frame)

    # df = df.T
    # for metric in df_dict:
    #     print(df_dict[metric])
    #     f, ax = plt.subplots(figsize=(30, 8), dpi=150)
    #     seaborn.heatmap(df_dict[metric], annot=True, linewidths=.5, ax=ax, square=True, cbar_kws={"shrink": 0.25}, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    #     # seaborn.set(font_scale=2)
    #
    #     ax.set_title(run_comp_name, size=35)
    #     ax.set_xlabel(xlabel, size=25)
    #     ax.set_ylabel(ylabel, size=25)
    #     ax.set_aspect("equal")
    #
    #     # ax.xaxis.tick_top()
    #     # plt.xlabel(xlabel)
    #     # plt.ylabel(ylabel)
    #
    #
    #     # plt.xticks(size=14)
    #     # plt.yticks(size=14)
    #
    #     # plt.rcParams["figure.dpi"] = 1600
    #
    #     plt.tight_layout()
    #
    #     plt.show()
    save_path_name = run_comp_name.replace(" ", "_")

    for metric in df_dict:
        print(df_dict[metric])

    ##################################
    ## Panoptic Quality
    ##################################
    f, ax = plt.subplots(figsize=(15, 10), dpi=150)
    # f, ax = plt.subplots(dpi=350)
    # seaborn.heatmap(df_dict["panoptic_quality"], annot=True, linewidths=.5, ax=ax, square=True, cbar_kws={"use_gridspec": False, "shrink": 0.5}, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    seaborn.heatmap(df_dict["panoptic_quality"], vmin=0, vmax=1, annot=True, linewidths=.5, ax=ax, square=True, cmap="viridis",
                    cbar_kws={"location": "top", "shrink": 0.5},
                    annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    # seaborn.set(font_scale=2)

    # ax.set_title(run_comp_name, size=35)
    ax.set_xlabel("Panoptic Quality", size=25)
    ax.set_ylabel(ylabel, size=25)
    ax.set_aspect("equal")

    plt.tight_layout()

    # plt.show()
    plt.savefig(os.path.join(save_path, save_path_name + "_panoptic_quality.png"))

    ##################################
    ## Segmentation Quality
    ##################################
    f, ax = plt.subplots(figsize=(15, 10), dpi=150)
    # f, ax = plt.subplots(dpi=350)
    # seaborn.heatmap(df_dict["panoptic_quality"], annot=True, linewidths=.5, ax=ax, square=True, cbar_kws={"use_gridspec": False, "shrink": 0.5}, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    seaborn.heatmap(df_dict["segmentation_quality"], vmin=0, vmax=1, annot=True, linewidths=.5, ax=ax, square=True, cmap="viridis",
                    cbar_kws={"location": "top", "shrink": 0.5},
                    annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    # seaborn.set(font_scale=2)

    # ax.set_title(run_comp_name, size=35)
    ax.set_xlabel("Segmentation Quality", size=25)
    ax.set_ylabel(ylabel, size=25)
    ax.set_aspect("equal")

    plt.tight_layout()

    # plt.show()
    plt.savefig(os.path.join(save_path, save_path_name + "_segmentation_quality.png"))

    ##################################
    ## Panoptic Quality
    ##################################
    f, ax = plt.subplots(figsize=(15, 10), dpi=150)
    # f, ax = plt.subplots(dpi=350)
    # seaborn.heatmap(df_dict["panoptic_quality"], annot=True, linewidths=.5, ax=ax, square=True, cbar_kws={"use_gridspec": False, "shrink": 0.5}, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    seaborn.heatmap(df_dict["recognition_quality"], vmin=0, vmax=1, annot=True, linewidths=.5, ax=ax, square=True, cmap="viridis",
                    cbar_kws={"location": "top", "shrink": 0.5},
                    annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    # seaborn.set(font_scale=2)

    # ax.set_title(run_comp_name, size=35)
    ax.set_xlabel("Recognition Quality", size=25)
    ax.set_ylabel(ylabel, size=25)
    ax.set_aspect("equal")

    plt.tight_layout()

    # plt.show()
    plt.savefig(os.path.join(save_path, save_path_name + "_recognition_quality.png"))

    #####################################
    ## Silhouette Score
    ######################################
    f, ax = plt.subplots(figsize=(18, 10), dpi=150)
    # f, ax = plt.subplots(dpi=350)
    # seaborn.heatmap(df_dict["silhouette_score"], annot=True, linewidths=.5, ax=ax, square=True,
    #                 cbar_kws={"use_gridspec": False, "shrink": 0.5}, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    seaborn.heatmap(df_dict["silhouette_score"], vmin=-1, vmax=1, annot=True, linewidths=.5, ax=ax, square=True, cmap="viridis",      # fmt=".2f",
                    cbar_kws={"location": "top", "shrink": 0.5},
                    annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    # seaborn.heatmap(df_dict["silhouette_score"], annot=True, linewidths=.5, ax=ax, square=True, annot_kws={'fontsize': 8, 'fontstyle': 'italic', 'alpha': 0.9})
    # seaborn.set(font_scale=2)

    # ax.set_title(run_comp_name, size=35)
    ax.set_xlabel("Silhouette Score", size=25)
    ax.set_ylabel(ylabel, size=25)
    ax.set_aspect("equal")

    plt.tight_layout()

    # plt.show()
    plt.savefig(os.path.join(save_path, save_path_name + "_silhouette_score.png"))

    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Name of the plot!")
    # parser.add_argument("-x", "--xlabel", type=str, help="Name of the x axis!")
    parser.add_argument("-y", "--ylabel", type=str, default=None, help="Name of the y axis!")
    parser.add_argument("-l", "--label", type=str, action="append", help="Name of the plot!")
    parser.add_argument("-d", "--eval_file_path", type=str, action="append", help="File path to eval_metrics json")
    # parser.add_argument("-s", "--in_path", type=str, help="Input Data Dir")
    parser.add_argument("-o", "--sort_numeric_labels", action="store_true",
                        help="Wether to sort the numeric label list!")
    parser.add_argument("-s", "--save_path", type=str,
                        help="Output Save Path!")
    parser.add_argument("-t", "--dataset_type", type=str, default="cityscapes",
                        help="Dataset type!")

    args = parser.parse_args()

    if args.dataset_type == "cityscapes":
        categories_dict = cityscapes_categories_dict
    else:
        raise NotImplementedError(f"Dataset {args.dataset_type} not implemented!")

    # visusalize_compare_eval_runs(args.name, args.label, args.eval_file_path, args.save_path, categories_dict, args.xlabel, args.ylabel)
    visusalize_compare_eval_runs(args.name, args.ylabel, args.label, args.eval_file_path, args.save_path, categories_dict, args.sort_numeric_labels)