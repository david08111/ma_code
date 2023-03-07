import argparse
import os
import json
from tqdm import tqdm
import seaborn
from matplotlib import pyplot as plt

def class_weights_inverse(rel_distr_list):
    final_cls_weight_list = []

    for elem in rel_distr_list:
        final_cls_weight_list.append(1/elem)

    return final_cls_weight_list

def class_weights_effective_number_of_samples(rel_distr_list, beta=0.99):
    final_cls_weight_list = []

    dividend = 1 - beta
    for elem in rel_distr_list:
        final_cls_weight_list.append(dividend/(1-beta**elem))

    return final_cls_weight_list


def class_weights_inverse_percentage(rel_distr_list):
    final_cls_weight_list = []
    # total_sum = sum(rel_distr_list)
    for elem in rel_distr_list:
        final_cls_weight_list.append(1 - elem)

    return final_cls_weight_list


def calc_class_weights_from_rel_distr(rel_distr_list, method):
    if method == "inverse":
        return class_weights_inverse(rel_distr_list)
    elif method == "eff_num_sampl":
        return class_weights_effective_number_of_samples(rel_distr_list)
    elif method == "inv_percentage":
        return class_weights_inverse_percentage(rel_distr_list)
    else:
        raise NameError(f"Method {method} not implemented!")

# def calc_cls_pixel_distribution(dataset_config_path, masks_source_path, save_path, calc_weights=True):
def calc_cls_pixel_distribution(dataset_config_path, save_path, calc_weights_method="inv"):
    dataset_config = None
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)

    categories = dataset_config["categories"]
    categories_id = {el['id']: el for el in dataset_config['categories']}
    categories_name = {el['name']: el for el in dataset_config['categories']}

    # categories_stat_data_tmp = {el['id']: el for el in dataset_config['categories']}
    categories_stat_data = {el['id']: {"area_list": [],
                                       "total_area": 0,
                                       "avg_area": 0,
                                       "min_area": float("inf"),
                                       "max_area": float("-inf"),
                                       "median_area": -1,
                                       "num_appearances": 0} for el in dataset_config['categories']}

    for annotation_instance in tqdm(dataset_config["annotations"]):
        for segment in annotation_instance["segments_info"]:
            categories_stat_data[segment["category_id"]]["area_list"].append(segment["area"])
            # categories_stat_data[segment["category_id"]]["total_area"] += segment["area"]
            #
            # if segment["area"] < categories_stat_data[segment["category_id"]]["min_area"]:
            #     categories_stat_data[segment["category_id"]]["min_area"] = segment["area"]
            #
            # if segment["area"] > categories_stat_data[segment["category_id"]]["max_area"]:
            #     categories_stat_data[segment["category_id"]]["max_area"] = segment["area"]
            #
            # categories_stat_data[segment["category_id"]]["num_appearances"] += 1


    total_cls_area_sum = 0

    for id in categories_stat_data:
        categories_stat_data[id]["num_appearances"] = len(categories_stat_data[id]["area_list"])
        categories_stat_data[id]["total_area"] = sum(categories_stat_data[id]["area_list"])
        categories_stat_data[id]["avg_area"] = categories_stat_data[id]["total_area"] / categories_stat_data[id]["num_appearances"]
        categories_stat_data[id]["min_area"] = min(categories_stat_data[id]["area_list"])
        categories_stat_data[id]["max_area"] = max(categories_stat_data[id]["area_list"])
        categories_stat_data[id]["median_area"] = sorted(categories_stat_data[id]["area_list"])[categories_stat_data[id]["num_appearances"]//2-1]

        total_cls_area_sum += categories_stat_data[id]["total_area"]

    id_list = []
    class_name_list = []
    weight_list = []


    num_appearances_list = []
    total_area_list = []
    avg_area_list = []
    min_area_list = []
    max_area_list = []
    median_area_list = []
    rel_dataset_area_list = []


    for id in categories_stat_data:
        categories_stat_data[id]["dataset_rel_area"] = categories_stat_data[id]["total_area"] / total_cls_area_sum

        num_appearances_list.append(categories_stat_data[id]['num_appearances'])
        total_area_list.append(categories_stat_data[id]['total_area'])
        avg_area_list.append(categories_stat_data[id]['avg_area'])
        min_area_list.append(categories_stat_data[id]['min_area'])
        max_area_list.append(categories_stat_data[id]['max_area'])
        median_area_list.append(categories_stat_data[id]['median_area'])
        rel_dataset_area_list.append(categories_stat_data[id]['dataset_rel_area'])

        print("-" * 20)
        print(f"Class {categories_id[id]['name']} - ID {id}")
        print(f"Num appearances: {categories_stat_data[id]['num_appearances']}")
        print(f"Total area: {categories_stat_data[id]['total_area']}")
        print(f"Avg area: {categories_stat_data[id]['avg_area']}")
        print(f"Min area: {categories_stat_data[id]['min_area']}")
        print(f"Max area: {categories_stat_data[id]['max_area']}")
        print(f"Median area: {categories_stat_data[id]['median_area']}")
        print(f"Rel dataset area: {categories_stat_data[id]['dataset_rel_area']}")

        id_list.append(id)
        class_name_list.append(categories_id[id]['name'])
        # weight_list.append(categories_stat_data[id]["dataset_rel_area"])


    seaborn.barplot(x=num_appearances_list, y=class_name_list)
    plt.xlabel("Num Appearances")
    plt.ylabel("Class")
    plt.savefig(os.path.join(save_path, "num_appearances.png"))
    plt.show()
    seaborn.barplot(x=total_area_list, y=class_name_list)
    plt.xlabel("Total Area")
    plt.ylabel("Class")
    plt.savefig(os.path.join(save_path, "total_area.png"))
    plt.show()
    seaborn.barplot(x=avg_area_list, y=class_name_list)
    plt.xlabel("Avg Area")
    plt.ylabel("Class")
    plt.savefig(os.path.join(save_path, "avg_area.png"))
    plt.show()
    seaborn.barplot(x=min_area_list, y=class_name_list)
    plt.xlabel("Min Area")
    plt.ylabel("Class")
    plt.savefig(os.path.join(save_path, "min_area.png"))
    plt.show()
    seaborn.barplot(x=max_area_list, y=class_name_list)
    plt.xlabel("Max Area")
    plt.ylabel("Class")
    plt.savefig(os.path.join(save_path, "max_area.png"))
    plt.show()
    seaborn.barplot(x=median_area_list, y=class_name_list)
    plt.xlabel("Median Area")
    plt.ylabel("Class")
    plt.savefig(os.path.join(save_path, "median_area.png"))
    plt.show()
    seaborn.barplot(x=rel_dataset_area_list, y=class_name_list)
    plt.xlabel("Rel Dataset Area")
    plt.ylabel("Class")
    plt.savefig(os.path.join(save_path, "rel_dataset_area.png"))
    plt.show()

    weight_list = calc_class_weights_from_rel_distr(rel_dataset_area_list, calc_weights_method)

    print("Calc class weight list:")
    print(weight_list)
    print("Related class ID order list:")
    print(id_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--dataset_config_path", type=str,
                        help="Filepath to base data")
    # parser.add_argument("-s", "--masks_source_path", type=str,
    #                     help="Filepath to mask data")
    parser.add_argument("-d", "--out_file_path", type=str,
                        help="File path for output")
    parser.add_argument("-w", "--calc_weights", type=str,
                        help="Method to calc related class weights")
    args = parser.parse_args()

    # calc_cls_pixel_distribution(args.dataset_config_path, args.masks_source_path, args.out_file_path, args.calc_weights)
    calc_cls_pixel_distribution(args.dataset_config_path, args.out_file_path, args.calc_weights)