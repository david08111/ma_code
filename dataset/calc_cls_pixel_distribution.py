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

    ################
    class_names_string = ""
    class_id_default_ordering = ""
    for elem in categories:
        # print(elem["name"] + " " + str(elem["id"]))
        class_names_string = class_names_string + ", " + elem["name"] + "(" + str(elem["id"]) + ")"
        class_id_default_ordering = class_id_default_ordering + ", " + str(elem["id"])
    class_id_reversed = ""
    for elem in reversed(categories):
        # print(elem["name"] + " " + str(elem["id"]))
        class_id_reversed = class_id_reversed + ", " + str(elem["id"])

    print(class_names_string)
    print(class_id_reversed)
    print(class_id_default_ordering)

    table_array = [["road(7)", "sidewalk(8)", "building(11)", "wall(12)", "fence(13)", "pole(17)", "traffic light(19)", "traffic sign(20)", "vegetation(21)", "terrain(22)", "sky(23)", "person(24)", "rider(25)",  "car(26)", "truck(27)", "bus(28)", "train(31)", "motorcycle(32)", "bicycle(33)"],
                    [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
                   [33, 32, 31, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 17, 13, 12, 11, 8, 7],
                   [23, 24, 25, 26, 27, 28, 31, 32, 33, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22],
                   [25, 21, 23, 8, 22, 28, 7, 20, 13, 33, 17, 24, 12, 31, 11, 32, 19, 27, 26],
                   [21, 7, 28, 17, 12, 13, 8, 25, 23, 24, 19, 26, 33, 27, 11, 22, 20, 32, 31],
                   [8, 31, 19, 13, 22, 11, 12, 28, 25, 20, 23, 33, 24, 27, 7, 32, 17, 21, 26],
                   [12, 8, 23, 24, 7, 27, 28, 11, 20, 32, 13, 22, 26, 17, 21, 25, 19, 31, 33],
                   [26, 11, 25, 8, 20, 23, 28, 13, 22, 21, 32, 27, 31, 33, 12, 17, 7, 19, 24],
                   [17, 21, 20, 32, 11, 24, 25, 26, 23, 28, 27, 31, 19, 13, 12, 7, 33, 22, 8 ],
                   [24, 21, 33, 27, 20, 25, 26, 19, 23, 28, 12, 22, 8, 31, 13, 11, 7, 17, 32],
                   
                   ]
    ############################

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
            # if segment["area"] > 7:
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

    ## seaborn.color_palette("muted")
    # seaborn.barplot(x=num_appearances_list, y=class_name_list)
    # plt.xlabel("Num Appearances")
    # plt.ylabel("Class")
    # plt.savefig(os.path.join(save_path, "num_appearances.png"))
    # plt.show()
    # seaborn.barplot(x=total_area_list, y=class_name_list)
    # plt.xlabel("Total Area")
    # plt.ylabel("Class")
    # plt.savefig(os.path.join(save_path, "total_area.png"))
    # plt.show()
    # seaborn.barplot(x=avg_area_list, y=class_name_list)
    # plt.xlabel("Avg Area")
    # plt.ylabel("Class")
    # plt.savefig(os.path.join(save_path, "avg_area.png"))
    # plt.show()
    # seaborn.barplot(x=min_area_list, y=class_name_list)
    # plt.xlabel("Min Area")
    # plt.ylabel("Class")
    # plt.savefig(os.path.join(save_path, "min_area.png"))
    # plt.show()
    # seaborn.barplot(x=max_area_list, y=class_name_list)
    # plt.xlabel("Max Area")
    # plt.ylabel("Class")
    # plt.savefig(os.path.join(save_path, "max_area.png"))
    # plt.show()
    # seaborn.barplot(x=median_area_list, y=class_name_list)
    # plt.xlabel("Median Area")
    # plt.ylabel("Class")
    # plt.savefig(os.path.join(save_path, "median_area.png"))
    # plt.show()
    # seaborn.barplot(x=rel_dataset_area_list, y=class_name_list)
    # plt.xlabel("Rel Dataset Area")
    # plt.ylabel("Class")
    # plt.savefig(os.path.join(save_path, "rel_dataset_area.png"))
    # plt.show()


    seaborn.barplot(y=num_appearances_list, x=class_name_list, palette="viridis")
    plt.gca().invert_yaxis()
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    seaborn.despine(top=False, bottom=True, right=True)
    plt.xticks(
        rotation=45,
        horizontalalignment='center',
        fontweight='light',
        fontsize='x-small')
    plt.ylabel("Num Appearances")
    plt.xlabel("Class")
    plt.savefig(os.path.join(save_path, "num_appearances.png"))
    plt.show()
    seaborn.barplot(y=total_area_list, x=class_name_list, palette="viridis")
    plt.gca().invert_yaxis()
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #################
    plt.gcf().canvas.draw()
    offset = plt.gca().yaxis.get_major_formatter().get_offset()
    # offset_text = plt.gca().yaxis.get_offset_text()
    # offset_text.set_size(8)
    plt.gca().yaxis.offsetText.set_visible(False)
    # offset = plt.gca().yaxis.get_scale()
    # plt.gca().yaxis.set_label_text("original label" + " " + offset)
    ###############
    seaborn.despine(top=False, bottom=True, right=True)
    plt.xticks(
        rotation=45,
        horizontalalignment='center',
        fontweight='light',
        fontsize='x-small')
    # plt.ylabel("Total Area" + " " + "(10^9)")
    plt.ylabel("Total Area" + " " + "(" + offset + ")")
    # plt.ylabel("Total Area * 10" )
    plt.xlabel("Class")
    plt.savefig(os.path.join(save_path, "total_area.png"))
    plt.show()
    seaborn.barplot(y=avg_area_list, x=class_name_list, palette="viridis")
    plt.gca().invert_yaxis()
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    seaborn.despine(top=False, bottom=True, right=True)
    plt.xticks(
        rotation=45,
        horizontalalignment='center',
        fontweight='light',
        fontsize='x-small')
    plt.ylabel("Avg Area")
    plt.xlabel("Class")
    plt.savefig(os.path.join(save_path, "avg_area.png"))
    plt.show()
    seaborn.barplot(y=min_area_list, x=class_name_list, palette="viridis")
    plt.gca().invert_yaxis()
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    seaborn.despine(top=False, bottom=True, right=True)
    plt.xticks(
        rotation=45,
        horizontalalignment='center',
        fontweight='light',
        fontsize='x-small')
    plt.ylabel("Min Area")
    plt.xlabel("Class")
    plt.savefig(os.path.join(save_path, "min_area.png"))
    plt.show()
    seaborn.barplot(y=max_area_list, x=class_name_list, palette="viridis")
    plt.gca().invert_yaxis()
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #################
    plt.gcf().canvas.draw()
    offset = plt.gca().yaxis.get_major_formatter().get_offset()
    # offset_text = plt.gca().yaxis.get_offset_text()
    # offset_text.set_size(4)
    plt.gca().yaxis.offsetText.set_visible(False)
    # plt.gca().yaxis.set_label_text("original label" + " " + offset)
    ###############
    seaborn.despine(top=False, bottom=True, right=True)
    plt.xticks(
        rotation=45,
        horizontalalignment='center',
        fontweight='light',
        fontsize='x-small')
    # plt.ylabel("Max Area" + " " + "(10^6)")
    plt.ylabel("Max Area" + " " + "(" + offset + ")")
    # plt.ylabel("Max Area")
    plt.xlabel("Class")
    plt.savefig(os.path.join(save_path, "max_area.png"))
    plt.show()
    seaborn.barplot(y=median_area_list, x=class_name_list, palette="viridis")
    plt.gca().invert_yaxis()
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    seaborn.despine(top=False, bottom=True, right=True)
    plt.xticks(
        rotation=45,
        horizontalalignment='center',
        fontweight='light',
        fontsize='x-small')
    plt.ylabel("Median Area")
    plt.xlabel("Class")
    plt.savefig(os.path.join(save_path, "median_area.png"))
    plt.show()
    seaborn.barplot(y=rel_dataset_area_list, x=class_name_list, palette="viridis")
    plt.gca().invert_yaxis()
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    seaborn.despine(top=False, bottom=True, right=True)
    plt.xticks(
        rotation=45,
        horizontalalignment='center',
        fontweight='light',
        fontsize='x-small')
    plt.ylabel("Rel Dataset Area")
    plt.xlabel("Class")
    plt.savefig(os.path.join(save_path, "rel_dataset_area.png"))
    plt.show()

    weight_list = calc_class_weights_from_rel_distr(rel_dataset_area_list, calc_weights_method)

    print("Calc class weight list:")
    print(weight_list)
    print("Related class ID order list:")
    print(id_list)

    with open(os.path.join(save_path, 'statistics.txt'), 'w') as f:
        f.write("-"*50 + "\n")
        f.write('Num Appearances:\n')
        for i in range(len(class_name_list)):
            f.write(class_name_list[i] + ": " + str(num_appearances_list[i]) + "\n")

        f.write("-" * 50 + "\n")
        f.write('Total Area:\n')
        for i in range(len(class_name_list)):
            f.write(class_name_list[i] + ": " + str(total_area_list[i]) + "\n")

        f.write("-" * 50 + "\n")
        f.write('Avg Area:\n')
        for i in range(len(class_name_list)):
            f.write(class_name_list[i] + ": " + str(avg_area_list[i]) + "\n")

        f.write("-" * 50 + "\n")
        f.write('Min Area:\n')
        for i in range(len(class_name_list)):
            f.write(class_name_list[i] + ": " + str(min_area_list[i]) + "\n")

        f.write("-" * 50 + "\n")
        f.write('Max Area:\n')
        for i in range(len(class_name_list)):
            f.write(class_name_list[i] + ": " + str(max_area_list[i]) + "\n")

        f.write("-" * 50 + "\n")
        f.write('Median Area:\n')
        for i in range(len(class_name_list)):
            f.write(class_name_list[i] + ": " + str(median_area_list[i]) + "\n")

        f.write("-" * 50 + "\n")
        f.write('Rel Dataset Area:\n')
        for i in range(len(class_name_list)):
            f.write(class_name_list[i] + ": " + str(rel_dataset_area_list[i]) + "\n")

        f.write("-" * 50 + "\n")
        f.write("Calc class weight list:\n")
        f.write(str(weight_list) + "\n")
        f.write("-" * 50 + "\n")
        f.write("Related class ID order list:\n")
        f.write(str(id_list) + "\n")

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