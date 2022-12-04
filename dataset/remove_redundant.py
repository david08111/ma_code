import os
import argparse
import shutil

def remove_redundant(base_data_path, remove_data_path, label_mark):

    # base_data_list = [f.split(".") for f in os.listdir(base_data_path) if os.path.isfile(os.path.join(base_data_path, f))]
    base_data_list = []
    base_data_name_list = []

    remove_data_list = []
    remove_data_name_list = []

    for f in os.listdir(base_data_path):
        if os.path.isfile(os.path.join(base_data_path, f)):
            if label_mark in f:
                if f not in base_data_list:
                    base_data_list.append(f)
                if f.split("_" + label_mark)[0] not in base_data_name_list:
                    base_data_name_list.append(f.split("_" + label_mark)[0])
            else:
                if f not in base_data_list:
                    base_data_list.append(f)
                if f.split(".")[0] not in base_data_name_list:
                    base_data_name_list.append(f.split(".")[0])


    # remove_data_list = [f for f in os.listdir(copy_data_path) if
    #                   os.path.isfile(os.path.join(copy_data_path, f))]


    for f in os.listdir(remove_data_path):
        if os.path.isfile(os.path.join(remove_data_path, f)):
            if label_mark in f:
                if f not in remove_data_list:
                    remove_data_list.append(f)
                if f.split("_" + label_mark)[0] not in remove_data_name_list:
                    remove_data_name_list.append(f.split("_" + label_mark)[0])
            else:
                if f not in remove_data_list:
                    remove_data_list.append(f)
                if f.split(".")[0] not in remove_data_name_list:
                    remove_data_name_list.append(f.split(".")[0])


    # intersection_data_list = [file for file in remove_data_name_list if file in base_data_name_list]

    intersection_data_list = []
    for file_cp in remove_data_name_list:
        for file_base in base_data_name_list:
            if file_base in file_cp:
                intersection_data_list.append(file_cp)

    for file in remove_data_name_list:
        if file not in intersection_data_list:
            if os.path.isfile(os.path.join(remove_data_path, file + "." + "png")):
                os.remove(os.path.join(remove_data_path, file  + "." + "png"))
            if os.path.isfile(os.path.join(remove_data_path, file.split(".")[0] + "_" + label_mark + "." + "png")):
                os.remove(os.path.join(remove_data_path, file.split(".")[0] + "_" + label_mark + "." + "png"))



    # for file in os.listdir(remove_data_path):
    #     if os.path.isfile(os.path.join(remove_data_path, file)):
    #         if file.split(".")[0] not in intersection_data_list:
    #             os.remove(os.path.join(remove_data_path, file))
    #             if os.path.isfile(os.path.join(remove_data_path, file.split(".")[0] + "_" + label_mark + "." + "png")):
    #                 os.remove(os.path.join(remove_data_path, file.split(".")[0] + "_" + label_mark + "." + "png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bd", "--base_data_path", type=str,
                        help="Filepath to base data")
    parser.add_argument("-rmd", "--remove_data_path", type=str,
                        help="Filepath to copied data")
    parser.add_argument("-lm", "--label_mark", type=str,
                        help="Label mark that is included in the files' name to tell labels apart")
    args = parser.parse_args()

    remove_redundant(args.base_data_path, args.remove_data_path, args.label_mark)