import os
import argparse
import shutil
import random
import math


def split_data_list(input_path, train_percentage, val_percentage, test_percentage, label_mark):

    file_list = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file)) if label_mark not in file]
    random.shuffle(file_list)
    file_list_train = file_list[0:math.floor((train_percentage / 100) * len(file_list))]
    file_list_val = file_list[
                    len(file_list_train):math.floor(((val_percentage / 100) * len(file_list))) + len(file_list_train)]
    file_list_test = file_list[
                      len(file_list_train) + len(file_list_val):]

    # for elem1 in file_list_train:
    #     if elem1 in file_list_val or elem1 in file_list_test:
    #         raise Exception("Redudandt data in list!")

    return file_list_train, file_list_val, file_list_test





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input_data_path", type=str,
                        help="Filepath to base data")
    parser.add_argument("-train_out", "--training_data_output_path", type=str,
                        help="Output filepath to train subset")
    parser.add_argument("-val_out", "--validation_data_output_path", type=str,
                        help="Output filepath to validation subset")
    parser.add_argument("-test_out", "--testset_data_output_path", type=str,
                        help="Output filepath to test subset")
    parser.add_argument("-train_p", "--train_percentage", type=str, default="70",
                        help="Percentage of train data in relation to input data")
    parser.add_argument("-val_p", "--val_percentage", type=str, default="15",
                        help="Percentage of val data in relation to input data")
    parser.add_argument("-test_p", "--test_percentage", type=str, default="15",
                        help="Percentage of test data in relation to input data")
    parser.add_argument("-lm", "--label_mark", type=str, default="mask",
                        help="Label mark that is included in the files' name to tell labels apart")
    args = parser.parse_args()

    file_list_train, file_list_val, file_list_test = split_data_list(args.input_data_path, int(args.train_percentage), int(args.val_percentage), int(args.test_percentage), args.label_mark)

    subset_list_list = [file_list_train, file_list_val, file_list_test]
    subset_output_path_list = [args.training_data_output_path, args.validation_data_output_path, args.testset_data_output_path]

    for output_path in subset_output_path_list:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    for indx in range(len(subset_output_path_list)):
        for file in subset_list_list[indx]:
            shutil.copyfile(os.path.join(args.input_data_path, file), os.path.join(subset_output_path_list[indx], file))
            shutil.copyfile(os.path.join(args.input_data_path, file.split(".")[0] + "_" + args.label_mark + "." + "json"), os.path.join(subset_output_path_list[indx], file.split(".")[0] + "_" + args.label_mark + "." + "json"))

