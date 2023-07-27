import argparse
import json
import seaborn
import pandas as pd


def visusalize_compare_eval_runs(run_comp_name, label_list, eval_file_path_list, save_path):

    df = pd.DataFrame()

    for eval_file_path in eval_file_path_list:
        with open(eval_file_path, 'r') as f:
            data_metrics_dict = json.load(f)
            for metric in data_metrics_dict:
                for key in data_metrics_dict[metric]:
                    df[metric + key] = metric[key]


    seaborn.heatmap(df)

    pass













if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, help="Name of the plot!")
    parser.add_argument("-l", "--label", type=str, action="append", help="Name of the plot!")
    parser.add_argument("-d", "--eval_file_path", type=str, action="append", help="File path to eval_metrics json")
    # parser.add_argument("-s", "--in_path", type=str, help="Input Data Dir")
    parser.add_argument("-s", "--save_path", type=str,
                        help="Output Save Path!")

    args = parser.parse_args()

    visusalize_compare_eval_runs(args.name, args.label, args.eval_file_path, args.save_path)