import argparse
from panopticapi import evaluation
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                                  Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                                  Default: X if the corresponding json file is X.json")
    parser.add_argument("-s", "--out_path", default="./outputs", type=str,
                        help="Output Data Dir")
    parser.add_argument("-n", "--out_name", default="", type=str,
                        help="Output Data Dir")
    args = parser.parse_args()
    results = evaluation.pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)

    with open(os.path.join(args.out_path, 'eval_stats.txt'), 'w') as f:
        f.write("Panoptic Segmentation Evaluation Metrics:\n")

        f.write("{:15s}| {:>5s}  {:>5s}  {:>5s} {:>5s}\n".format("", "PQ", "SQ", "RQ", "N"))
        f.write("-" * (15 + 7 * 4) + "\n")

        metrics = [("All", None), ("Things", True), ("Stuff", False)]

        for name, _isthing in metrics:
            f.write("{:15s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}\n".format(
                name,
                100 * results[name]['pq'],
                100 * results[name]['sq'],
                100 * results[name]['rq'],
                results[name]['n'])
            )

        f.write("-" * (15 + 7 * 4) + "\n")
        f.write("-" * 15 + " Class Based " + "-" * 15 + "\n")
        for cat_id in results["All_per_class"]:
            f.write("{:15s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}\n".format(
                results["categories"][cat_id]["name"],
                100 * results["All_per_class"][cat_id]['pq'],
                100 * results["All_per_class"][cat_id]['sq'],
                100 * results["All_per_class"][cat_id]['rq'],
                len(list(results["All_per_class"].keys())))
            )

        # for elem in results:
        #
        #     f.write("-"*50 + "\n")
        #     f.write('Num Appearances:\n')
        #     for i in range(len(class_name_list)):
        #         f.write(class_name_list[i] + ": " + str(num_appearances_list[i]) + "\n")

