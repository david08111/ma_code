import panopticapi
import argparse
from panopticapi import evaluation

def eval_ps(gt_json_file, pred_json_file, gt_folder, pred_folder):

    evaluation.pq_compute(gt_json_file, pred_json_file, gt_folder, pred_folder)





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
    args = parser.parse_args()
    eval_ps(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)