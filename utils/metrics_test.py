import argparse
import pandas as pd
import sys
from tqdm import tqdm

from training.metrics import Metrics_Wrapper, MetricsHandler
from utils.load_config import Config
from utils.datasets import DataHandler_Tests, custom_collate_fn_3

from torch.utils.data import DataLoader

def test_metrics_handler(metrics_dict, dataloader, ref_results):
    """
        Function to test MetricsHandler
    Args:
        metrics_handler: MetricsHandler object
        predictions: Test prediction nd array with shape: (batch_size, predictions_amount, 4+C) - C: amount of classes
        labels: Test labels nd array with shape: (batch_size, labels_amount, 10)

    Returns:
        Boolean with result of test
    """

    metrics_handler = MetricsHandler(metrics_dict)

    for batch_id, datam in enumerate(tqdm(dataloader, desc="Test_batch", file=sys.stdout)):
        [pred, labels] = datam
        metrics_handler.process_batch(pred, labels)

    metrics_handler.process_end_of_batch()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='./data/metrics_test.config',
                        help="Path to source directory")
    parser.add_argument("-d", "--data_path", type=str,
                        help="Path to prediction/label file")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Batch size")
    parser.add_argument("-w", "--num_workers", type=int,
                        help="Number of workers")
    # parser.add_argument("-p", "--prediction_path", type=str,
    #                     help="Path to prediction file")
    # parser.add_argument("-l", "--labels_path", type=str, help="Path to labels file")
    args = parser.parse_args()

    config_loader = Config()
    metrics_names_cfg_dict = config_loader(args.config)

    metrics_dict = {key: Metrics_Wrapper(metrics_names_cfg_dict["metrics"][key]) for key in
                    metrics_names_cfg_dict["metrics"]}

    # metrics_handler = MetricsHandler(metrics_dict)

    # predictions = pd.read_excel(args.prediction_path, engine="odf")
    #
    # labels = pd.read_excel(args.labels_path, engine="odf")
    dataloader = DataLoader(DataHandler_Tests(args.data_path),
                  batch_size=args.batch_size,
                  shuffle=True,
                  num_workers=args.num_workers,
                  pin_memory=True,
                  collate_fn=custom_collate_fn_3)

    ref_results = pd.read_excel(args.prediction_path, engine="odf")

    test_result = test_metrics_handler(metrics_dict, dataloader, ref_results)
    # test_result = test_metrics_handler(metrics_handler, predictions, labels)

    if test_result:
        print("Test successfull!")
    else:
        print("Test failed!")
