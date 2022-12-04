import math
import warnings
from pathlib import Path
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import torch
from abc import ABC, abstractmethod


data_size_treshold = 10000

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

############## Deprecated
# class ConfusionMatrix:
#     # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
#     def __init__(self, nc, conf=0.25, iou_thres=0.45):
#         self.matrix = np.zeros((nc + 1, nc + 1))
#         self.nc = nc  # number of classes
#         self.conf = conf
#         self.iou_thres = iou_thres
#
#     def process_batch(self, detections, labels):
#         """
#         Return intersection-over-union (Jaccard index) of boxes.
#         Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#         Arguments:
#             detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#             labels (Array[M, 5]), class, x1, y1, x2, y2
#         Returns:
#             None, updates confusion matrix accordingly
#         """
#         detections = detections[detections[:, 4] > self.conf]
#         gt_classes = labels[:, 0].int()
#         detection_classes = detections[:, 5].int()
#         iou = box_iou(labels[:, 1:], detections[:, :4])
#
#         x = torch.where(iou > self.iou_thres)
#         if x[0].shape[0]:
#             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
#             if x[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#         else:
#             matches = np.zeros((0, 3))
#
#         n = matches.shape[0] > 0
#         m0, m1, _ = matches.transpose().astype(np.int16)
#         for i, gc in enumerate(gt_classes):
#             j = m0 == i
#             if n and sum(j) == 1:
#                 self.matrix[detection_classes[m1[j]], gc] += 1  # correct
#             else:
#                 self.matrix[self.nc, gc] += 1  # background FP
#
#         if n:
#             for i, dc in enumerate(detection_classes):
#                 if not any(m1 == i):
#                     self.matrix[dc, self.nc] += 1  # background FN
#
#     def matrix(self):
#         return self.matrix
#
#     def tp_fp(self):
#         tp = self.matrix.diagonal()  # true positives
#         fp = self.matrix.sum(1) - tp  # false positives
#         # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
#         return tp[:-1], fp[:-1]  # remove background class
#
#     def plot(self, normalize=True, save_dir='', names=()):
#         try:
#             import seaborn as sn
#
#             array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
#             array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
#
#             fig = plt.figure(figsize=(12, 9), tight_layout=True)
#             sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
#             labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
#                 sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
#                            xticklabels=names + ['background FP'] if labels else "auto",
#                            yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
#             fig.axes[0].set_xlabel('True')
#             fig.axes[0].set_ylabel('Predicted')
#             fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
#             plt.close()
#         except Exception as e:
#             print(f'WARNING: ConfusionMatrix plot failure: {e}')
#
#     def print(self):
#         for i in range(self.nc + 1):
#             print(' '.join(map(str, self.matrix[i])))
#
#
# class ConfusionMatrix_vision_only:
#     # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
#     def __init__(self, nc, conf=0.25, dist_thres=0.25):
#         self.matrix = np.zeros((nc + 1, nc + 1))
#         self.nc = nc  # number of classes
#         self.conf = conf
#         self.dist_thres = dist_thres
#
#     def process_batch(self, detections, labels):
#         """
#         Return intersection-over-union (Jaccard index) of boxes.
#         Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#         Arguments:
#             detections (Array[N, 6]), x, y, depth, conf, class
#             labels (Array[M, 5]), class, x1, y1, x2, y2
#         Returns:
#             None, updates confusion matrix accordingly
#         """
#         detections = detections[detections[:, 3] > self.conf]
#         gt_classes = labels[:, 0].int()  # (b,cls,x,y,depth)?
#         detection_classes = detections[:, 5].int()
#         iou = box_iou(labels[:, 1:], detections[:, :4])
#
#         x = torch.where(iou > self.iou_thres)
#         if x[0].shape[0]:
#             matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
#             if x[0].shape[0] > 1:
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                 matches = matches[matches[:, 2].argsort()[::-1]]
#                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#         else:
#             matches = np.zeros((0, 3))
#
#         n = matches.shape[0] > 0
#         m0, m1, _ = matches.transpose().astype(np.int16)
#         for i, gc in enumerate(gt_classes):
#             j = m0 == i
#             if n and sum(j) == 1:
#                 self.matrix[detection_classes[m1[j]], gc] += 1  # correct
#             else:
#                 self.matrix[self.nc, gc] += 1  # background FP
#
#         if n:
#             for i, dc in enumerate(detection_classes):
#                 if not any(m1 == i):
#                     self.matrix[dc, self.nc] += 1  # background FN
#
#     def matrix(self):
#         return self.matrix
#
#     def tp_fp(self):
#         tp = self.matrix.diagonal()  # true positives
#         fp = self.matrix.sum(1) - tp  # false positives
#         # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
#         return tp[:-1], fp[:-1]  # remove background class
#
#     def plot(self, normalize=True, save_dir='', names=()):
#         try:
#             import seaborn as sn
#
#             array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
#             array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
#
#             fig = plt.figure(figsize=(12, 9), tight_layout=True)
#             sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
#             labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
#             with warnings.catch_warnings():
#                 warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
#                 sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
#                            xticklabels=names + ['background FP'] if labels else "auto",
#                            yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
#             fig.axes[0].set_xlabel('True')
#             fig.axes[0].set_ylabel('Predicted')
#             fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
#             plt.close()
#         except Exception as e:
#             print(f'WARNING: ConfusionMatrix plot failure: {e}')
#
#     def print(self):
#         for i in range(self.nc + 1):
#             print(' '.join(map(str, self.matrix[i])))
# ##############

class Metric_Base(nn.Module):
    """
        Base class for all metrics
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        """
            Constructor
        Args:
            metric_filter_list: List containing related FilterWrapper to apply to input
            matcher: Matcher class
        """
        super().__init__()
        self.metric_filter_list = metric_filter_list
        self.base_metrics_list = []
        self.matcher = matcher

        self.metric = []
        self.filter_name_list = []
        self.filter_wrapper_name_list = []
        self.filter_value_list = []
        # self.matching_filter_list = matching_filter_list

    def forward(self, x, labels):
        """
            Metrics calculation
        Args:
            x: Input array
            labels: Target array
        """
        filter_x_list = [x]
        filter_name_list = []
        filter_wrapper_name_list = []
        filter_value_list = []

        filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list = self._apply_metric_filter_list(filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels)

        return filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list

    def _apply_metric_filter_list(self, filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels):
        """
            Applies metric filter list to all x elements
        Args:
            filter_x_list: List containing inputs
            filter_name_list: List containing List with names of applied filters for every input
            filter_wrapper_name_list: List containing List with wrapper-names of applied filters for every input
            filter_value_list: List containing List with values of applied filters for every input
            x: Input array
            labels: Target array

        Returns:
            All combinations of possible filters specified with filter names, values etc.
        """
        # filter_x_list = [x]
        # filter_name_list = []
        # filter_value_list = []

        for metric_filter in self.metric_filter_list:
            filter_x_tmp_list = []
            filter_name_tmp_list = []
            filter_wrapper_name_tmp_list = []
            filter_value_tmp_list = []
            for elem in filter_x_list:
                filter_x_tmp_list.extend(metric_filter.apply_filter_wrapper(elem, labels))

            # filter_value_list_len = metric_filter.get_filter_list_len()
            if filter_value_list:
                new_filter_metric_name = metric_filter.get_filter_name()
                new_filter_wrapper_name = metric_filter.get_wrapper_name()
                # for i in range(filter_value_list_len):
                for i in range(len(filter_value_list)):
                    for new_filter_value in metric_filter.get_filter_value_name_list():
                        new_filter_value_sublist = filter_value_list[i].copy()
                        new_filter_value_sublist.append(new_filter_value)
                        filter_value_tmp_list.append(new_filter_value_sublist)

                        new_filter_name_sublist = filter_name_list[i].copy()
                        new_filter_name_sublist.append(new_filter_metric_name)
                        filter_name_tmp_list.append(new_filter_name_sublist)

                        new_wrapper_name_sublist = filter_wrapper_name_list[i].copy()
                        new_wrapper_name_sublist.append(new_filter_wrapper_name)
                        filter_wrapper_name_tmp_list.append(new_wrapper_name_sublist)
            else:
                filter_value_tmp_list = [[elem] for elem in metric_filter.get_filter_value_name_list()]
                # for i in range(filter_value_list_len):
                for i in range(len(filter_value_tmp_list)):
                    filter_name_tmp_list.append([metric_filter.get_filter_name()])
                    filter_wrapper_name_tmp_list.append([metric_filter.get_wrapper_name()])

            filter_x_list = filter_x_tmp_list
            filter_name_list = filter_name_tmp_list
            filter_wrapper_name_list = filter_wrapper_name_tmp_list
            filter_value_list = filter_value_tmp_list

        filter_x_list_final = []
        filter_name_list_final = []
        filter_wrapper_name_list_final = []
        filter_value_list_final = []

        for i in range(len(filter_x_list)):
            if filter_x_list[i].any():
                filter_x_list_final.append(filter_x_list[i])
                filter_name_list_final.append(filter_name_list[i])
                filter_wrapper_name_list_final.append(filter_wrapper_name_list[i])
                filter_value_list_final.append(filter_value_list[i])

        return filter_x_list_final, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final

    # def _apply_matching_filter_list(self, filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels):
    #     #combine with metric_filter_list applying?
    #
    #     for metric_filter in self.matching_filter_list:
    #         filter_x_tmp_list = []
    #         filter_wrapper_name_tmp_list = []
    #         filter_name_tmp_list = []
    #         filter_value_tmp_list = []
    #         for elem in filter_x_list:
    #             metric_matrix_list, binary_filter_label_indx_list, matching_x_indx_list = metric_filter.apply_filter_wrapper(elem, labels)
    #
    #             for matching_indx in matching_x_indx_list:
    #                 filter_x_tmp_list.append(elem[matching_indx])
    #             # filter_x_tmp_list.extend()
    #
    #         # filter_value_list_len = metric_filter.get_filter_list_len()
    #         if filter_value_list:
    #             new_filter_metric_name = metric_filter.get_filter_name()
    #             new_filter_wrapper_name = metric_filter.get_wrapper_name()
    #             # for i in range(filter_value_list_len):
    #             for i in range(len(filter_value_list)):
    #                 for new_filter_value in metric_filter.get_filter_value_name_list():
    #                     new_filter_value_sublist = filter_value_list[i].copy()
    #                     new_filter_value_sublist.append(new_filter_value)
    #                     filter_value_tmp_list.append(new_filter_value_sublist)
    #
    #                     new_filter_name_sublist = filter_name_list[i].copy()
    #                     new_filter_name_sublist.append(new_filter_metric_name)
    #                     filter_name_tmp_list.append(new_filter_name_sublist)
    #
    #                     new_wrapper_name_sublist = filter_wrapper_name_list[i].copy()
    #                     new_wrapper_name_sublist.append(new_filter_wrapper_name)
    #                     filter_wrapper_name_tmp_list.append(new_wrapper_name_sublist)
    #         else:
    #             filter_value_tmp_list = [[elem] for elem in metric_filter.get_filter_value_name_list()]
    #             # for i in range(filter_value_list_len):
    #             for i in range(len(filter_value_tmp_list)):
    #                 filter_name_tmp_list.append([metric_filter.get_filter_name()])
    #                 filter_wrapper_name_tmp_list.append([metric_filter.get_wrapper_name()])
    #
    #         filter_x_list = filter_x_tmp_list
    #         filter_name_list = filter_name_tmp_list
    #         filter_wrapper_name_list = filter_wrapper_name_tmp_list
    #         filter_value_list = filter_value_tmp_list
    #
    #     filter_x_list_final = []
    #     filter_name_list_final = []
    #     filter_wrapper_name_list_final = []
    #     filter_value_list_final = []
    #
    #     for i in range(len(filter_x_list)):
    #         if filter_x_list[i].any():
    #             filter_x_list_final.append(filter_x_list[i])
    #             filter_name_list_final.append(filter_name_list[i])
    #             filter_wrapper_name_list_final.append(filter_wrapper_name_list[i])
    #             filter_value_list_final.append(filter_value_list[i])
    #
    #     return filter_x_list_final, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final

    def get_base_metrics_list(self):
        return self.base_metrics_list

    def get_metric_filter_list(self):
        return self.metric_filter_list

    def get_metric(self):
        return self.metric

    def get_filter_name_list(self):
        return self.filter_name_list

    def get_filter_wrapper_name_list(self):
        return self.filter_wrapper_name_list

    def get_filter_value_list(self):
        return self.filter_value_list


    # def get_matching_filter_list(self):
    #     return self.matching_filter_list

    def __eq__(self, other):

        if type(other) == type(self):   ### for self -> Metric_Base or Child class?
            own_base_metric_list = self.base_metrics_list.copy()
            other_base_metric_list = other.get_base_metrics_list().copy()
            for own_base_metric in own_base_metric_list:
                for other_base_metric in other_base_metric_list:
                    if type(own_base_metric) == type(other_base_metric):
                        if own_base_metric.get_base_metrics_list() and other.get_base_metrics_list():
                            if own_base_metric == other_base_metric:
                                other_base_metric_list.remove(other_base_metric)
                                break
                        elif not own_base_metric.get_base_metrics_list() and not other_base_metric.get_base_metrics_list():
                            return self._cmp_object_filter_lists(own_base_metric.get_metric_filter_list(), other_base_metric.get_metric_filter_list()) #and self._cmp_object_filter_lists(own_base_metric.get_matching_filter_list(), other_base_metric.get_matching_filter_list())
                        else:
                            continue
                if other_base_metric_list:
                    if other_base_metric == other_base_metric_list[-1]: ####  buggy behaviour?
                        return False

            return self._cmp_object_filter_lists(other.get_metric_filter_list(), self.get_metric_filter_list()) #and self._cmp_object_filter_lists(other.get_matching_filter_list(), self.get_matching_filter_list())

        return False

    def _cmp_object_filter_lists(self, metric1_filter_list, metric2_filter_list):
        """
            Compares two filter lists and gives back if equal
        Args:
            metric1_filter_list: List containing FilterWrappers
            metric2_filter_list: List containing FilterWrappers

        Returns:
            Bool type if lists equal
        """
        tmp_list = metric2_filter_list.copy()
        if (metric1_filter_list and metric2_filter_list) or (not metric1_filter_list and not metric2_filter_list):
            for filter1 in metric1_filter_list:
                for filter2 in tmp_list:
                    if filter1 == filter2:
                        tmp_list.remove(filter2)
                        break
                if filter2 == metric2_filter_list[-1] and tmp_list:
                    return False
            return True
        else:
            return False

    @abstractmethod
    def process_end_of_batch(self):
        """ abstract method """

class ConfusionMatrix(Metric_Base):
    """
        Confusion Matrix calculation
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)

    def forward(self, outputs, labels):
        filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list = super().forward(outputs, labels)

        confusion_matrix, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final = self.matcher.match2confusionMatrix(
            filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, outputs, labels)

        if self.metric:
            for i in range(len(self.metric)):
                self.metric[i] += confusion_matrix[i]
        else:
            self.metric = confusion_matrix

        if not self.filter_name_list:
            self.filter_name_list = filter_name_list_final
        if not self.filter_wrapper_name_list:
            self.filter_wrapper_name_list = filter_wrapper_name_list_final
        if not self.filter_value_list:
            self.filter_value_list = filter_value_list_final

    def process_end_of_batch(self):
        pass

class Accuracy(Metric_Base):
    """
        Accuracy calculation: (TP+TN)/(TP+TN+FN+FP)

        Notice: TN for object detection is infinity (in Classification_cases simply set to 0) - therefore not reasonable as metric
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        accuracy_list = []
        for confusion_matrix in confusion_matrix_list:
            accuracy = np.zeros((confusion_matrix.shape[0] - 1))
            indx_list = [*range(confusion_matrix.shape[0])]
            for i in range(confusion_matrix.shape[0] - 1):
                tp = confusion_matrix[i, i]
                tn = confusion_matrix[confusion_matrix.shape[0], confusion_matrix.shape[0]]
                add_list = indx_list.copy()
                div = 0
                for j in add_list:
                    div += confusion_matrix[j, i]
                    div += confusion_matrix[i, j]
                div -= confusion_matrix[i, i]
                div += tn
                if div == 0:
                    accuracy[i] = 0
                else:
                    accuracy[i] = (tp + tn) / div

            accuracy_list.append(accuracy)

        self.metric = accuracy_list

class Precision(Metric_Base):
    """
        Precision calculation: Precision = TP / (TP + FP)
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()
        precision_list = []
        for confusion_matrix in confusion_matrix_list:
            precisions = np.zeros((confusion_matrix.shape[0] - 1))
            indx_list = [*range(confusion_matrix.shape[0])]
            for i in range(confusion_matrix.shape[0] - 1):
                tp = confusion_matrix[i, i]
                add_list = indx_list.copy()
                div = 0
                for j in add_list:
                    div += confusion_matrix[j, i]
                if div == 0:
                    precisions[i] = 0
                else:
                    precisions[i] = tp / div

            precision_list.append(precisions)

        self.metric = precision_list

class Recall(Metric_Base):
    """
        Recall calculation: Recall = TP / (TP + FN)
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        recall_list = []
        for confusion_matrix in confusion_matrix_list:
            recall = np.zeros((confusion_matrix.shape[0] - 1))

            for i in range(confusion_matrix.shape[0] - 1):
                tp = confusion_matrix[i, i]
                fn = confusion_matrix[i, -1]
                div = tp + fn

                if div == 0:
                    recall[i] = 0
                else:
                    recall[i] = tp / div

            recall_list.append(recall)

        self.metric = recall_list

class Class_PR_Calc(Metric_Base):
    """
        Class-Precision-Recall calculation - calculates List containing [TP, FP, matching-criterion-value]
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)

    def forward(self, outputs, labels):
        filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list = super().forward(outputs, labels)

        tp_fp_case_list, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final = self.matcher.match2tp_fp_crit_lists(filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, outputs, labels)

        if self.metric:
            for i in range(len(self.metric)):
                for j in range(len(self.metric[i])):
                    self.metric[i][j].extend(tp_fp_case_list[i][j])
        else:
            self.metric = tp_fp_case_list

        if not self.filter_name_list:
            self.filter_name_list = filter_name_list_final
        if not self.filter_wrapper_name_list:
            self.filter_wrapper_name_list = filter_wrapper_name_list_final
        if not self.filter_value_list:
            self.filter_value_list = filter_value_list_final

    def process_end_of_batch(self):

        metric_tmp_list = [[] for foo in range(len(self.metric))]

        for i in range(len(self.metric)):
            local_metric_tmp_list = [[] for foo in range(len(self.metric[i]))]
            for j in range(len(self.metric[i])):
                elem_arr = np.array(self.metric[i][j])
                local_metric_tmp_list[j] = elem_arr[np.argsort(elem_arr[:, 2])]

            metric_tmp_list[i] = local_metric_tmp_list

        self.metric = metric_tmp_list


# class Precision_AP(Metric_Base):
#     def __init__(self, iou_threshold=0.5, conf_threshold=0.5):
#         super().__init__()
#         self.iou_threshold = iou_threshold
#         # self.num_conf_thresholds = num_conf_thresholds
#         self.conf_threshold = conf_threshold
#         self.value_list = []
#         self.recall_div = 0
#
#
#     def calc(self, pr_classification_list):
#         for iou_thresh_indx in range(len(pr_classification_list)):
#             precision_class_list =[]
#             for class_list_indx in range(len(pr_classification_list[iou_thresh_indx].value_list)):
#                 class_list_temp = pr_classification_list[iou_thresh_indx].value_list[class_list_indx]
#                 class_list_array = np.array(class_list_temp)
#
#                 conf_outputs_ind_sort = np.argsort(class_list_array, axis=0) ## aufsteigend
#                 conf_outputs_ind_sort_list = class_list_array[conf_outputs_ind_sort[:, 0], :]
#
#                 tp_temp = 0
#                 fp_temp = 0
#                 prec_temp_list = []
#                 for i in range(len(conf_outputs_ind_sort_list) - 1 , -1, -1):
#                     if conf_outputs_ind_sort_list[i, 1] == 1:
#                         tp_temp += 1
#                     elif conf_outputs_ind_sort_list[i, 2] == 1:
#                         fp_temp += 1
#                     precision_temp = tp_temp / (tp_temp + fp_temp)
#                     prec_temp_list.append(precision_temp)
#
#                 precision_class_list.append(prec_temp_list)
#
#             self.value_list.append(precision_class_list)

class Precision_AP(Metric_Base):
    """
        Precision calculation for AP - calculates List containing precision value for every descending, filtered input value

        Notice: Not class based - for class based add class threshold filters
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [Class_PR_Calc(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        tp_fp_case_list = self.base_metrics_list[0].get_metric()

        precision_list = []


        for tp_fp_case_arr in tp_fp_case_list:
            prec_temp_list = [[] for foo in range(len(tp_fp_case_arr))]
            for class_indx in range(len(tp_fp_case_arr)):
                tp_temp = 0
                fp_temp = 0
                # prec_temp_list = []
                for i in range(len(tp_fp_case_arr) - 1, -1, -1):
                    if tp_fp_case_arr[i, 1] == 1:
                        tp_temp += 1
                    elif tp_fp_case_arr[i, 2] == 1:
                        fp_temp += 1
                    precision_temp = tp_temp / (tp_temp + fp_temp)
                    prec_temp_list[class_indx].append(precision_temp)

            precision_list.append(prec_temp_list)

        # for tp_fp_case_arr in tp_fp_case_list:
        #     tp_temp = 0
        #     fp_temp = 0
        #     prec_temp_list = []
        #     for i in range(len(tp_fp_case_arr) - 1, -1, -1):
        #         if tp_fp_case_arr[i, 1] == 1:
        #             tp_temp += 1
        #         elif tp_fp_case_arr[i, 2] == 1:
        #             fp_temp += 1
        #         precision_temp = tp_temp / (tp_temp + fp_temp)
        #         prec_temp_list.append(precision_temp)
        #
        #     precision_list.append(prec_temp_list)

        self.metric = precision_list



# class Recall_AP(Metric_Base):
#     def __init__(self, iou_threshold=0.5, conf_threshold=0.5):
#         super(Recall_AP, self).__init__()
#         self.iou_threshold = iou_threshold
#         # self.num_conf_thresholds = num_conf_thresholds
#         self.conf_threshold = conf_threshold
#         self.value_list = []
#         # self.recall_div_class_list = []
#
#     def calc(self, pr_classification_list):
#         for iou_thresh_indx in range(len(pr_classification_list)):
#             recall_class_list = []
#             for class_list_indx in range(len(pr_classification_list[iou_thresh_indx].value_list)):
#                 class_list_temp = pr_classification_list[iou_thresh_indx].value_list[class_list_indx]
#                 class_list_array = np.array(class_list_temp)
#
#                 conf_outputs_ind_sort = np.argsort(class_list_array, axis=0)  ## aufsteigend
#                 conf_outputs_ind_sort_list = class_list_array[conf_outputs_ind_sort[:, 0], :]
#
#                 tp_temp = 0
#
#                 rec_temp_list = []
#                 for i in range(len(conf_outputs_ind_sort_list) - 1 , -1, -1):
#                     if conf_outputs_ind_sort_list[i, 1] == 1:
#                         tp_temp += 1
#                     recall_temp = tp_temp / pr_classification_list[iou_thresh_indx].recall_div_class_list[class_list_indx]
#                     rec_temp_list.append(recall_temp)
#
#                 recall_class_list.append(rec_temp_list)
#
#             self.value_list.append(recall_class_list)

class Recall_AP(Metric_Base):
    """
        Recall calculation for AP - calculates List containing recall value for every descending, filtered input value

        Notice: Not class based - for class based add class threshold filters
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [Class_PR_Calc(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        tp_fp_case_list = self.base_metrics_list[0].get_metric()

        recall_list = []

        for tp_fp_case_arr in tp_fp_case_list:
            rec_temp_list = [[] for foo in range(len(tp_fp_case_arr))]
            for class_indx in range(len(tp_fp_case_arr)):
                tp_temp = 0

                # rec_temp_list = []

                gt_samples_div = np.sum(tp_fp_case_arr[:, 1])

                for i in range(len(tp_fp_case_arr) - 1, -1, -1):
                    if tp_fp_case_arr[i, 1] == 1:
                        tp_temp += 1
                    recall_temp = tp_temp / gt_samples_div
                    rec_temp_list[class_indx].append(recall_temp)

            recall_list.append(rec_temp_list)

        # for tp_fp_case_arr in tp_fp_case_list:
        #     tp_temp = 0
        #
        #     rec_temp_list = []
        #
        #     gt_samples_div = np.sum(tp_fp_case_arr[:, 1])
        #
        #     for i in range(len(tp_fp_case_arr) - 1, -1, -1):
        #         if tp_fp_case_arr[i, 1] == 1:
        #             tp_temp += 1
        #         recall_temp = tp_temp / gt_samples_div
        #         rec_temp_list.append(recall_temp)
        #
        #     recall_list.append(rec_temp_list)

        self.metric = recall_list



# class AveragePrecision(Metric_Base):
#     def __init__(self, iou_threshold, conf_threshold):
#         super(AveragePrecision, self).__init__()
#         self.iou_threshold = iou_threshold
#         self.conf_threshold = conf_threshold
#         self.value_list = []
#
#     def calc(self, recall, precision):
#         param_pair_list = []
#         for i in range(len(recall)):
#             ap_class_list = []
#             for class_indx in range(len(recall[i])):
#                 ap, mpre, mrec = compute_ap(recall[i][class_indx], precision[i][class_indx])
#                 ap_class_list.append(ap)
#
#             param_pair_list.append(ap_class_list)
#         ap_array = np.array(param_pair_list)
#         ap_mean = np.mean(ap_array, axis=0)
#
#         self.value_list = ap_mean.tolist()
#
#
#         # return ap

class AveragePrecision(Metric_Base):
    """
        Average Precision (AP) calculation

        Notice: Not class based (by definition already mAP) - for class based add class threshold filters
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [Precision_AP(metric_filter_list, matcher, *args, **kwargs), Recall_AP(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        precision_list = self.base_metrics_list[0].get_metric()
        recall_list = self.base_metrics_list[1].get_metric()

        ap_list = [[] for foo in range(len(recall_list))]
        for i in range(len(recall_list)):
            ap_list_tmp = [[] for foo in range(len(recall_list[i]))]
            for j in range(len(recall_list[i])):
                ap, mpre, mrec = compute_ap(recall_list[i][j], precision_list[i][j])
                ap_list_tmp[j] = ap

            ap_list[i] = ap_list_tmp

        self.metric = ap_list


# class meanAveragePrecision(Metric_Base):
#     def __init__(self, iou_threshold, conf_threshold):
#         super(meanAveragePrecision, self).__init__()
#         self.iou_threshold = iou_threshold
#         self.conf_threshold = conf_threshold
#         self.value_list = []
#
#     def calc(self, average_precision_list):
#         sum = 0
#         for i in range(len(average_precision_list)):
#             sum += average_precision_list[i]
#
#         self.value_list.append(sum/len(average_precision_list))

class meanAveragePrecision(Metric_Base):
    """
        mean Average Precision (mAP) calculation

        Notice: If no filter set for AP then AP is already mAP by definition
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [AveragePrecision(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        ap_list = self.base_metrics_list[0].get_metric()

        map_list = [[] for foo in range(len(ap_list))]
        for i in range(len(ap_list)):
            # ap_list_tmp = [[] for foo in range(len(ap_list[i]))]
            map_sum = 0
            class_amount = len(ap_list[i])

            for j in range(class_amount):
                map_sum += ap_list[i][j]

            map_sum /= class_amount

            map_list[i] = map_sum

        self.metric = map_list



# class F1_Score(Metric_Base):
#     def __init__(self, iou_threshold=0.5, num_conf_thresholds=50):
#         super(F1_Score, self).__init__()
#         self.iou_threshold = iou_threshold
#         self.num_conf_thresholds = num_conf_thresholds
#         self.value_list = []
#
#     def forward(self, outputs, labels):
#         pass
#
#     def calc(self, precision, recall):
#         return 2 * (precision * recall) / (precision + recall)

class F1_Score(Metric_Base):
    """
        F1-Score calculation: F1-Score = 2 * precision * recall / (precision + recall)
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [Precision(metric_filter_list, matcher, *args, **kwargs), Recall(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        precision_list = self.base_metrics_list[0].get_metric()
        recall_list = self.base_metrics_list[1].get_metric()
        f1_score_list = []

        for i in range(len(precision_list)):
            f1_score = []
            for k in range(precision_list[i].shape[0]):
                f1_score.append(2 * precision_list[i][k] * recall_list[i][k] / (precision_list[i][k] + recall_list[i][k]))

                f1_score_list.append(f1_score)

        self.metric = f1_score_list

class F_Beta_Score(Metric_Base):
    """
        F-Beta Score calculation: F-Beta-Score = (1+beta^2) * precision * recall / ((beta^2 * precision) + recall)
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [Precision(metric_filter_list, matcher, *args, **kwargs), Recall(metric_filter_list, matcher, *args, **kwargs)]
        if "beta" in kwargs:
            self.beta = kwargs["beta"]
        else:
            raise("No Beta value for F-Beta score specified!")

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        precision_list = self.base_metrics_list[0].get_metric()
        recall_list = self.base_metrics_list[1].get_metric()
        f_beta_score_list = []

        for i in range(len(precision_list)):
            f_beta_score = []
            for k in range(precision_list[i].shape[0]):
                f_beta_score.append(((1 + self.beta) ** 2) * precision_list[i][k] * recall_list[i][k] / ((self.beta ** 2) * precision_list[i][k] + recall_list[i][k]))

                f_beta_score_list.append(f_beta_score)

        self.metric = f_beta_score_list

class False_Pos_Rate(Metric_Base):
    """
        False-Positive-Rate calculation: False-Positive-Rate = FP / (FP + TN)

        Notice: TN for object detection is infinity (in Classification_cases simply set to 0) - therefore not reasonable as metric
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        false_pos_rate_list = []
        for confusion_matrix in confusion_matrix_list:
            false_pos_rate = np.zeros((confusion_matrix.shape[0] - 1))
            indx_list = [*range(confusion_matrix.shape[0])]
            for i in range(confusion_matrix.shape[0] - 1):
                fp = 0
                add_list = indx_list.copy()
                add_list.remove(i)
                tn = confusion_matrix[-1, -1]

                for j in add_list:
                    fp += confusion_matrix[j, i]

                div = tn + fp

                if div == 0:
                    false_pos_rate[i] = 0
                else:
                    false_pos_rate[i] = fp / div

            false_pos_rate_list.append(false_pos_rate)

        self.metric = false_pos_rate_list

class False_Neg_Rate(Metric_Base):
    """
        False-negative-rate calculation: False-Negative-Rate = FN / (FN + TP)

        Notice: TN for object detection is infinity (in Classification_cases simply set to 0) - therefore not reasonable as metric
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        false_neg_rate_list = []
        for confusion_matrix in confusion_matrix_list:
            false_neg_rate = np.zeros((confusion_matrix.shape[0] - 1))
            for i in range(confusion_matrix.shape[0] - 1):
                fn = confusion_matrix[-1, i]
                tp = confusion_matrix[i, i]

                div = tp + fn

                if div == 0:
                    false_neg_rate[i] = 0
                else:
                    false_neg_rate[i] = fn / div

            false_neg_rate_list.append(false_neg_rate)

        self.metric = false_neg_rate_list

class True_Neg_Rate(Metric_Base):
    """
        True-negative-rate calculation: True-Negative-Rate = TN / (TN + FP)

        Notice: TN for object detection is infinity (in Classification_cases simply set to 0) - therefore not reasonable as metric
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        true_neg_rate_list = []
        for confusion_matrix in confusion_matrix_list:
            true_neg_rate = np.zeros((confusion_matrix.shape[0] - 1))
            indx_list = [*range(confusion_matrix.shape[0])]
            for i in range(confusion_matrix.shape[0] - 1):
                tn = confusion_matrix[-1, -1]
                add_list = indx_list.copy()
                add_list.remove(i)

                div = 0

                for j in add_list:
                    div += confusion_matrix[j, i]
                div += tn

                if div == 0:
                    true_neg_rate[i] = 0
                else:
                    true_neg_rate[i] = tn / div

            true_neg_rate_list.append(true_neg_rate)

        self.metric = true_neg_rate_list

class False_Discovery_Rate(Metric_Base):
    """
        False-Discovery-Rate calculation: FP / (TP + FP)
    """
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        false_discovery_rate_list = []
        for confusion_matrix in confusion_matrix_list:
            false_discovery_rate = np.zeros((confusion_matrix.shape[0] - 1))
            indx_list = [*range(confusion_matrix.shape[0])]
            for i in range(confusion_matrix.shape[0] - 1):
                add_list = indx_list.copy()
                add_list.remove(i)

                tp = confusion_matrix[i, i]

                fp = 0

                for j in add_list:
                    fp += confusion_matrix[j, i]

                div = tp + fp

                if div == 0:
                    false_discovery_rate[i] = 0
                else:
                    false_discovery_rate[i] = fp / div

            false_discovery_rate_list.append(false_discovery_rate)

        self.metric = false_discovery_rate_list


class False_Omission_Rate(Metric_Base):
    """
        False-Omission-Rate calculation: FN / (FN + TN)
    """

    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        false_omission_rate_list = []
        for confusion_matrix in confusion_matrix_list:
            false_omission_rate = np.zeros((confusion_matrix.shape[0] - 1))

            for i in range(confusion_matrix.shape[0] - 1):

                fn = confusion_matrix[i, i]

                tn = confusion_matrix[-1, -1]

                div = fn + tn

                if div == 0:
                    false_omission_rate[i] = 0
                else:
                    false_omission_rate[i] = fn / div

            false_omission_rate_list.append(false_omission_rate)

        self.metric = false_omission_rate_list


class Negative_Predictive_Value(Metric_Base):
    """
        Negative-Predictive-Value calculation: TN / (FN + TN)
    """

    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

        confusion_matrix_list = self.base_metrics_list[0].get_metric()

        negative_predictive_value_list = []
        for confusion_matrix in confusion_matrix_list:
            negative_predictive_value = np.zeros((confusion_matrix.shape[0] - 1))

            for i in range(confusion_matrix.shape[0] - 1):

                fn = confusion_matrix[i, i]

                tn = confusion_matrix[-1, -1]

                div = fn + tn

                if div == 0:
                    negative_predictive_value[i] = 0
                else:
                    negative_predictive_value[i] = tn / div

            negative_predictive_value_list.append(negative_predictive_value)

        self.metric = negative_predictive_value_list

class AverageRecall(Metric_Base):
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [ConfusionMatrix(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        super().forward(outputs, labels)

        return None

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

class meanAverageRecall(Metric_Base):
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [AverageRecall(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        super().forward(outputs, labels)

        return None

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

class ROC(Metric_Base):
    def __init__(self, metric_filter_list, matcher, *args, **kwargs):
        super().__init__(metric_filter_list, matcher, *args, **kwargs)
        self.base_metrics_list = [Precision(metric_filter_list, matcher, *args, **kwargs), Recall(metric_filter_list, matcher, *args, **kwargs)]

    def forward(self, outputs, labels):
        pass

    def get_auc(self):
        pass

    def process_end_of_batch(self):
        self.filter_name_list = self.base_metrics_list[0].get_filter_name_list()
        self.filter_wrapper_name_list = self.base_metrics_list[0].get_filter_wrapper_name_list()
        self.filter_value_list = self.base_metrics_list[0].get_filter_value_list()

###########################

class Filter_Wrapper_Base():
    """
        Base class to derive FilterWrapper classes from
    """
    def __init__(self, filter_name, threshold_list, filter_mode="greq", *args, **kwargs):
        """
            Constructor
        Args:
            filter_name: String containing filter name
            threshold_list: List containing threshold values
            mode: String containing filter comparison mode (e.g. greater - "gr" etc.)
            *args: Optional args
            **kwargs: optional kwargs
        """
        self.filter_name = filter_name
        self.filter_list = threshold_list
        self.filter_mode = filter_mode

        self._set_filter(filter_name, args, kwargs)

    def _set_filter(self, filter_name, *args, **kwargs):
        """
            Sets filter by name
        Args:
            filter_name: String containing name of the filter
            *args: Optional args
            **kwargs: optional kwargs
        """
        if "iou" in filter_name:
            self.filter = IOU_Filter()
        elif "confidence" in filter_name:
            self.filter = Confidence_Filter()
        elif "distance3d" in filter_name:
            self.filter = Distance3D_Filter()
        elif "depth" in filter_name:
            self.filter = Depth_filter()
        elif "height" in filter_name:
            self.filter = Height_Filter()
        elif "positionx" in filter_name:
            self.filter = PositionX_Filter()
        elif "positiony" in filter_name:
            self.filter = PositionY_Filter()
        elif "class" in filter_name:
            self.filter = Class_Filter()
        else:
            raise(filter_name + "Filter not availabe!")


    def get_filter_name(self):
        return self.filter_name

    @abstractmethod
    def get_wrapper_name(self):
        """ abstract method """

    def get_filter_list(self):
        return self.filter_list

    def get_mode(self):
        return self.mode

    # def _get_filter_array_indx(self, tensor_name_list):
    #     filter_column_indx = None
    #     filter_dim_indx = None
    #     for dim_indx in range(len(tensor_name_list)):
    #         if tensor_name_list[dim_indx].index(self.filter):
    #             filter_column_indx = tensor_name_list[dim_indx].index(self.filter)
    #             filter_dim_indx = dim_indx
    #             return filter_dim_indx, filter_column_indx
    #     #x_column_indx = [range(tmp) for tmp in x.shape]
    #
    #     return None, None
    #     #return column_name_list.index(self.filter)

    @abstractmethod
    def apply_filter_wrapper(self, x, label):
        """ abstract method """

    def __eq__(self, other):
        if type(other) == type(self):
            if other.get_filter_list() == self.get_filter_list():
                return True

        return False

    def get_filter_list_len(self):
        return len(self.filter_list)

    def get_filter_value_name_list(self):
        return self.filter_list

#Range Selection Wrapper applies filter on different ranges (e.g. from 0 to 1 in 0.2 steps - 0-0.2;0.2-0.4 ...)
class Range_Selection_Filter_Wrapper(Filter_Wrapper_Base):
    """
        Wrapper class to filter for specified range
    """
    def __init__(self, filter_name, filter_list, filter_mode="greq", *args, **kwargs):
        """
            Constructor
        Args:
            filter_name: String containing filter name
            threshold_list: List containing threshold values
            mode: String containing filter comparison mode (e.g. greater - "gr" etc.)
            *args: Optional args
            **kwargs: optional kwargs
        """
        super().__init__(filter_name, filter_list, filter_mode, *args, **kwargs)

    def apply_filter_wrapper(self, x, label):

        filtered_x = []

        for i in range(len(self.filter_list) - 1):
            # x_column_indx = [range(tmp) for tmp in x.shape]
            filtered_low = self.filter.apply_filter(x, label, self.filter_list[i], mode="greq")
            filtered_high = self.filter.apply_filter(filtered_low, label, self.filter_list[i + 1], mode="lw")

            # filtered_concat_indx = np.where(filtered_low == filtered_high) # ??
            filtered_x.append(filtered_high)

        return filtered_x

    def get_filter_list_len(self):
        return len(self.filter_list) - 1

    def get_filter_value_name_list(self):
        filter_value_name_list = []

        for i in range(len(self.filter_list) - 1):
            filter_value_name_list.append(str(self.filter_list[i]) + "-" + str(self.filter_list[i+1]))

        return filter_value_name_list

    def get_wrapper_name(self):
        return "range_sel"

#Threshold Wrapper applies classical threshold filtering ( > filter stays) to specified filter metric
class Threshold_Filter_Wrapper(Filter_Wrapper_Base):
    """
        Wrapper class to filter by threshold
    """
    def __init__(self, filter_name, filter_list, filter_mode="greq", *args, **kwargs):
        """
            Constructor
        Args:
            filter_name: String containing filter name
            threshold_list: List containing threshold values
            mode: String containing filter comparison mode (e.g. greater - "gr" etc.)
            *args: Optional args
            **kwargs: optional kwargs
        """

        super().__init__(filter_name, filter_list, filter_mode, *args, **kwargs)

    def apply_filter_wrapper(self, x, label):
        filtered_x = []

        for i in range(len(self.filter_list)):
            filtered_x.append(self.filter.apply_filter(x, label, self.filter_list[i], self.filter_mode))


        return filtered_x

    def get_wrapper_name(self):
        return "thresh"

class Matching_Filter_Wrapper(Filter_Wrapper_Base):
    ## add criteria vector and criteria mode
    """
        Wrapper class to match inputs to targets and give filter matrix (of input-labels metric - zero entries for every
        filtered combination), input filter indices and matching filter indices afterwards (1 to 1 matching)
    """
    def __init__(self, filter_name, filter_list, matching_crit, filter_mode="greq", matching_mode="gr", *args, **kwargs):
        """
            Constructor
        Args:
            filter_name: String containing filter name
            filter_list: List containing threshold values
            matching_crit:
            filter_mode: String containing filter comparison mode (e.g. greater - "gr" etc.)
            *args: Optional args
            **kwargs: optional kwargs
        """
        super().__init__(filter_name, filter_list, filter_mode, *args, **kwargs)

        # self.filter_mode = filter_mode
        self.matching_filter_crit = matching_crit
        self.matching_mode = matching_mode

    def apply_filter_wrapper(self, x, labels):
        if labels.shape[0] > data_size_treshold or x.shape[0] > data_size_treshold:
            warnings.warn("Matching Wrapper Array size could be too big.")

        matching_crit_vector = self.matching_filter_crit.apply_filter_wrapper(x, labels)

        filtered_x = []


        for i in range(len(self.filter_list)):
            metric_matrix, binary_filter_label_indx, matching_x_indx = self.filter.apply_matching_filter(x, labels, self.filter_list[i], matching_crit_vector, self.filter_mode, self.matching_mode)

            filtered_x_tmp = x.copy()
            filtered_x.append(filtered_x_tmp[matching_x_indx])

        return filtered_x

    def apply_filter_and_match(self, x, labels):
        if labels.shape[0] > data_size_treshold or x.shape[0] > data_size_treshold:
            warnings.warn("Matching Wrapper Array size could be too big.")

        matching_crit_vector = self.matching_filter_crit.apply_filter_wrapper(x, labels)

        # filtered_x = []

        filtered_metric_matrix_list = []

        binary_filter_label_indx_list = []

        matching_x_indx_list = []


        for i in range(len(self.filter_list)):
            metric_matrix, binary_filter_label_indx, matching_x_indx = self.filter.apply_matching_filter(x, labels, self.filter_list[i], matching_crit_vector, self.filter_mode, self.matching_mode)

            filtered_metric_matrix_list.append(metric_matrix)
            binary_filter_label_indx_list.append(binary_filter_label_indx)
            matching_x_indx_list.append(matching_x_indx)
            # filtered_x_tmp = x.copy()
            # filtered_x.append(filtered_x_tmp[matching_x_indx])

        return filtered_metric_matrix_list, binary_filter_label_indx_list, matching_x_indx_list

    def get_wrapper_name(self):
        return "matching"


class Identity_Filter_Wrapper(Filter_Wrapper_Base):
    def __init__(self, filter_name, filter_list=[], filter_mode="greq", *args, **kwargs):
        super().__init__(filter_name, filter_list, filter_mode, *args, **kwargs)


    def apply_filter_wrapper(self, x, labels):
        return self.filter.get_unfiltered(x, labels)

    def get_wrapper_name(self):
        return "identity"


class Filter_Base():
    """
        Abstract filter base class to derive filter Classes from
    """
    def __init__(self):
        if self.__class__.__name__ == "Filter_Base":
            raise NotImplementedError("You can't instantiate this abstract class.")

    @abstractmethod
    def apply_filter(self, x, labels, filter, filter_mode="lweq"):
        """
            Applies the filter class on the input x - gives filtered x back
        Args:
            x: Input
            filter: Filter value
            mode: Filter mode

        Returns:
            Filtered x
        """

    @abstractmethod
    def apply_matching_filter(self, x, labels, filter, matching_crit_vector, filter_mode="greq", matching_mode="greq"):
        """
            Abstract method.
        """

    @abstractmethod
    def get_unfiltered(self, x, labels):
        """
            Gets either filter class related vector or matrix (for filter metric related values) back
        Args:
            x: Input
            filter: Filter value

        Returns:
            Array (can be 1D or 2D) with the related metric values
        """

    def _get_filter_indx_by_mode(self, x, label, filter, filter_mode):
        """
            Gets the indices of x that fullfills the filter with the corresponding filter_mode
        Args:
            x: Input array
            label: Target array
            filter: Filter threshold
            filter_mode: Specified filter string mode

        Returns:
            Index array of x where x fullfills the filter
        """

        if filter_mode == "lw":
            return np.argwhere(x < filter)
        elif filter_mode == "lweq":
            return np.argwhere(x <= filter)
        elif filter_mode == "gr":
            return np.argwhere(x > filter)
        elif filter_mode == "greq":
            return np.argwhere(x >= filter)
        elif filter_mode == "eq":
            # delta = 0.01
            # return np.argwhere(x <= threshold + delta && x >= threshold - delta)
            return np.argwhere(x == filter)
        else:
            raise("No suitable mode!")


    def _eval_filter_statement(self, x, filter, filter_mode):
        if filter_mode == "greq":
            if x >= filter:
                return True
            else:
                return False
        elif filter_mode == "lw":
            if x < filter:
                return True
            else:
                return False
        elif filter_mode == "lweq":
            if x <= filter:
                return True
            else:
                return False
        elif filter_mode == "gr":
            if x > filter:
                return True
            else:
                return False
        elif filter_mode == "eq":
            if x == filter:
                return True
            else:
                return False

class IOU_Filter(Filter_Base):
    """
        IOU filter
    """
    def __init__(self):
        pass

    def apply_filter(self, x, label, filter, filter_mode="greq"):
        raise(self.__class__.__name__ + "Filter not implemented")

    def apply_matching_filter(self, x, labels, filter, matching_crit_vector, filter_mode="greq", matching_mode="greq"):
        raise("Not implemented yet")

    def get_unfiltered(self, x, labels):
        raise("Not implemented yet")


#reasonable to reduce of filtered values with confidence threshold filter (even small threshold value)
#with some filters matching needed

class Confidence_Filter(Filter_Base):
    """
        Confidence filter
    """
    def __init__(self):
        pass

    def apply_filter(self, x, label, filter, filter_mode="greq"):
        x_value2filter_by = x[:, 3]
        filtered_indx = super()._get_filter_indx_by_mode(x_value2filter_by, label, filter, filter_mode)[:, 0]
        return x[filtered_indx]

    def apply_matching_filter(self, x, labels, filter, matching_crit_vector, filter_mode="greq", matching_mode="greq"):
        raise("Not implemented yet")

    def get_unfiltered(self, x, labels):
        return x[:, 3]


class Distance3D_Filter(Filter_Base):
    """
        Distance filter
    """
    def __init__(self):
        pass

    def apply_filter(self, x, labels, filter, filter_mode="greq"):
        # calculates distance from every x elem to every label elem - if a distance from a specific x to any label fulfills the filter then it will be kept, otherwise filtered

        if labels.shape[0] > data_size_treshold or x.shape[0] > data_size_treshold:
            warnings.warn("Matching Wrapper Array size could be too big.")

        # distance_matrix = self.get_metric(x, labels)

        filtered_indx_list = []

        for i in range(x.shape[0]):
            distance = self.get_metric(np.expand_dims(x[i, :3], axis=0), labels[:, [5, 6, 3]])
            dist_indx = super()._get_filter_indx_by_mode(distance[0, :], labels, filter, filter_mode)[:, 0]
            # filtered_indx_list
            if dist_indx.any():
                filtered_indx_list.append(i)


        # filtered_indx = super()._get_filter_indx_by_mode(distance, filter, mode)

        return x[filtered_indx_list, ...]

    def get_metric(self, x, labels):
        # auf beliebige dimensionen anwendbar - beachte das dimensionsreihenfolge von x und labels für distanz gleich sein muss
        metric_matrix = np.zeros([x.shape[0], labels.shape[0]])
        for i in range(metric_matrix.shape[0]):
            for j in range(labels.shape[0]):
                mse = 0
                for d in range(labels.shape[1]):
                    mse += (x[i, d] - labels[j, d]) ** 2
                    # metric_matrix[i, j] = np.sqrt((x[i, 0] - labels[j, 1]) ** 2 + (x[i, 1] - labels[j, 2]) ** 2 + (x[i, 2] - labels[j, 3]) ** 2)

                metric_matrix[i, j] = np.sqrt(mse)

        return metric_matrix

    def apply_matching_filter(self, x, labels, filter, matching_crit_vector, filter_mode="greq", matching_mode="gr"):
        """
            Applies distance based matching with previous filtering given a matching-criteria-vector
        Args:
            x: Input array
            labels: Target array
            filter: Filter threshold
            matching_crit_vector: Matching criteria array - same shape as input array
            filter_mode: String with filter mode
            matching_mode: String with matching mode

        Returns:
            Metric-Matrix, Filter fullfilling x indices, Matching x indices
        """
        # calculates distance from every x elem to every label elem + filters every x without filter fullfillment  - deprecated idea

        # calculates input-target matrix + applies threshold + matches via criteria + returns metric-matrix, filter-fullfillment-indices, matching-indices

        metric_matrix = self.get_metric(x[:, :3], labels[:, [5, 6, 3]])  # replace with directly filtered matrix for better performance
        if filter_mode == "greq":
            inv_filter_mode = "lw"
        elif filter_mode == "lw":
            inv_filter_mode = "greq"
        elif filter_mode == "lweq":
            inv_filter_mode = "gr"
        elif filter_mode == "gr":
            inv_filter_mode = "lweq"
        elif filter_mode == "eq":
            pass
        else:
            raise("No suitable mode!")

        zero_filter_indx_arr = super()._get_filter_indx_by_mode(metric_matrix, labels, filter, inv_filter_mode)

        for i in range(zero_filter_indx_arr.shape[0]):
            # test = zero_filter_indx_arr[i, :].tolist()
            metric_matrix[zero_filter_indx_arr[i, 0], zero_filter_indx_arr[i, 1]] = 0

        # bin_metric_matrix = metric_matrix.copy()
        # bin_metric_matrix[bin_metric_matrix > 0] = 1


        if matching_mode == "lw":
            crit_sort_indx = np.argsort(matching_crit_vector)
        elif matching_mode == "gr":
            crit_sort_indx = np.argsort(matching_crit_vector)[::-1]
        else:
            raise("No suitable matching mode!")



        metric_matrix_sorted = metric_matrix[crit_sort_indx, ...]

        binary_filter_label_indx = np.zeros(metric_matrix.shape[0]) # at least one filter match for an input instance
        matching_x_indx_sorted = np.full(metric_matrix.shape[1], -1) # -1 as default value for non existent match

        # if multiple matches possible first possible match will be assigned - change to biggest metric value?
        for i in range(metric_matrix.shape[0]):
            x_data_inst_filter_match = False
            # x_matching = False

            for j in range(metric_matrix.shape[1]):
                if metric_matrix_sorted[i, j] != 0:
                    x_data_inst_filter_match = True
                    if matching_x_indx_sorted[j] == -1:
                        # matching_ref_value_arr = metric_matrix[i, j]
                        # matching_x_indx[j] = np.argwhere(crit_sort_indx == i)
                        matching_x_indx_sorted[j] = i

                        break


            if x_data_inst_filter_match:
                # binary_filter_label_indx[i] = 1
                binary_filter_label_indx[crit_sort_indx[i]] = 1
        matching_x_indx = np.full(metric_matrix.shape[1], -1)
        for i in range(matching_x_indx_sorted.shape[0]):

            matching_x_indx[i] = crit_sort_indx[matching_x_indx_sorted[i]]

        return metric_matrix, binary_filter_label_indx, matching_x_indx

    def get_unfiltered(self, x, labels):
        # bin_metric_matrix, metric_matrix = self.get_metric(x, labels)
        metric_matrix = self.get_metric(x, labels)
        return metric_matrix

class Depth_filter(Filter_Base):
    """
        Depth filter
    """
    def __init__(self):
        pass

    def apply_filter(self, x, label, filter, filter_mode="greq"):
        x_value2filter_by = x[:, 2]
        filtered_indx = super()._get_filter_indx_by_mode(x_value2filter_by, label, filter, filter_mode)[:, 0]
        return x[filtered_indx]

    def apply_matching_filter(self, x, labels, filter, filter_mode="greq"):
        raise ("Not implemented yet")

    def get_unfiltered(self, x, labels):
        raise ("Not implemented yet")

class Height_Filter(Filter_Base):
    """
        Height filter
    """
    def __init__(self):
        pass

    def apply_filter(self, x, label, filter, mode="greq"):
        raise(self.__class__.__name__ + "Filter not implemented")

    def apply_matching_filter(self, x, labels, filter, filter_mode="greq"):
        raise ("Not implemented yet")

    def get_unfiltered(self, x, labels):
        raise ("Not implemented yet")

class PositionX_Filter(Filter_Base):
    """
        Position filter in X direction
    """
    def __init__(self):
        pass

    def apply_filter(self, x, labels, filter, filter_mode="greq"):
        x_value2filter_by = x[:, 0]
        filtered_indx = super()._get_filter_indx_by_mode(x_value2filter_by, labels, filter, filter_mode)
        return x[filtered_indx]

    def apply_matching_filter(self, x, labels, filter, filter_mode="greq"):
        raise ("Not implemented yet")

    def get_unfiltered(self, x, labels):
        return x[:, 0]

class PositionY_Filter(Filter_Base):
    """
        Position filter in Y direction
    """
    def __init__(self):
        pass

    def apply_filter(self, x, labels, filter, filter_mode="greq"):
        x_value2filter_by = x[:, 1]
        filtered_indx = super()._get_filter_indx_by_mode(x_value2filter_by, labels, filter, filter_mode)
        return x[filtered_indx]

    def apply_matching_filter(self, x, labels, filter, filter_mode="greq"):
        raise ("Not implemented yet")

    def get_unfiltered(self, x, labels):
        return x[:, 1]

class Class_Filter(Filter_Base):
    """
        Filters to highest confidence Class of class confidence vector
    """
    def __init__(self):
        pass

    def apply_filter(self, x, labels, filter, filter_mode="greq"):
        x_value2filter_by = np.argmax(x[:, 4:])
        filtered_indx = super()._get_filter_indx_by_mode(x_value2filter_by, labels, filter, filter_mode)
        return x[filtered_indx]

    def apply_matching_filter(self, x, labels, filter, filter_mode="greq"):
        raise ("Not implemented yet")

    def get_unfiltered(self, x, labels):
        warnings.warn("Filter " + type(self).__name__ + " might not make sense here")
        return x[:, 4:]
###########################

def set_filter_wrapper(metric_filter_config):
    """
        Helper function to set FilterWrapper
    Args:
        metric_filter_config: Dictionary containing filter settings

    Returns:
        List containing FilterWrapper
    """
    ## check for unneces
    metric_filter_list = []
    for key in metric_filter_config:
        filter_type = metric_filter_config[key].pop("metric_filter_type")
        if filter_type == "threshold":
            metric_filter_list.append(Threshold_Filter_Wrapper(**metric_filter_config[key]))
        elif filter_type == "range_selection":
            metric_filter_list.append(Range_Selection_Filter_Wrapper(**metric_filter_config[key]))
        elif filter_type == "matching":
            metric_filter_config[key]["matching_crit"] = Identity_Filter_Wrapper(metric_filter_config[key].pop("matching_crit"))
            metric_filter_list.append(Matching_Filter_Wrapper(**metric_filter_config[key]))
        else:
            raise("Filter type " + filter_type + " doesnt exist!")
    return metric_filter_list

def set_matching_filter_wrapper_list(matching_filter_config):
    """
        Helper function to set MatchingFilter from dictionary config
    Args:
        matching_filter_config: Dictionary containing filter settings

    Returns:
        List containing MatchingFilterWrappers
    """
    matching_filter_list = []
    for key in matching_filter_config:
        matching_filter_config[key]["matching_crit"] = Identity_Filter_Wrapper(matching_filter_config[key].pop("matching_crit"))
        matching_filter_list.append(Matching_Filter_Wrapper(**matching_filter_config[key]))
    return matching_filter_list

class Matcher():
    def __init__(self, matching_filter_list, matching_crit, matching_mode="gr", *args, **kwargs):
        self.matching_filter_list = matching_filter_list
        self.matching_crit = matching_crit
        self.matching_mode = matching_mode

    def match2confusionMatrix(self, filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels):

        filtered_metric_matrix_list, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final = self.apply_filter(filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels)

        matching_crit_vector = self.matching_crit.apply_filter_wrapper(x, labels)

        if self.matching_mode == "lw":
            crit_sort_indx = np.argsort(matching_crit_vector)
        elif self.matching_mode == "gr":
            crit_sort_indx = np.argsort(matching_crit_vector)[::-1]
        else:
            raise ("No suitable matching mode!")

        class_no = x[0, 4:].shape[0]  ## hardcoded - changes if standard input definition changes  #WIP

        confusion_matrix_list = []
        #  confusion matrix with rows -> groundtruth, columns -> predictions
        #  last element in row/column is for background


        # if multiple matches possible, first possible match will be assigned - change to biggest metric value (due to matching filter multiplication value doesnt have related meaning anymore)?
        for metric_matrix in filtered_metric_matrix_list:
            metric_matrix_sorted = metric_matrix[crit_sort_indx, ...]

            confusion_matrix = np.zeros([class_no + 1, class_no + 1])

            label_indx_list = range(metric_matrix_sorted.shape[1])
            for i in range(metric_matrix.shape[0]):
                x_data_inst_filter_match = False
                predicted_class = int(np.argmax(x[crit_sort_indx[i], 4:]))
                for j in label_indx_list:
                    if metric_matrix_sorted[i, j] != 0:
                        if predicted_class == labels[j, 1]:
                            label_indx_list.remove(j)
                            x_data_inst_filter_match = True
                            break

                if x_data_inst_filter_match:
                    confusion_matrix[predicted_class, predicted_class] += 1
                else:
                    confusion_matrix[class_no + 1, predicted_class] += 1

            if label_indx_list:
                for indx in label_indx_list:
                    label_class = labels[indx, 1]
                    confusion_matrix[label_class, class_no + 1] += 1

            confusion_matrix_list.append(confusion_matrix)

        return confusion_matrix_list, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final

    def match2tp_fp_crit_lists(self, filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels):
        filtered_metric_matrix_list, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final, related_filter_x_list = self.apply_filter(filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels)


        tp_fp_case_list = []  # list containing [tp, fp, matching_crit_value] as elements

        # if multiple matches possible first possible match will be assigned - change to biggest metric value? - e.g. more close but wrong class wont get associated but more distant one could still
        for l in range(len(filtered_metric_matrix_list)):
            metric_matrix = filtered_metric_matrix_list[l]

            matching_crit_vector = self.matching_crit.apply_filter_wrapper(related_filter_x_list[l], labels)

            if self.matching_mode == "lw":
                crit_sort_indx = np.argsort(matching_crit_vector)
            elif self.matching_mode == "gr":
                crit_sort_indx = np.argsort(matching_crit_vector)[::-1]
            else:
                raise ("No suitable matching mode!")


            metric_matrix_sorted = metric_matrix[crit_sort_indx, ...]

            tp_fp_case_list_local = [[] for foo in range(related_filter_x_list[l][0, 4:].shape[0])]

            label_indx_list = [*range(metric_matrix_sorted.shape[1])]
            for i in range(metric_matrix.shape[0]):
                x_data_inst_filter_match = False
                predicted_class = np.argmax(related_filter_x_list[l][crit_sort_indx[i], 4:])
                for j in label_indx_list:
                    if metric_matrix_sorted[i, j] != 0:
                        if predicted_class == labels[j, 1]:   # hardcoded for class equality - change more general configuration?   ##  WIP  - if right class detected
                            label_indx_list.remove(j)
                            x_data_inst_filter_match = True
                            break

                if x_data_inst_filter_match:
                    tp_fp_case_list_local[predicted_class].append([1, 0, matching_crit_vector[crit_sort_indx[i]]])
                else:
                    tp_fp_case_list_local[predicted_class].append([0, 1, matching_crit_vector[crit_sort_indx[i]]])

            tp_fp_case_list.append(tp_fp_case_list_local)

        return tp_fp_case_list, filter_name_list_final, filter_wrapper_name_list_final, filter_value_list_final

    def apply_filter(self, x, labels):
        filter_x_list = [x]
        filter_name_list = []
        filter_wrapper_name_list = []
        filter_value_list = []

        return self.apply_filter(filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels)

    def apply_filter(self, filter_x_list, filter_name_list, filter_wrapper_name_list, filter_value_list, x, labels):

        filtered_metric_matrix_list_comb = []
        filter_name_list_comb = []
        filter_wrapper_name_list_comb = []
        filter_value_list_comb = []

        for metric_filter in self.matching_filter_list:
            filtered_metric_matrix_list_comb_tmp = []
            filter_name_list_comb_tmp = []
            filter_wrapper_name_list_comb_tmp = []
            filter_value_list_comb_tmp = []

            filter_name_tmp_list = []
            filter_wrapper_name_tmp_list = []
            filter_value_tmp_list = []

            filtered_metric_matrix_list_tmp = []

            filter_x_list_len = len(filter_x_list)

            for elem in filter_x_list:
                filtered_metric_matrix_list, binary_filter_label_indx_list, matching_x_indx_list = metric_filter.apply_filter_and_match(elem, labels)
                filtered_metric_matrix_list_tmp.extend(filtered_metric_matrix_list)
                # filter_x_tmp_list.extend(metric_filter.apply_filter_wrapper(elem, labels))


            # filter_value_list_len = metric_filter.get_filter_list_len()
            if filter_value_list:
                new_filter_metric_name = metric_filter.get_filter_name()
                # new_filter_wrapper_name = metric_filter.get_wrapper_name()
                new_filter_wrapper_name = "matcher"
                # for i in range(filter_value_list_len):
                for i in range(len(filter_value_list)):
                    for new_filter_value in metric_filter.get_filter_value_name_list():
                        new_filter_value_sublist = filter_value_list[i].copy()
                        new_filter_value_sublist.append(new_filter_value)
                        filter_value_tmp_list.append(new_filter_value_sublist)

                        new_filter_name_sublist = filter_name_list[i].copy()
                        new_filter_name_sublist.append(new_filter_metric_name)
                        filter_name_tmp_list.append(new_filter_name_sublist)

                        new_wrapper_name_sublist = filter_wrapper_name_list[i].copy()
                        new_wrapper_name_sublist.append(new_filter_wrapper_name)
                        filter_wrapper_name_tmp_list.append(new_wrapper_name_sublist)
            else:
                filter_value_tmp_list = [[elem] for elem in metric_filter.get_filter_value_name_list()]
                # for i in range(filter_value_list_len):
                for i in range(len(filter_value_tmp_list)):
                    filter_name_tmp_list.append([metric_filter.get_filter_name()])
                    filter_wrapper_name_tmp_list.append([metric_filter.get_wrapper_name()])

            if filtered_metric_matrix_list_comb:

                new_metric_matrix_list_norm_len = int(len(filtered_metric_matrix_list_tmp) / filter_x_list_len)
                old_metric_matrix_list_norm_len = int(len(filtered_metric_matrix_list_comb) / filter_x_list_len)

                for x_indx in range(filter_x_list_len):
                    for i in range(old_metric_matrix_list_norm_len):
                        for j in range(new_metric_matrix_list_norm_len):
                            filtered_metric_matrix_list_comb_tmp.append(np.multiply(filtered_metric_matrix_list_comb[x_indx * old_metric_matrix_list_norm_len + i], filtered_metric_matrix_list_tmp[x_indx * new_metric_matrix_list_norm_len + j]))

                            new_filter_name_list = filter_name_list_comb[x_indx * old_metric_matrix_list_norm_len + i].copy()
                            new_filter_name_list.append(filter_name_tmp_list[j][-1])
                            # filter_name_list_comb_tmp.append(filter_name_list_comb[x_indx * old_metric_matrix_list_norm_len + i])
                            filter_name_list_comb_tmp.append(new_filter_name_list)

                            new_wrapper_name_list = filter_wrapper_name_list_comb[x_indx * old_metric_matrix_list_norm_len + i].copy()
                            new_wrapper_name_list.append(filter_wrapper_name_tmp_list[j][-1])

                            # filter_wrapper_name_list_comb_tmp.append(filter_wrapper_name_list_comb[x_indx * old_metric_matrix_list_norm_len + i])
                            filter_wrapper_name_list_comb_tmp.append(new_wrapper_name_list)

                            new_filter_value_list = filter_value_list_comb[x_indx * old_metric_matrix_list_norm_len + i].copy()
                            new_filter_value_list.append(filter_value_tmp_list[j][-1])
                            # filter_value_list_comb_tmp.append(filter_value_list_comb[x_indx * old_metric_matrix_list_norm_len + i])
                            filter_value_list_comb_tmp.append(new_filter_value_list)

            else:
                filtered_metric_matrix_list_comb_tmp.extend(filtered_metric_matrix_list_tmp)
                filter_name_list_comb_tmp.extend(filter_name_tmp_list)
                filter_wrapper_name_list_comb_tmp.extend(filter_wrapper_name_tmp_list)
                filter_value_list_comb_tmp.extend(filter_value_tmp_list)

            filtered_metric_matrix_list_comb = filtered_metric_matrix_list_comb_tmp
            filter_name_list_comb = filter_name_list_comb_tmp
            filter_wrapper_name_list_comb = filter_wrapper_name_list_comb_tmp
            filter_value_list_comb = filter_value_list_comb_tmp

        related_filter_x_list = []
        filter_value_list_len = len(filter_value_list)
        for i in range(filter_value_list_len):
            for j in range(int(len(filter_value_list_comb) / filter_value_list_len)):
                related_filter_x_list.append(filter_x_list[i])

        return filtered_metric_matrix_list_comb, filter_name_list_comb, filter_wrapper_name_list_comb, filter_value_list_comb, related_filter_x_list


    # def _recursive_metric_matrix_filter_combination(self, filtered_metric_matrix_list_comb, filter_name_list_comb, filter_wrapper_name_list_comb, filter_value_list_comb):
    #
    #     filter_metric_matrix_list_final = []
    #     filter_name_list_final = []
    #     filter_wrapper_name_list_final = []
    #     filter_value_list_final = []
    #
    #     # for metric_matrix in filtered_metric_matrix_list_comb:
    #     #     metric_matrix_list, filter_name_list, filter_wrapper_name_list, filter_value_list = self._rec_combine_metric_matrix_filter(metric_matrix, filtered_metric_matrix_list_comb, filter_name_list_comb, filter_wrapper_name_list_comb, filter_value_list_comb)
    #     #     filter_metric_matrix_list_final.append(metric_matrix_list)
    #     #     filter_name_list_final.append(filter_name_list)
    #     #     filter_wrapper_name_list_final.append(filter_wrapper_name_list)
    #     #     filter_value_list_final.append(filter_value_list)
    #     filter_metric_matrix_list_final = filtered_metric_matrix_list_comb.pop(0)      # assign first elem to final_metric_matrix_list to start recursion
    #     final_name_list = filter_name_list_comb.pop(0)
    #     final_wrapper_name_list = filter_wrapper_name_list_comb.pop(0)
    #     final_filter_value_list = filter_value_list_comb.pop(0)
    #
    #     filter_metric_matrix_list_final, to_combine_metrix_matrix_list, filter_name_list, filter_wrapper_name_list, filter_value_list = self._rec_combine_metric_matrix_filter(filter_metric_matrix_list_final, final_name_list, final_wrapper_name_list, final_filter_value_list, filtered_metric_matrix_list_comb, filter_name_list_comb, filter_wrapper_name_list_comb, filter_value_list_comb)
    #     return filter_metric_matrix_list_final, filter_name_list, filter_wrapper_name_list, filter_value_list


    # def _rec_combine_metric_matrix_filter(self, final_metric_matrix_list, final_name_list, final_wrapper_name_list, final_filter_value_list, to_combine_metric_matrix_list, to_combine_filter_name_list_comb, to_combine_filter_wrapper_name_list_comb, to_combine_filter_value_list_comb):
    #
    #     if to_combine_metric_matrix_list:
    #         to_combine_metric_matrix_list_tmp = to_combine_metric_matrix_list.pop(0)
    #         final_metrix_matrix_list_tmp = []
    #         # for to_combine_metric_matrix in to_combine_metric_matrix_list_tmp:
    #         #     for final_metric_matrix in final_metric_matrix_list:
    #         #         final_metrix_matrix_list_tmp.append(np.multiply(final_metric_matrix, to_combine_metric_matrix))
    #         for i in range(len(to_combine_metric_matrix_list_tmp)):
    #             for j in range(len(final_metric_matrix_list)):
    #                 final_metrix_matrix_list_tmp.append(np.multiply(final_metric_matrix_list[j], to_combine_metric_matrix_list_tmp[i]))
    #
    #
    #                 final_name_list[j].append(to_combine_filter_name_list_comb[i][-1])
    #                 final_wrapper_name_list[j].append("matcher")
    #                 final_filter_value_list[j].append(to_combine_filter_value_list_comb[i][-1])
    #
    #         return self._rec_combine_metric_matrix_filter(final_metrix_matrix_list_tmp, final_name_list, final_wrapper_name_list, final_filter_value_list, to_combine_metric_matrix_list, to_combine_filter_name_list_comb, to_combine_filter_wrapper_name_list_comb, to_combine_filter_value_list_comb)
    #
    #     else:
    #         return final_metric_matrix_list, to_combine_metric_matrix_list, filter_name_list_comb, filter_wrapper_name_list_comb, filter_value_list_comb


class Metrics_Wrapper():
    """
        Wrapper class for metrics calculation/handling
    """
    def __init__(self, metric_config):
        """
            Constructor setting/configuring metric
        Args:
            metric_config: Dictionary containing metric settings
        """
        self.metric_type = metric_config.pop("metric_type")
        self.metric_config = metric_config.copy()
        # if "num_thresholds" in metric_config.keys():
        #     metric_config.pop("num_thresholds")
        #
        if "metric_filter" in metric_config.keys():
            self.metric_filter_list = self._set_metric_filter_list(metric_config["metric_filter"])
        else:
            self.metric_filter_list = []
        # if "matching_filter" in metric_config.keys():
        #     self.metric_matching_filter_list = self._set_matching_filter_list(metric_config.pop("matching_filter"))
        # else:
        #     self.metric_matching_filter_list = []
        # metric_config["matching_filter_list"] = self.metric_matching_filter_list
        # self.metric = self.set_metric(self.metric_type, self.metric_filter_list, self.metric_matching_filter_list, metric_config)

        if "matcher" in metric_config.keys():
            matcher_config = metric_config.pop("matcher")
            metric_matching_filter_list = self._set_matching_filter_list(matcher_config.pop("matching_filter"))
            global_matching_krit = Identity_Filter_Wrapper(matcher_config.pop("matching_crit"))
            global_matching_mode = matcher_config.pop("matching_mode")
            self.matcher = Matcher(metric_matching_filter_list, global_matching_krit, global_matching_mode, **matcher_config)
        else:
            self.matcher = None


        self.metric = self.set_metric(self.metric_type, self.metric_filter_list, self.matcher, metric_config)

    # def set_metric(self, metric_type, metric_filter_list, matching_filter_list, metric_config):
    def set_metric(self, metric_type, metric_filter_list, matcher, metric_config):
        """
            Sets metrics by name
        Args:
            metric_type: Metric name
            metric_config: Dictionary containing metric settings

        Returns:

        """
        if metric_type == "confusion_matrix":
            return ConfusionMatrix(metric_filter_list, matcher, **metric_config)
        if metric_type == "class_pr":
            return Class_PR_Calc(metric_filter_list, matcher, **metric_config)
        elif metric_type == "accuracy":
            return Accuracy(metric_filter_list, matcher, **metric_config)
        elif metric_type == "precision":
            return Precision(metric_filter_list, matcher, **metric_config)
        elif metric_type == "recall":
            return Recall(metric_filter_list, matcher, **metric_config)
        elif metric_type == "f1_score":
            return F1_Score(metric_filter_list, matcher, **metric_config)
        elif metric_type == "f_beta_score":
            return F_Beta_Score(metric_filter_list, matcher, **metric_config)
        elif metric_type == "false_pos_rate":
            return False_Pos_Rate(metric_filter_list, matcher, **metric_config)
        elif metric_type == "false_neg_rate":
            return False_Neg_Rate(metric_filter_list, matcher, **metric_config)
        elif metric_type == "true_neg_rate":
            return True_Neg_Rate(metric_filter_list, matcher, **metric_config)
        elif metric_type == "false_discovery_rate":
            return False_Discovery_Rate(metric_filter_list, matcher, **metric_config)
        elif metric_type == "false_omission_rate":
            return False_Omission_Rate(metric_filter_list, matcher, **metric_config)
        elif metric_type == "neg_predictive_value":
            return Negative_Predictive_Value(metric_filter_list, matcher, **metric_config)
        elif metric_type == "AverageRecall":
            return AverageRecall(metric_filter_list, matcher, **metric_config)
        elif metric_type == "meanAverageRecall":
            return meanAverageRecall(metric_filter_list, matcher, **metric_config)
        elif metric_type == "AP":
            return AveragePrecision(metric_filter_list, matcher, **metric_config)
        elif metric_type == "mAP":
            return meanAveragePrecision(metric_filter_list, matcher, **metric_config)

    def calc(self, x, label):
        self.metric(x, label)

    def _set_metric_filter_list(self, metric_filter_config):
        return set_filter_wrapper(metric_filter_config)

    def _set_matching_filter_list(self, matching_filter_config):
        return set_matching_filter_wrapper_list(matching_filter_config)

    def get_base_metrics_list(self):
        return self.metric.get_base_metrics_list()


class MetricsHandler():
    """
        MetricsHandler class to add, configure and calculate metrics with.
    """
    def __init__(self, metrics_dict):
        """
            Constructor
        Args:
            metrics_dict: Dictionary containing all metrics with name (string) and corresponding metric - wrapped in a MetricsWrapper object
        """
        self.metrics_dict = metrics_dict
        self.metric_proc_list = []
        for metric in self.metrics_dict:
            # self.add_metric2handler(self.metrics_dict[metric])
            self.add_metric2handler(self.metrics_dict[metric].metric)

    def add_metric2handler(self, metric):
        """
            Adds a metric (and related base metrics) to the corresponding MetricsHandler processing list
        Args:
            metric: Metric - Derived Class of Metric_Base
        """
        # add real comparison - not memory reference but actual class instance params etc.
        base_metrics_list = metric.get_base_metrics_list() # every metrics class got an own base metric list - metrics that need to be calculated beforehand

        if base_metrics_list:
            for base_metric in base_metrics_list:
                if not self._metric_in_metric_proc_list(base_metric):
                    if base_metric.get_base_metrics_list():
                        for base_tmp_metric in base_metric.get_base_metrics_list():
                            self.add_metric2handler(base_tmp_metric)
                    else:
                        self.metric_proc_list.insert(0, base_metric)
                    if not self._metric_in_metric_proc_list(base_metric): # need to check again due to recursion
                        self.metric_proc_list.append(base_metric)
                    else:
                        base_metric_tmp = self._get_eq_elem_from_proc_list(base_metric)
                        base_metrics_list.remove(base_metric)
                        base_metrics_list.append(base_metric_tmp)
                else:
                    # change entry in base_metrics_list to globally unique object from processing list (proc_list)
                    base_metric_tmp = self._get_eq_elem_from_proc_list(base_metric)
                    base_metrics_list.remove(base_metric)
                    base_metrics_list.append(base_metric_tmp)

        # if metric.metric not in self.metric_proc_list:
        if metric not in self.metric_proc_list:
            self.metric_proc_list.append(metric)

    def add_metric_list2handler(self, metric_list):
        """
            Adds List of metrics to the corresponding MetricsHandler processing list
        Args:
            metric_list:

        Returns:

        """
        for metric in metric_list:
            self.add_metric2handler(metric)

    def process_batch(self, x, labels):
        """
            Processes a batch for all metrics in processing list
        Args:
            x: Input array
            labels: Target array
        """
        for metric in self.metric_proc_list:
            for batch_indx in range(x.shape[0]):
                metric(x[batch_indx], labels[labels[:, 0] == batch_indx])

    def process_end_of_batch(self):
        """
            Calculates final metrics after processed all batches
        Returns:

        """
        for metric in self.metric_proc_list:
            metric.process_end_of_batch()

    def _metric_in_metric_proc_list(self, metric):
        """
            Checks if specified metric in processing list
        Args:
            metric: Specified metric to check

        Returns:
            boolean type
        """
        for proc_metric in self.metric_proc_list:
            if proc_metric == metric:
                return True

        return False

    def _get_eq_elem_from_proc_list(self, metric):
        """
            Gets the metric from the processing list that is equal to the specified metric
        Args:
            metric: Specified metric

        Returns:
            Equal metric from processing list or None if no equal object available
        """
        for proc_metric in self.metric_proc_list:
            if proc_metric == metric:
                return proc_metric
        return None

    def get_metrics_dict(self):
        return self.metrics_dict


##################################



def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

