import torch
import torch.nn as nn
import torch.nn.modules as nn_modules
import torch.nn.functional as F
import numpy as np

class Loss_Wrapper():
    def __init__(self, loss_config):
        self.loss_type = list(loss_config.keys())[0] # if multiple keys then only first one describes loss type
        self.loss_config = loss_config[self.loss_type]
        self.loss = self.parse_loss_config(self.loss_type, self.loss_config)

    def parse_loss_config(self, loss_type, loss_config):
        if loss_type == "weighted_sum":
            return Weighted_sum(**loss_config)
        elif loss_type == "average_sum":
            return Average_sum(**loss_config)
        elif loss_type == "info_nce":
            return InfoNCE(**loss_config)
        elif loss_type == "discriminative_contrast": # from the related paper
            return Discriminative_contrast_loss(**loss_config)
        elif loss_type == "spherical_contrast_panoptic":
            return Panoptic_spherical_contrastive_loss(**loss_config)

        ## deprecated
        elif loss_type == "mse":
            return nn_modules.MSELoss(**loss_config)
        elif loss_type == "l1":
            return nn_modules.L1Loss(**loss_config)
        elif loss_type == "cross_entropy":
            return nn_modules.CrossEntropyLoss(**loss_config)
        elif loss_type == "bin_cross_entropy":
            return nn_modules.BCELoss(**loss_config)
        elif loss_type == "bin_cross_entropy_logits":
            return nn_modules.BCEWithLogitsLoss(**loss_config)
        elif loss_type == "kullback_leibler":
            return nn_modules.KLDivLoss(**loss_config)
        elif loss_type == "poisson_nnll":
            return nn_modules.PoissonNLLLoss(**loss_config)
        elif loss_type == "nll":
            return nn_modules.NLLLoss2d(**loss_config)
        elif loss_type == "smooth_l1":
            return nn_modules.SmoothL1Loss(**loss_config)
        elif loss_type == "cosine_embedding":
            return nn_modules.CosineEmbeddingLoss(**loss_config)
        elif loss_type == "bin_dice":
            return BinaryDiceLoss(**loss_config)
        elif loss_type == "dice":
            return DiceLoss(**loss_config)
        elif loss_type == "threedbbox_mse":
            return BBox3d_custom(**loss_config)


class Metrics_Wrapper():
    def __init__(self, metric_config):
        self.metric_type = metric_config.pop("metric_type")
        self.metric_config = metric_config.copy()
        if "num_thresholds" in metric_config.keys():
            metric_config.pop("num_thresholds")
        self.metric = self.set_metric(self.metric_type, metric_config)

    def set_metric(self, metric_type, metric_config):
        if metric_type == "bin_dice":
            return BinaryDiceLoss(**metric_config)
        elif metric_type == "dice":
            return DiceLoss(**metric_config)
        elif metric_type == "iou":
            return IoU(**metric_config)
        elif metric_type == "accuracy":
            return Accuracy(**metric_config)
        elif metric_type == "precision":
            return Precision(**metric_config)
        elif metric_type == "recall":
            return Recall(**metric_config)
        elif metric_type == "f1_score":
            return F1_Score(**metric_config)
        elif metric_type == "false_pos_rate":
            return False_Pos_Rate(**metric_config)
        elif metric_type == "class_cases":
            return Classification_cases(**metric_config)
        elif metric_type == "mse_relcamdepth":
            return MSE_relCamDepth(**metric_config)
        elif metric_type == "mse_relyaw":
            return MSE_relYaw(**metric_config)
        elif metric_type == "mse_objpt":
            return MSE_objPT(**metric_config)
        elif metric_type == "mse_objpt_x":
            return MSE_objPT_X(**metric_config)
        elif metric_type == "mse_objpt_y":
            return MSE_objPT_Y(**metric_config)
        elif metric_type == "mse_bboxshape":
            return MSE_BBoxShape(**metric_config)
        elif metric_type == "mse_bboxshape_1":
            return MSE_BBoxShape_1(**metric_config)
        elif metric_type == "mse_bboxshape_2":
            return MSE_BBoxShape_2(**metric_config)
        elif metric_type == "mse_bboxshape_3":
            return MSE_BBoxShape_3(**metric_config)
        elif metric_type == "bbox3d_iou":
            return BBox3D_IOU(**metric_config)
        elif metric_type == "bbox3d_bev_iou":
            return BBox3D_BEV_IOU(**metric_config)
        # elif metric_type == "confusion_matrix":
        #     raise NotImplementedError

class InfoNCE(nn.Module):
    def __init__(self, temperature):
        self.temperature = temperature

    def forward(self, outputs_dict, labels_dict):
        raise ValueError("Discriminative_contrast_loss not implemented yet!")

class Discriminative_contrast_loss(nn.Module):
    def __init__(self, margin_variance, margin_distance, weighting_list):
        self.margin_variance = margin_variance
        self.margin_distance = margin_distance
        self.weighting_list = weighting_list

    def forward(self, outputs_dict, labels_dict):
        raise ValueError("Discriminative_contrast_loss not implemented yet!")

class Panoptic_spherical_contrastive_loss(nn.Module):
    def __init__(self, cat_id_radius_order_map_list, radius_diff_dist=1, radius_start_val=0, radius_loss_weight=0.5, similarity_loss_weight=0.5, hypsph_radius_map_list=None):

        self.cat_id_radius_order_map_list = cat_id_radius_order_map_list
        self.radius_diff_dist = radius_diff_dist
        self.radius_start_val = radius_start_val

        self.radius_loss_weight = radius_loss_weight
        self.similarity_loss_weight = similarity_loss_weight

        if hypsph_radius_map_list:
            self.hypsph_radius_map_list = hypsph_radius_map_list
        else:
            self.hypsph_radius_map_list = list(range(self.radius_start_val, self.radius_start_val + self.radius_diff_dist * len(self.cat_id_radius_order_map_list), self.radius_diff_dist))

    def forward(self, outputs_dict, labels_dict):


        embeddings = outputs_dict["embeddings"]



class Weighted_sum(nn.Module):
    def __init__(self, loss_list, weights_list):
        self.loss_list = []
        self.weights_list = weights_list

        for loss in loss_list:
            self.loss_list.append(Loss_Wrapper(loss))


    def forward(self, outputs_dict, labels_dict):
        total_loss = 0

        for i in range(len(self.loss_list)):
            total_loss += self.weights_list * self.loss_list[i](outputs_dict, labels_dict)

        return total_loss

class Average_sum(nn.Module):
    def __init__(self, loss_list):
        self.loss_list = []

        for loss in loss_list:
            self.loss_list.append(Loss_Wrapper(loss))


    def forward(self, outputs_dict, labels_dict):
        total_loss = 0

        for i in range(len(self.loss_list)):
            total_loss += self.loss_list[i](outputs_dict, labels_dict)

        return total_loss / len(self.loss_list)

## from here on deprecated - maybe still usefull
class BBox3d_custom(nn.Module):
    def __init__(self, focal_length, img_size, weight_list):
        super(BBox3d_custom, self).__init__()
        self.focal_length = focal_length
        self.image_size = img_size
        self.weight_list = weight_list

    def forward(self, outputs, labels, *args, **kwargs):
        return self.weight_list[0] * torch.mean((outputs[:, 0] - labels[:, 1] / self.focal_length) ** 2) + \
                       self.weight_list[1] * torch.mean(((outputs[:, 1] - 0.5) * 180 - labels[:, 2]) ** 2) + \
                       self.weight_list[2] * torch.mean((outputs[:, 2] * self.image_size - labels[:, 3]) ** 2) + \
                       self.weight_list[3] * torch.mean((outputs[:, 3] * self.image_size - labels[:, 4]) ** 2) + \
                       self.weight_list[4] * torch.mean(
                (outputs[:, 4] * outputs[:, 0] * self.image_size - labels[:, 5]) ** 2) + \
                       self.weight_list[5] * torch.mean(
                (outputs[:, 5] * outputs[:, 0] * self.image_size - labels[:, 6]) ** 2) + \
                       self.weight_list[6] * torch.mean(
                (outputs[:, 6] * outputs[:, 0] * self.image_size - labels[:, 7]) ** 2)

class MSE_relCamDepth(nn.Module):
    def __init__(self, focal_length):
        super(MSE_relCamDepth, self).__init__()
        self.focal_length = focal_length

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 0] - labels[:, 1] / self.focal_length) ** 2)

class MSE_relYaw(nn.Module):
    def __init__(self):
        super(MSE_relYaw, self).__init__()

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean(((outputs[:, 1] - 0.5) * 180 - labels[:, 2]) ** 2)


class MSE_objPT(nn.Module):
    def __init__(self, image_size):
        super(MSE_objPT, self).__init__()
        self.image_size = image_size

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 2] * self.image_size - labels[:, 3]) ** 2) \
            + torch.mean((outputs[:, 3] * self.image_size - labels[:, 4]) ** 2)

class MSE_objPT_X(nn.Module):
    def __init__(self, image_size):
        super(MSE_objPT_X, self).__init__()
        self.image_size = image_size

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 2] * self.image_size - labels[:, 3]) ** 2)

class MSE_objPT_Y(nn.Module):
    def __init__(self, image_size):
        super(MSE_objPT_Y, self).__init__()
        self.image_size = image_size

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 3] * self.image_size - labels[:, 4]) ** 2)

class MSE_BBoxShape(nn.Module):
    def __init__(self, image_size):
        super(MSE_BBoxShape, self).__init__()
        self.image_size = image_size

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 4] * outputs[:, 0] * self.image_size - labels[:, 5]) ** 2) \
            + torch.mean((outputs[:, 5] * outputs[:, 0] * self.image_size - labels[:, 6]) ** 2) \
            + torch.mean((outputs[:, 6] * outputs[:, 0] * self.image_size - labels[:, 7]) ** 2)

class MSE_BBoxShape_1(nn.Module):
    def __init__(self, image_size):
        super(MSE_BBoxShape_1, self).__init__()
        self.image_size = image_size

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 4] * outputs[:, 0] * self.image_size - labels[:, 5]) ** 2)

class MSE_BBoxShape_2(nn.Module):
    def __init__(self, image_size):
        super(MSE_BBoxShape_2, self).__init__()
        self.image_size = image_size

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 5] * outputs[:, 0] * self.image_size - labels[:, 6]) ** 2)

class MSE_BBoxShape_3(nn.Module):
    def __init__(self, image_size):
        super(MSE_BBoxShape_3, self).__init__()
        self.image_size = image_size

    def forward(self, outputs, labels, *args, **kwargs):
        return torch.mean((outputs[:, 6] * outputs[:, 0] * self.image_size - labels[:, 7]) ** 2)

class BBox3D_IOU(nn.Module):
    def __init__(self, focal_length):
        super(BBox3D_IOU, self).__init__()
        self.focal_length = focal_length

    def forward(self, outputs, labels, *args, **kwargs):

        def get_ct_pt_in_world(pred_data, crop_pos, cam_config, cam_extrinsic, input_image_size):
            crop_resize_factor = []

            crop_resize_factor.append((crop_pos[1] - crop_pos[0]) / cam_config["width"])
            crop_resize_factor.append((crop_pos[3] - crop_pos[2]) / cam_config["height"])

            proj_3dmean_pt = np.array([pred_data[2] * crop_resize_factor[0], pred_data[3] * crop_resize_factor[1]])

            proj_3d_mean_pt_full_img = proj_3dmean_pt
            proj_3d_mean_pt_full_img[0] += crop_pos[0]
            proj_3d_mean_pt_full_img[1] += crop_pos[2]

            reldepth = pred_data[0]
            relyaw = pred_data[1]  # negative due to definition
            bbox_shape = np.array([pred_data[4], pred_data[5], pred_data[6]])  # in blender coordsystem (z looking up)

            cam_intrinsics = cam_config["camera_intrinsic"]
            # cam_intrinsics[0, 2] = cam_config["width"] / 2
            # cam_intrinsics[1, 2] = cam_config["height"] / 2

            bbox3d_mean_pt_camcoord = np.dot(np.linalg.inv(cam_intrinsics),
                                             np.array([proj_3d_mean_pt_full_img[0], proj_3d_mean_pt_full_img[1], 1]))

            bbox3d_mean_pt_camcoord *= reldepth

            bbox3d_mean_pt_camcoord = np.dot(cam_extrinsic, np.append(bbox3d_mean_pt_camcoord, 1))[:3]
            return bbox3d_mean_pt_camcoord

        bbox3d_iou_sum = 0
        labels_tmp = labels.detach().cpu().numpy()[:, 1:]
        outputs_tmp = kwargs["final_outputs"].detach().cpu().numpy()
        crop_pos_list = kwargs["crop_pos_list"]
        cam_config = kwargs["cam_config"]
        cam_extr_list = kwargs["cam_extr_list"]
        input_img_size = kwargs["input_img_size"]

        for i in range(outputs.shape[0]):
            pred_mean_pt_world = get_ct_pt_in_world(outputs_tmp[i], crop_pos_list[i], cam_config, cam_extr_list[i], input_img_size)
            pred_relyaw_rad = outputs_tmp[i, 1] * np.pi / 180
            pred_bbox_shape = (outputs_tmp[i, 4], outputs_tmp[i, 5], outputs_tmp[i, 6])

            pred_box_dict = {"mean_pt_world": pred_mean_pt_world,
                             "relyaw_rad": pred_relyaw_rad,
                             "bbox_shape": pred_bbox_shape}

            labels_mean_pt_world = get_ct_pt_in_world(labels_tmp[i], crop_pos_list[i], cam_config, cam_extr_list[i], input_img_size)
            labels_relyaw_rad = labels_tmp[i, 1] * np.pi / 180
            labels_bbox_shape = (labels_tmp[i, 4], labels_tmp[i, 5], labels_tmp[i, 6])

            labels_box_dict = {"mean_pt_world": labels_mean_pt_world,
                             "relyaw_rad": labels_relyaw_rad,
                             "bbox_shape": labels_bbox_shape}

            bbox3d_iou, bev_iou = calc_bbox3d_iou(pred_box_dict, labels_box_dict)
            bbox3d_iou_sum += bbox3d_iou

        return bbox3d_iou_sum / outputs.shape[0]

class BBox3D_BEV_IOU(nn.Module):
    def __init__(self, focal_length):
        super(BBox3D_BEV_IOU, self).__init__()
        self.focal_length = focal_length

    def forward(self, outputs, labels, *args, **kwargs):

        def get_ct_pt_in_world(pred_data, crop_pos, cam_config, cam_extrinsic, input_image_size):
            crop_resize_factor = []

            crop_resize_factor.append((crop_pos[1] - crop_pos[0]) / cam_config["width"])
            crop_resize_factor.append((crop_pos[3] - crop_pos[2]) / cam_config["height"])

            proj_3dmean_pt = np.array([pred_data[2] * crop_resize_factor[0], pred_data[3] * crop_resize_factor[1]])

            proj_3d_mean_pt_full_img = proj_3dmean_pt
            proj_3d_mean_pt_full_img[0] += crop_pos[0]
            proj_3d_mean_pt_full_img[1] += crop_pos[2]

            reldepth = pred_data[0]
            # relyaw = pred_data[1]  # negative due to definition
            # bbox_shape = np.array([pred_data[4], pred_data[5], pred_data[6]])  # in blender coordsystem (z looking up)

            cam_intrinsics = cam_config["camera_intrinsic"]
            # cam_intrinsics[0, 2] = cam_config["width"] / 2
            # cam_intrinsics[1, 2] = cam_config["height"] / 2

            bbox3d_mean_pt_camcoord = np.dot(np.linalg.inv(cam_intrinsics),
                                             np.array([proj_3d_mean_pt_full_img[0], proj_3d_mean_pt_full_img[1], 1]))

            bbox3d_mean_pt_camcoord *= reldepth

            bbox3d_mean_pt_camcoord = np.dot(cam_extrinsic, np.append(bbox3d_mean_pt_camcoord, 1))[:3]
            return bbox3d_mean_pt_camcoord

        bbox3d_bev_iou_sum = 0
        labels_tmp = labels.detach().cpu().numpy()[:, 1:]
        outputs_tmp = kwargs["final_outputs"].detach().cpu().numpy()
        crop_pos_list = kwargs["crop_pos_list"]
        cam_config = kwargs["cam_config"]
        cam_extr_list = kwargs["cam_extr_list"]
        input_img_size = kwargs["input_img_size"]

        for i in range(outputs.shape[0]):
            pred_mean_pt_world = get_ct_pt_in_world(outputs_tmp[i], crop_pos_list[i], cam_config, cam_extr_list[i], input_img_size)
            pred_relyaw_rad = outputs_tmp[i, 1] * np.pi / 180
            pred_bbox_shape = (outputs_tmp[i, 4], outputs_tmp[i, 5], outputs_tmp[i, 6])

            pred_box_dict = {"mean_pt_world": pred_mean_pt_world,
                             "relyaw_rad": pred_relyaw_rad,
                             "bbox_shape": pred_bbox_shape}

            labels_mean_pt_world = get_ct_pt_in_world(labels_tmp[i], crop_pos_list[i], cam_config, cam_extr_list[i], input_img_size)
            labels_relyaw_rad = labels_tmp[i, 1] * np.pi / 180
            labels_bbox_shape = (labels_tmp[i, 4], labels_tmp[i, 5], labels_tmp[i, 6])

            labels_box_dict = {"mean_pt_world": labels_mean_pt_world,
                             "relyaw_rad": labels_relyaw_rad,
                             "bbox_shape": labels_bbox_shape}

            bbox3d_iou, bev_iou = calc_bbox3d_iou(pred_box_dict, labels_box_dict)
            bbox3d_bev_iou_sum += bev_iou

        return bbox3d_bev_iou_sum / outputs.shape[0]

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


class IoU(nn.Module):
    def __init__(self, threshold, reduction="mean", eps=0.00001):
        super(IoU, self).__init__()
        self.threshold = threshold
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs, labels):
        bin_outputs = (outputs > self.threshold).float()
        intersection = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) & (labels.view(labels.shape[0], -1) == 1)).sum(dim=1)
        union = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) | (labels.view(bin_outputs.shape[0], -1) == 1)).sum(dim=1)
        out = torch.true_divide(torch.add(intersection, self.eps), torch.add(union, self.eps))
        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        elif self.reduction == 'none':
            return out
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        return


class Classification_cases(nn.Module):
    def __init__(self, threshold=0.5, reduction="mean"):
        super(Classification_cases, self).__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, outputs, labels):
        return get_classification_case_rates(self.threshold, outputs, labels, self.reduction)


class Accuracy(nn.Module):
    def __init__(self, threshold=0.5):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        return (classification_case_rates.narrow(0, 0, 1) + classification_case_rates.narrow(0, 1, 1)) / classification_case_rates.sum().item()


class Precision(nn.Module):
    def __init__(self, threshold=0.5):
        super(Precision, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        return get_precision(classification_case_rates)


class Recall(nn.Module):
    def __init__(self, threshold=0.5):
        super(Recall, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        return get_recall(classification_case_rates)


class F1_Score(nn.Module):
    def __init__(self, threshold=0.5):
        super(F1_Score, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        pass

    def calc(self, classification_case_rates):
        precision = get_precision(classification_case_rates)
        recall = get_recall(classification_case_rates)
        return 2 * (precision * recall) / (precision + recall)


class False_Pos_Rate(nn.Module):
    def __init__(self, threshold=0.5):
        super(False_Pos_Rate, self).__init__()
        self.threshold = threshold

    def forward(self, classification_case_rates):
        pass

    def calc(self, classification_case_rates):
        return get_false_pos_rate(classification_case_rates)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = nn.functional.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]




def get_classification_case_rates(threshold, outputs, labels, reduction):
    bin_outputs = (outputs > threshold).float()

    true_pos = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) & (labels.view(bin_outputs.shape[0], -1) == 1)).sum(dim=1).float()
    true_neg = ((bin_outputs.view(bin_outputs.shape[0], -1) == 0) & (labels.view(bin_outputs.shape[0], -1) == 0)).sum(dim=1).float()
    false_pos = ((bin_outputs.view(bin_outputs.shape[0], -1) == 1) & (labels.view(bin_outputs.shape[0], -1) == 0)).sum(dim=1).float()
    false_neg = ((bin_outputs.view(bin_outputs.shape[0], -1) == 0) & (labels.view(bin_outputs.shape[0], -1) == 1)).sum(dim=1).float()

    if reduction == 'mean':
        true_pos = true_pos.mean()
        true_neg = true_neg.mean()
        false_pos = false_pos.mean()
        false_neg = false_neg.mean()
    elif reduction == 'sum':
        true_pos = true_pos.sum()
        true_neg = true_neg.sum()
        false_pos = false_pos.sum()
        false_neg = false_neg.sum()
    elif reduction == 'none':
        pass
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))

    return torch.stack((true_pos, true_neg, false_pos, false_neg), dim=0)


def get_precision(classification_case_rates):
    return classification_case_rates.narrow(0, 0, 1) / (
                classification_case_rates.narrow(0, 0, 1) + classification_case_rates.narrow(0, 2, 1))


def get_recall(classification_case_rates):
    return classification_case_rates.narrow(0, 0, 1) / (classification_case_rates.narrow(0, 0, 1) + classification_case_rates.narrow(0, 3, 1))


def get_false_pos_rate(classification_case_rates):
    return classification_case_rates.narrow(0, 2, 1) / (classification_case_rates.narrow(0, 2, 1) + classification_case_rates.narrow(0, 1, 1))

