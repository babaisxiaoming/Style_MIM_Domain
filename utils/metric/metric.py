import numpy as np
import torch
from torch import mean, diag
from torch import bincount
from scipy.spatial.distance import directed_hausdorff


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, axes, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = axes  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .000001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


def mean_acc_np(y_true, y_pred):
    return np.sum(y_true == y_pred) / (y_true.shape[0] * y_pred.shape[1])


# ----- Evaluation Metrics
def _fast_hist(out, target, num_classes):
    num_classes = num_classes + 1
    mask = (target >= 0) & (target < num_classes)

    hist = bincount(
        num_classes * target[mask] + out[mask],
        minlength=num_classes ** 2,
    )
    hist = hist.reshape(num_classes, num_classes)
    hist = hist.float()
    return hist


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return mean(x[x == x])


def jaccard_index(hist, smooth=1):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + smooth)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist, smooth=1):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B + smooth) / (A + B + smooth)
    avg_dice = nanmean(dice)
    return avg_dice


def dice_jaccard_sing_class(inputs, target, smooth=1):
    intersection = ((target * inputs).sum() + smooth).float()
    union_sum = (target.sum() + inputs.sum() + smooth).float()
    if target.sum() == 0 and inputs.sum() == 0:
        return (1.0, 1.0)

    dice_score = 2.0 * intersection / union_sum
    jaccard_score = intersection / (union_sum - intersection)

    return dice_score, jaccard_score


def hausforff_distance_scipy(out, target):
    out = out.detach().to('cpu').numpy().astype(bool)
    target = target.to('cpu').numpy().astype(bool)

    avg_hd = []
    for i in range(target.shape[0]):
        hd = max(directed_hausdorff(target[i, 0, :, :], out[i, 0, :, :]),
                 directed_hausdorff(out[i, 0, :, :], target[i, 0, :, :]))
        avg_hd.append(hd)

    return np.array(avg_hd).mean(), np.array(avg_hd).mean()


def eval_metrics(out, target, num_classes):
    avg_hd_95, avg_hd_100 = hausforff_distance_scipy(out, target)

    if num_classes > 1:
        hist = _fast_hist(out, target, num_classes)
        avg_jaccard = jaccard_index(hist)
        avg_dice = dice_coefficient(hist)
    else:
        avg_dice, avg_jaccard = dice_jaccard_sing_class(out, target)

    return {'avg_dice': avg_dice,
            'avg_jaccard': avg_jaccard,
            'avg_hd_95': avg_hd_95,
            'avg_hd_100': avg_hd_100}


def dice_compute(pred, groundtruth):  # batchsize*channel*H*W
    dice = []
    for i in range(4):
        dice_i = 2 * (np.sum((pred == i) * (groundtruth == i), dtype=np.float32) + 0.0001) / (
                np.sum(pred == i, dtype=np.float32) + np.sum(groundtruth == i, dtype=np.float32) + 0.0001)
        dice = dice + [dice_i]
    return dice
