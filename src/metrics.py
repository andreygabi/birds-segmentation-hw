import numpy as np
import torch
from torch.nn import functional as F


def image_iou(prediction, target):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Mask shapes differ: {prediction.shape} versus {target.shape}")
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def restore_logits(logits, metadata):
    if logits.ndim == 2:
        logits = logits[None, None]
    elif logits.ndim == 3:
        logits = logits[None]
    left, top = metadata["left"], metadata["top"]
    width, height = metadata["resized_width"], metadata["resized_height"]
    logits = logits[..., top:top + height, left:left + width]
    return F.interpolate(logits, size=(metadata["height"], metadata["width"]), mode="bilinear", align_corners=False)[0, 0]


def binary_mask(logits, metadata, threshold=.5):
    return (restore_logits(logits, metadata).sigmoid() >= threshold).cpu().numpy().astype(np.uint8)


def segmentation_loss(logits, targets, dice_weight=.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    if dice_weight == 0:
        return bce
    probability = logits.sigmoid()
    dimensions = (1, 2, 3)
    intersection = (probability * targets).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return (1 - dice_weight) * bce + dice_weight * dice
