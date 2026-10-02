"""
Segmentation metrics for the IBSR-18 brain tissue segmentation project.

This module provides reusable metrics for multi-class 3D segmentation.

The primary metric is the Dice similarity coefficient (DSC):

    Dice = 2 * |Prediction ∩ Target|
           -------------------------
           |Prediction| + |Target|

Model outputs are expected to be raw logits with shape:

    [B, C, D, H, W]

Ground-truth labels can have either shape:

    [B, 1, D, H, W]
    [B, D, H, W]
"""

from __future__ import annotations

import torch
from torch import Tensor


def dice_per_class(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
    smooth: float = 1e-5,
) -> Tensor:
    """
    Calculate Dice score for each segmentation class.

    Parameters
    ----------
    prediction:
        Raw model logits with shape [B, C, D, H, W].

    target:
        Ground-truth class indices with shape [B, 1, D, H, W]
        or [B, D, H, W].

    num_classes:
        Number of segmentation classes.

    include_background:
        Whether class 0 should be included in the returned scores.

    smooth:
        Small value for numerical stability.

    Returns
    -------
    Tensor
        Dice score for each selected class.
    """

    if prediction.ndim != 5:
        raise ValueError(
            "Prediction must have shape [B, C, D, H, W]. "
            f"Received {tuple(prediction.shape)}."
        )

    if target.ndim not in (4, 5):
        raise ValueError(
            "Target must have shape [B, D, H, W] or "
            "[B, 1, D, H, W]. "
            f"Received {tuple(target.shape)}."
        )

    if prediction.shape[1] != num_classes:
        raise ValueError(
            "Prediction channel count does not match num_classes: "
            f"{prediction.shape[1]} vs {num_classes}."
        )

    if target.ndim == 5:
        if target.shape[1] != 1:
            raise ValueError("A 5D target must have exactly one channel.")
        target = target[:, 0]

    if prediction.shape[0] != target.shape[0]:
        raise ValueError("Prediction and target batch sizes do not match.")

    if prediction.shape[2:] != target.shape[1:]:
        raise ValueError("Prediction and target spatial dimensions do not match.")

    target = target.long()

    # Convert logits into predicted class indices.
    predicted_labels = prediction.argmax(dim=1)

    start_class = 0 if include_background else 1

    scores: list[Tensor] = []

    for class_index in range(start_class, num_classes):
        predicted_mask = predicted_labels == class_index
        target_mask = target == class_index

        intersection = (predicted_mask & target_mask).sum(dtype=torch.float32)

        predicted_volume = predicted_mask.sum(dtype=torch.float32)

        target_volume = target_mask.sum(dtype=torch.float32)

        denominator = predicted_volume + target_volume

        # If both prediction and target contain no voxels for a class,
        # that class is not present in the current batch. We assign
        # a perfect score rather than introducing NaN.
        dice = torch.where(
            denominator > 0,
            (2.0 * intersection + smooth) / (denominator + smooth),
            torch.ones_like(denominator),
        )

        scores.append(dice)

    return torch.stack(scores)


def mean_dice(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> Tensor:
    """
    Calculate mean Dice across segmentation classes.

    By default, background is excluded because the primary interest
    is tissue segmentation quality.

    Returns
    -------
    Tensor
        Scalar mean Dice score.
    """

    class_scores = dice_per_class(
        prediction=prediction,
        target=target,
        num_classes=num_classes,
        include_background=include_background,
        smooth=smooth,
    )

    return class_scores.mean()
