"""
Loss functions for multi-class 3D brain tissue segmentation.

The baseline training objective combines:

    Total Loss = Dice Loss + Cross-Entropy Loss

Optional class weights can emphasize selected tissue classes, such as
CSF, when improving boundary-sensitive segmentation is the objective.

Expected tensors
----------------
Prediction:
    [B, C, D, H, W]

Target:
    [B, 1, D, H, W]
    or
    [B, D, H, W]

where C is the number of segmentation classes.
"""

from __future__ import annotations

import torch
from monai.losses import DiceLoss
from torch import Tensor, nn


class DiceCrossEntropyLoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss for multi-class segmentation.

    Parameters
    ----------
    dice_weight:
        Weight applied to the Dice loss.

    ce_weight:
        Weight applied to the Cross-Entropy loss.

    include_background:
        Whether the background class contributes to the Dice loss.

    class_weights:
        Optional per-class weights in channel order. For IBSR-18:

            [background, CSF, GM, WM]

        The same weights are applied to Dice and Cross-Entropy loss.

    smooth_nr:
        Smoothing term added to the Dice numerator.

    smooth_dr:
        Smoothing term added to the Dice denominator.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = True,
        class_weights: list[float] | None = None,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ) -> None:
        super().__init__()

        if dice_weight < 0:
            raise ValueError("dice_weight must be >= 0.")

        if ce_weight < 0:
            raise ValueError("ce_weight must be >= 0.")

        if dice_weight == 0 and ce_weight == 0:
            raise ValueError("At least one loss weight must be greater than zero.")

        if class_weights is not None:
            if len(class_weights) != 4:
                raise ValueError(
                    "class_weights must contain exactly 4 values "
                    "for [background, CSF, GM, WM]."
                )

            if any(weight <= 0 for weight in class_weights):
                raise ValueError("All class weights must be > 0.")

            class_weights_tensor = torch.tensor(
                class_weights,
                dtype=torch.float32,
            )
        else:
            class_weights_tensor = None

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            weight=class_weights_tensor,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
        )

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute the combined Dice + Cross-Entropy loss.

        Parameters
        ----------
        prediction:
            Raw model logits with shape [B, C, D, H, W].

        target:
            Integer segmentation labels with shape
            [B, 1, D, H, W] or [B, D, H, W].

        Returns
        -------
        Tensor
            Scalar combined loss.
        """

        if prediction.ndim != 5:
            raise ValueError(
                "Prediction must have shape [B, C, D, H, W]. "
                f"Received shape: {tuple(prediction.shape)}"
            )

        if target.ndim not in (4, 5):
            raise ValueError(
                "Target must have shape [B, D, H, W] or "
                "[B, 1, D, H, W]. "
                f"Received shape: {tuple(target.shape)}"
            )

        if target.ndim == 4:
            target = target.unsqueeze(1)

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                "Prediction and target batch sizes do not match: "
                f"{prediction.shape[0]} vs {target.shape[0]}."
            )

        if prediction.shape[2:] != target.shape[2:]:
            raise ValueError(
                "Prediction and target spatial dimensions do not match: "
                f"{tuple(prediction.shape[2:])} vs "
                f"{tuple(target.shape[2:])}"
            )

        target = target.long()

        dice = self.dice_loss(prediction, target)

        ce = self.cross_entropy_loss(
            prediction,
            target[:, 0],
        )

        return self.dice_weight * dice + self.ce_weight * ce
