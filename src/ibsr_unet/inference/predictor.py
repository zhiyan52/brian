"""
Inference utilities for IBSR-18 3D brain tissue segmentation.

This module provides memory-efficient inference for full 3D MRI volumes
using MONAI's sliding-window inference.

Instead of passing an entire MRI volume through the network at once,
the volume is divided into overlapping patches. Predictions from the
patches are then combined to reconstruct the full-volume prediction.
"""

from __future__ import annotations

from collections.abc import Sequence

from monai.inferers import sliding_window_inference
from torch import Tensor, nn


def sliding_window_predict(
    model: nn.Module,
    images: Tensor,
    roi_size: Sequence[int] = (96, 96, 96),
    sw_batch_size: int = 1,
    overlap: float = 0.25,
) -> Tensor:
    """
    Run memory-efficient sliding-window inference.

    Parameters
    ----------
    model:
        Trained 3D segmentation model.

    images:
        Input images with shape
        ``[B, C, D, H, W]``.

    roi_size:
        Spatial size of each inference window.

    sw_batch_size:
        Number of windows processed simultaneously.

        A value of 1 is recommended for GPUs with limited VRAM.

    overlap:
        Fraction of overlap between neighboring windows.

    Returns
    -------
    Tensor
        Model predictions with shape
        ``[B, num_classes, D, H, W]``.

    Raises
    ------
    ValueError
        If the input tensor does not have five dimensions.
    ValueError
        If ``sw_batch_size`` is less than 1.
        If ``overlap`` is outside the valid range.
    """

    if images.ndim != 5:
        raise ValueError(
            "Expected images with shape [B, C, D, H, W], "
            f"but received shape {tuple(images.shape)}."
        )

    if sw_batch_size < 1:
        raise ValueError("sw_batch_size must be at least 1.")

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the range [0.0, 1.0).")

    return sliding_window_inference(
        inputs=images,
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )
