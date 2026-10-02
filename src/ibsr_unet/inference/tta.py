from __future__ import annotations

import torch
from torch import Tensor, nn

from ibsr_unet.inference.predictor import sliding_window_predict


def left_right_flip_tta(
    model: nn.Module,
    images: Tensor,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    anatomical_axis: int = 2,
) -> Tensor:
    """
    Run conservative left-right flip test-time augmentation.

    The input tensor is expected to have shape:

        [B, C, X, Y, Z]

    after RAS orientation.

    For the current preprocessing convention, tensor dimension 2
    corresponds to the X axis of the RAS-oriented image and therefore
    represents the left-right anatomical direction.

    The procedure is:

        1. Run inference on the original image.
        2. Flip the image along the left-right axis.
        3. Run inference on the flipped image.
        4. Flip the resulting logits back.
        5. Average the original and flipped-back logits.

    Parameters
    ----------
    model:
        Trained segmentation model.

    images:
        Input MRI tensor with shape [B, C, X, Y, Z].

    roi_size:
        Sliding-window inference ROI.

    sw_batch_size:
        Number of sliding-window patches processed together.

    overlap:
        Sliding-window overlap.

    anatomical_axis:
        Tensor dimension corresponding to the left-right anatomical
        direction. For the current RAS preprocessing convention,
        this is dimension 2.

    Returns
    -------
    Tensor
        Averaged segmentation logits with shape
        [B, num_classes, X, Y, Z].
    """

    if images.ndim != 5:
        raise ValueError(
            "Expected images with shape [B, C, X, Y, Z], "
            f"but received shape {tuple(images.shape)}."
        )

    if anatomical_axis not in (2, 3, 4):
        raise ValueError(
            "anatomical_axis must be one of the spatial tensor "
            f"dimensions 2, 3, or 4, got {anatomical_axis}."
        )

    # -------------------------------------------------------------
    # Original inference.
    # -------------------------------------------------------------
    original_logits = sliding_window_predict(
        model=model,
        images=images,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )

    # -------------------------------------------------------------
    # Left-right flipped inference.
    # -------------------------------------------------------------
    flipped_images = torch.flip(
        images,
        dims=[anatomical_axis],
    )

    flipped_logits = sliding_window_predict(
        model=model,
        images=flipped_images,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )

    # -------------------------------------------------------------
    # Restore the flipped prediction to the original anatomical
    # coordinate system.
    # -------------------------------------------------------------
    flipped_logits = torch.flip(
        flipped_logits,
        dims=[anatomical_axis],
    )

    # -------------------------------------------------------------
    # Average raw logits.
    #
    # dice_per_class() performs argmax internally, so logits should
    # remain untouched here.
    # -------------------------------------------------------------
    return 0.5 * (original_logits + flipped_logits)
