"""
Visualization utilities for IBSR-18 brain tissue segmentation.

The functions in this module create qualitative comparisons between
the input MRI, ground-truth segmentation, model prediction, and
prediction errors.

The implementation is intentionally independent of the training loop
so that the same visualization utilities can be reused for validation,
testing, and future experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = {
    0: "Background",
    1: "CSF",
    2: "GM",
    3: "WM",
}


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an MRI slice for visualization."""
    image = image.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value <= min_value:
        return np.zeros_like(image)

    return (image - min_value) / (max_value - min_value)


def _get_slices(
    image: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract representative axial, coronal, and sagittal slices.

    Parameters
    ----------
    image:
        3D MRI volume with shape [D, H, W].
    label:
        3D ground-truth segmentation.
    prediction:
        3D predicted segmentation.

    Returns
    -------
    Dictionary containing one slice for each anatomical orientation.
    """

    foreground = image > 0

    if np.any(foreground):
        coordinates = np.argwhere(foreground)

        min_coords = coordinates.min(axis=0)
        max_coords = coordinates.max(axis=0)

        center = (min_coords + max_coords) // 2
    else:
        center = np.array(image.shape) // 2

    d, h, w = center

    return {
        "Axial": (
            image[d, :, :],
            label[d, :, :],
            prediction[d, :, :],
        ),
        "Coronal": (
            image[:, h, :],
            label[:, h, :],
            prediction[:, h, :],
        ),
        "Sagittal": (
            image[:, :, w],
            label[:, :, w],
            prediction[:, :, w],
        ),
    }


def _create_error_map(
    label: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Return a binary map of segmentation disagreement."""
    return label != prediction


def plot_prediction_comparison(
    image: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    subject_id: str,
    output_path: Path,
) -> None:
    """
    Save a qualitative comparison of MRI, ground truth, prediction,
    and segmentation error.

    Parameters
    ----------
    image:
        3D MRI volume.
    label:
        3D ground-truth segmentation.
    prediction:
        3D predicted segmentation.
    subject_id:
        IBSR subject identifier.
    output_path:
        Destination PNG path.
    """

    slices = _get_slices(
        image=image,
        label=label,
        prediction=prediction,
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(14, 10),
    )

    fig.suptitle(
        f"{subject_id} - Qualitative Segmentation Results",
        fontsize=16,
    )

    for row, (orientation, values) in enumerate(slices.items()):
        image_slice, label_slice, prediction_slice = values

        image_slice = _normalize_image(image_slice)
        error_slice = _create_error_map(label_slice, prediction_slice)

        # MRI
        axes[row, 0].imshow(
            np.rot90(image_slice),
            cmap="gray",
        )
        axes[row, 0].set_title(f"{orientation} - MRI")

        # Ground truth
        axes[row, 1].imshow(
            np.rot90(label_slice),
            cmap="viridis",
            vmin=0,
            vmax=3,
        )
        axes[row, 1].set_title("Ground Truth")

        # Prediction
        axes[row, 2].imshow(
            np.rot90(prediction_slice),
            cmap="viridis",
            vmin=0,
            vmax=3,
        )
        axes[row, 2].set_title("Prediction")

        # Error
        axes[row, 3].imshow(
            np.rot90(error_slice),
            cmap="Reds",
            vmin=0,
            vmax=1,
        )
        axes[row, 3].set_title("Prediction Error")

        for col in range(4):
            axes[row, col].axis("off")

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)
