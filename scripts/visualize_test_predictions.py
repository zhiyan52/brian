from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions" / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "test"

TEST_SUBJECTS = ("IBSR_02", "IBSR_10", "IBSR_15")


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize MRI intensities robustly for visualization."""
    image = image.astype(np.float32)

    nonzero = image[image > 0]

    if nonzero.size == 0:
        return np.zeros_like(image)

    low, high = np.percentile(nonzero, [1, 99])

    if high <= low:
        return np.zeros_like(image)

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)

    return image


def get_middle_slices(
    prediction: np.ndarray,
) -> tuple[int, int, int]:
    """Return representative axial, coronal, and sagittal slices.

    The slice locations are based on the predicted foreground bounding box
    rather than the full zero-padded native volume.
    """
    foreground = prediction > 0

    if not np.any(foreground):
        return (
            prediction.shape[2] // 2,
            prediction.shape[1] // 2,
            prediction.shape[0] // 2,
        )

    coords = np.where(foreground)

    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    sagittal = (x_min + x_max) // 2
    coronal = (y_min + y_max) // 2
    axial = (z_min + z_max) // 2

    return axial, coronal, sagittal


def plot_test_prediction(
    subject_id: str,
    image: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    """Create MRI/prediction comparison figure."""

    image = normalize_image(image)

    axial, coronal, sagittal = get_middle_slices(prediction)

    # MRI views
    axial_mri = image[:, :, axial].T
    coronal_mri = image[:, coronal, :].T
    sagittal_mri = image[sagittal, :, :].T

    # Prediction views
    axial_pred = prediction[:, :, axial].T
    coronal_pred = prediction[:, coronal, :].T
    sagittal_pred = prediction[sagittal, :, :].T

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 14),
        constrained_layout=True,
    )

    views = [
        ("Axial", axial_mri, axial_pred),
        ("Coronal", coronal_mri, coronal_pred),
        ("Sagittal", sagittal_mri, sagittal_pred),
    ]

    for row, (view_name, mri, pred) in enumerate(views):
        axes[row, 0].imshow(
            mri,
            cmap="gray",
            origin="lower",
        )
        axes[row, 0].set_title(f"{view_name} — MRI")

        axes[row, 1].imshow(
            pred,
            cmap="viridis",
            origin="lower",
            vmin=0,
            vmax=3,
        )
        axes[row, 1].set_title(f"{view_name} — Prediction")

        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    fig.suptitle(
        f"{subject_id} — Test-set 3D U-Net Prediction",
        fontsize=16,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("GENERATING TEST PREDICTION FIGURES")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subject_id in TEST_SUBJECTS:
        image_path = RAW_DIR / subject_id / f"{subject_id}.nii.gz"
        prediction_path = PRED_DIR / subject_id / f"{subject_id}_pred.nii.gz"

        output_path = OUTPUT_DIR / f"{subject_id}_test_prediction.png"

        print()
        print(f"{subject_id}")
        print("-" * 70)

        if not image_path.exists():
            raise FileNotFoundError(f"MRI not found: {image_path}")

        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction not found: {prediction_path}")

        image = np.asarray(nib.load(str(image_path)).dataobj)

        prediction = np.asarray(nib.load(str(prediction_path)).dataobj)

        if image.ndim == 4 and image.shape[-1] == 1:
            image = image[..., 0]

        if image.shape != prediction.shape:
            raise ValueError(
                f"{subject_id}: shape mismatch: "
                f"MRI={image.shape}, prediction={prediction.shape}"
            )

        plot_test_prediction(
            subject_id=subject_id,
            image=image,
            prediction=prediction,
            output_path=output_path,
        )

        print(f"  Saved: {output_path}")

    print()
    print("=" * 70)
    print("TEST VISUALIZATION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
