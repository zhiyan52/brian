from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions" / "test"
QC_DIR = PROJECT_ROOT / "outputs" / "qc"
QC_CSV = QC_DIR / "test_prediction_qc.csv"

TEST_SUBJECTS = (
    "IBSR_02",
    "IBSR_10",
    "IBSR_15",
)

FOREGROUND_LABELS = (1, 2, 3)
EXPECTED_LABELS = (0, 1, 2, 3)

# 26-connectivity in 3D.
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def get_spatial_shape(image_data: np.ndarray) -> tuple[int, int, int]:
    """Return the three-dimensional spatial shape."""
    if image_data.ndim == 4 and image_data.shape[-1] == 1:
        return tuple(image_data.shape[:3])

    if image_data.ndim == 3:
        return tuple(image_data.shape)

    raise ValueError(
        "Expected a 3D image or a 4D image with a singleton fourth "
        f"dimension, got shape {image_data.shape}."
    )


def connected_component_stats(
    mask: np.ndarray,
) -> tuple[int, int, int, float]:
    """
    Calculate 3D connected-component statistics.

    Returns:
        number_of_components,
        largest_component_voxels,
        second_largest_component_voxels,
        largest_component_fraction
    """
    if not np.any(mask):
        return 0, 0, 0, 0.0

    labeled, num_components = ndimage.label(
        mask,
        structure=CONNECTIVITY,
    )

    component_sizes = np.bincount(labeled.ravel())[1:]

    component_sizes = np.sort(component_sizes)[::-1]

    largest = int(component_sizes[0])

    second_largest = int(component_sizes[1]) if len(component_sizes) > 1 else 0

    total = int(mask.sum())

    largest_fraction = largest / total if total > 0 else 0.0

    return (
        int(num_components),
        largest,
        second_largest,
        largest_fraction,
    )


def slice_continuity_stats(
    mask: np.ndarray,
) -> dict[str, float | int]:
    """
    Summarize foreground continuity along each spatial axis.

    These are diagnostic measures only. They do not determine whether
    the predicted anatomy is correct.
    """
    results: dict[str, float | int] = {}

    for axis, axis_name in enumerate(("x", "y", "z")):
        other_axes = tuple(i for i in range(3) if i != axis)

        slice_counts = mask.sum(axis=other_axes).astype(np.int64)

        nonempty = slice_counts > 0

        nonempty_count = int(nonempty.sum())

        if nonempty_count > 0:
            first = int(np.argmax(nonempty))

            last = int(len(nonempty) - 1 - np.argmax(nonempty[::-1]))

            internal_counts = slice_counts[first : last + 1]

            internal_empty = int(np.sum(internal_counts == 0))
        else:
            internal_empty = 0

        nonzero_counts = slice_counts[slice_counts > 0]

        if len(nonzero_counts) >= 2:
            previous = nonzero_counts[:-1].astype(np.float64)

            current = nonzero_counts[1:].astype(np.float64)

            relative_changes = np.abs(current - previous) / np.maximum(previous, 1.0)

            max_relative_change = float(np.max(relative_changes))
        else:
            max_relative_change = 0.0

        results[f"{axis_name}_nonempty_slices"] = nonempty_count

        results[f"{axis_name}_internal_empty_slices"] = internal_empty

        results[f"{axis_name}_max_relative_slice_change"] = max_relative_change

    return results


def analyze_subject(
    subject_id: str,
) -> dict[str, object]:
    """Run integrity and diagnostic QC for one test subject."""

    image_path = RAW_DIR / subject_id / f"{subject_id}.nii.gz"

    prediction_path = PRED_DIR / subject_id / f"{subject_id}_pred.nii.gz"

    print()
    print(subject_id)
    print("-" * 70)

    row: dict[str, object] = {
        "subject_id": subject_id,
    }

    # -------------------------------------------------------------
    # Existence
    # -------------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(f"MRI not found: {image_path}")

    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction not found: {prediction_path}")

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    image = nib.load(str(image_path))
    prediction = nib.load(str(prediction_path))

    image_data = np.asarray(image.dataobj)

    prediction_data = np.asarray(prediction.dataobj)

    image_spatial_shape = get_spatial_shape(image_data)

    print(f"  MRI shape:         {image_spatial_shape}")

    print(f"  Prediction shape:  {prediction_data.shape}")

    row["shape"] = str(image_spatial_shape)
    row["prediction_shape"] = str(prediction_data.shape)

    # -------------------------------------------------------------
    # Shape
    # -------------------------------------------------------------

    shape_pass = prediction_data.shape == image_spatial_shape

    if shape_pass:
        print("  PASS: spatial shape matches MRI.")
    else:
        print("  FAIL: spatial shape mismatch.")

    row["shape_pass"] = shape_pass

    # -------------------------------------------------------------
    # Affine
    # -------------------------------------------------------------

    affine_matches = np.allclose(
        image.affine,
        prediction.affine,
        atol=1e-5,
    )

    if affine_matches:
        print("  PASS: affine matches MRI.")
    else:
        print("  FAIL: affine mismatch.")

    row["affine_pass"] = affine_matches

    # -------------------------------------------------------------
    # Finite values
    # -------------------------------------------------------------

    finite_pass = bool(np.all(np.isfinite(prediction_data)))

    if finite_pass:
        print("  PASS: prediction contains only finite values.")
    else:
        print("  FAIL: prediction contains NaN or Inf.")

    row["finite_pass"] = finite_pass

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------

    labels = np.unique(prediction_data)

    print(f"  Labels:            {labels.tolist()}")

    valid_labels = bool(
        np.all(
            np.isin(
                labels,
                EXPECTED_LABELS,
            )
        )
    )

    if valid_labels:
        print("  PASS: labels are within {0,1,2,3}.")
    else:
        print("  FAIL: unexpected segmentation labels.")

    row["labels"] = ",".join(str(int(label)) for label in labels)

    row["labels_pass"] = valid_labels

    # -------------------------------------------------------------
    # Foreground
    # -------------------------------------------------------------

    prediction_foreground = prediction_data > 0

    foreground_voxels = int(prediction_foreground.sum())

    total_voxels = int(prediction_data.size)

    foreground_fraction = foreground_voxels / total_voxels if total_voxels > 0 else 0.0

    print(f"  Foreground voxels: {foreground_voxels:,}")

    print(f"  Foreground ratio:  {foreground_fraction:.4%}")

    foreground_pass = foreground_voxels > 0

    if foreground_pass:
        print("  PASS: prediction contains foreground.")
    else:
        print("  FAIL: prediction contains no foreground.")

    row["foreground_voxels"] = foreground_voxels
    row["foreground_fraction"] = foreground_fraction
    row["foreground_pass"] = foreground_pass

    # -------------------------------------------------------------
    # Data type
    # -------------------------------------------------------------

    print(f"  Prediction dtype:  {prediction_data.dtype}")

    dtype_pass = prediction_data.dtype == np.uint8

    if dtype_pass:
        print("  PASS: prediction dtype is uint8.")
    else:
        print("  WARNING: prediction is not uint8.")

    row["dtype"] = str(prediction_data.dtype)

    row["dtype_pass"] = dtype_pass

    # -------------------------------------------------------------
    # MRI foreground containment
    # -------------------------------------------------------------
    #
    # IMPORTANT:
    # MRI != 0 is NOT a validated brain mask.
    #
    # IBSR images are skull-stripped, so this metric is useful as a
    # conservative spatial sanity check, but predicted voxels outside
    # MRI foreground must NOT automatically be interpreted as
    # outside-brain false positives.
    #
    # Therefore this section is DIAGNOSTIC, not a hard pass/fail test.
    # -------------------------------------------------------------

    if image_data.ndim == 4:
        mri_volume = image_data[..., 0]
    else:
        mri_volume = image_data

    mri_foreground = mri_volume != 0

    predicted_outside_mri = prediction_foreground & ~mri_foreground

    outside_voxels = int(predicted_outside_mri.sum())

    outside_fraction_of_prediction = (
        outside_voxels / foreground_voxels if foreground_voxels > 0 else 0.0
    )

    print()
    print("  MRI foreground containment (diagnostic)")

    print(f"    MRI foreground voxels: {int(mri_foreground.sum()):,}")

    print(f"    Predicted foreground:  {foreground_voxels:,}")

    print(f"    Outside MRI foreground: {outside_voxels:,}")

    print(f"    Outside fraction:       {outside_fraction_of_prediction:.4%}")

    print("    NOTE: MRI foreground is a proxy, not a validated brain mask.")

    row["mri_foreground_voxels"] = int(mri_foreground.sum())

    row["predicted_outside_mri_foreground_voxels"] = outside_voxels

    row["predicted_outside_mri_foreground_fraction"] = outside_fraction_of_prediction

    # This is intentionally diagnostic rather than an automatic
    # failure criterion.
    row["mri_foreground_containment_status"] = (
        "NOTE: predicted foreground extends beyond MRI nonzero region"
        if outside_voxels > 0
        else "OK: all predicted foreground lies within MRI nonzero region"
    )

    # -------------------------------------------------------------
    # Native voxel volume
    # -------------------------------------------------------------

    spacing = np.asarray(
        image.header.get_zooms()[:3],
        dtype=np.float64,
    )

    voxel_volume_mm3 = float(np.prod(spacing))

    voxel_volume_ml = voxel_volume_mm3 / 1000.0

    print()
    print("  Native voxel spacing:")

    print(f"    {spacing[0]:.4f} x {spacing[1]:.4f} x {spacing[2]:.4f} mm")

    print(f"    Voxel volume: {voxel_volume_ml:.6f} mL")

    row["spacing_x_mm"] = float(spacing[0])

    row["spacing_y_mm"] = float(spacing[1])

    row["spacing_z_mm"] = float(spacing[2])

    row["voxel_volume_ml"] = voxel_volume_ml

    # -------------------------------------------------------------
    # Per-class volumes and proportions
    # -------------------------------------------------------------

    print()
    print("  Tissue volumes")

    foreground_total = sum(
        int(np.sum(prediction_data == label)) for label in FOREGROUND_LABELS
    )

    for label, name in (
        (1, "CSF"),
        (2, "GM"),
        (3, "WM"),
    ):
        mask = prediction_data == label

        count = int(mask.sum())

        volume_ml = count * voxel_volume_ml

        proportion = count / foreground_total if foreground_total > 0 else 0.0

        print(f"    {name}: {count:,} voxels | {volume_ml:.2f} mL | {proportion:.2%}")

        prefix = name.lower()

        row[f"{prefix}_voxels"] = count

        row[f"{prefix}_volume_ml"] = volume_ml

        row[f"{prefix}_foreground_fraction"] = proportion

    # -------------------------------------------------------------
    # Connected components
    # -------------------------------------------------------------

    print()
    print("  Connected components (26-connectivity)")

    for label, name in (
        (1, "CSF"),
        (2, "GM"),
        (3, "WM"),
    ):
        mask = prediction_data == label

        (
            num_components,
            largest,
            second_largest,
            largest_fraction,
        ) = connected_component_stats(mask)

        print(
            f"    {name}: "
            f"{num_components} components | "
            f"largest={largest:,} | "
            f"second={second_largest:,} | "
            f"largest fraction={largest_fraction:.2%}"
        )

        prefix = name.lower()

        row[f"{prefix}_components"] = num_components

        row[f"{prefix}_largest_component_voxels"] = largest

        row[f"{prefix}_second_largest_component_voxels"] = second_largest

        row[f"{prefix}_largest_component_fraction"] = largest_fraction

    # -------------------------------------------------------------
    # Slice continuity
    # -------------------------------------------------------------

    print()
    print("  Slice continuity (all foreground)")

    continuity = slice_continuity_stats(prediction_foreground)

    for axis_name in ("x", "y", "z"):
        nonempty = continuity[f"{axis_name}_nonempty_slices"]

        internal_empty = continuity[f"{axis_name}_internal_empty_slices"]

        max_change = continuity[f"{axis_name}_max_relative_slice_change"]

        print(
            f"    {axis_name}: "
            f"non-empty={nonempty} | "
            f"internal empty={internal_empty} | "
            f"max relative change={max_change:.2f}"
        )

        row[f"{axis_name}_nonempty_slices"] = nonempty

        row[f"{axis_name}_internal_empty_slices"] = internal_empty

        row[f"{axis_name}_max_relative_slice_change"] = max_change

    # -------------------------------------------------------------
    # Overall automatic integrity status
    # -------------------------------------------------------------
    #
    # These are HARD checks:
    #   - spatial shape
    #   - affine
    #   - finite values
    #   - valid labels
    #   - non-empty foreground
    #   - expected output dtype
    #
    # MRI foreground containment is deliberately excluded because
    # MRI != 0 is only a proxy and is not a validated brain mask.
    #
    # Connected-component topology and slice continuity are also
    # diagnostic. They cannot determine anatomical correctness
    # automatically.
    # -------------------------------------------------------------

    integrity_pass = (
        shape_pass
        and affine_matches
        and finite_pass
        and valid_labels
        and foreground_pass
        and dtype_pass
    )

    row["overall_sanity_pass"] = integrity_pass

    if integrity_pass:
        if outside_voxels > 0:
            overall_status = "PASS_WITH_NOTES"
        else:
            overall_status = "PASS"
    else:
        overall_status = "FAIL"

    row["overall_status"] = overall_status

    print()

    if overall_status == "PASS":
        print("  Overall automatic integrity status: PASS")
    elif overall_status == "PASS_WITH_NOTES":
        print("  Overall automatic integrity status: PASS_WITH_NOTES")
        print(
            "  Note: predicted foreground extends "
            "beyond MRI nonzero voxels; this is "
            "diagnostic and does not automatically "
            "indicate an anatomical error."
        )
    else:
        print("  Overall automatic integrity status: FAIL")

    return row


def write_csv(
    rows: list[dict[str, object]],
) -> None:
    """Write per-subject QC results to CSV."""
    QC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with QC_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 70)
    print("IBSR-18 TEST PREDICTION 3D QC")
    print("=" * 70)

    rows: list[dict[str, object]] = []
    all_passed = True

    for subject_id in TEST_SUBJECTS:
        try:
            row = analyze_subject(subject_id)

            rows.append(row)

            if not row["overall_sanity_pass"]:
                all_passed = False

        except Exception as exc:
            print()
            print(f"{subject_id}: ERROR: {exc}")

            all_passed = False

    write_csv(rows)

    print()
    print("=" * 70)

    print(f"QC CSV: {QC_CSV}")

    if all_passed:
        print("ALL AUTOMATIC TEST PREDICTION INTEGRITY CHECKS PASSED.")
    else:
        print("ONE OR MORE AUTOMATIC TEST PREDICTION INTEGRITY CHECKS FAILED.")

    print("Anatomical topology and MRI-foreground containment remain diagnostic.")

    print("=" * 70)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
