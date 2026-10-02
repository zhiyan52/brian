"""
Validate the IBSR-18 dataset before training.

This script performs dataset-level integrity checks without modifying
the original NIfTI files.

Checks include:
    - Subject split consistency
    - Image and segmentation file existence
    - Image/label shape consistency
    - Image/label affine consistency
    - Voxel spacing
    - Orientation
    - Image intensity statistics
    - Segmentation label values

The script is intended to be run before preprocessing and training.

Example
-------
From the project root:

    python scripts/prepare_data.py \
        --data-dir data/raw \
        --splits-dir data/splits
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from ibsr_unet.data.datasets import get_image_path, get_label_path
from ibsr_unet.data.splits import load_splits

EXPECTED_LABELS = {0, 1, 2, 3}


@dataclass
class SubjectReport:
    """
    Store validation information for one IBSR subject.
    """

    subject_id: str
    split: str

    image_shape: tuple[int, ...]
    image_spacing: tuple[float, ...]
    image_orientation: tuple[str, ...]
    image_dtype: str

    image_min: float
    image_max: float
    image_mean: float
    image_nonzero_voxels: int

    label_shape: tuple[int, ...] | None
    label_spacing: tuple[float, ...] | None
    label_orientation: tuple[str, ...] | None
    label_values: set[int] | None

    shape_match: bool | None
    affine_match: bool | None


def get_spacing(image: nib.Nifti1Image) -> tuple[float, ...]:
    """
    Return voxel spacing from NIfTI header metadata.
    """
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def get_orientation(image: nib.Nifti1Image) -> tuple[str, ...]:
    """
    Return the anatomical orientation codes of a NIfTI image.
    """
    return tuple(code for code in nib.aff2axcodes(image.affine))


def get_image_statistics(
    image: nib.Nifti1Image,
) -> tuple[float, float, float, int]:
    """
    Calculate basic image intensity statistics.

    Returns
    -------
    tuple
        Minimum, maximum, mean intensity, and number of nonzero voxels.
    """
    data = np.asanyarray(image.dataobj)

    data_float = np.asarray(data, dtype=np.float32)

    return (
        float(data_float.min()),
        float(data_float.max()),
        float(data_float.mean()),
        int(np.count_nonzero(data_float)),
    )


def get_label_values(
    label_image: nib.Nifti1Image,
) -> set[int]:
    """
    Return unique integer segmentation labels.
    """
    data = np.asanyarray(label_image.dataobj)

    return {int(value) for value in np.unique(data)}


def affines_match(
    image: nib.Nifti1Image,
    label: nib.Nifti1Image,
    tolerance: float = 1e-5,
) -> bool:
    """
    Check whether image and label affine matrices match.
    """
    return bool(
        np.allclose(
            image.affine,
            label.affine,
            atol=tolerance,
            rtol=0.0,
        )
    )


def validate_subject(
    data_dir: Path,
    subject_id: str,
    split: str,
) -> SubjectReport:
    """
    Validate one IBSR subject and generate a report.
    """
    image_path = get_image_path(
        root_dir=data_dir,
        subject_id=subject_id,
    )

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for {subject_id}: {image_path}")

    image = nib.load(image_path)

    image_shape = tuple(int(value) for value in image.shape)
    image_spacing = get_spacing(image)
    image_orientation = get_orientation(image)

    image_min, image_max, image_mean, image_nonzero = get_image_statistics(image)

    label_shape = None
    label_spacing = None
    label_orientation = None
    label_values = None
    shape_match = None
    affine_match = None

    if split in {"train", "val"}:
        label_path = get_label_path(
            root_dir=data_dir,
            subject_id=subject_id,
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Segmentation not found for {subject_id}: {label_path}"
            )

        label = nib.load(label_path)

        label_shape = tuple(int(value) for value in label.shape)
        label_spacing = get_spacing(label)
        label_orientation = get_orientation(label)
        label_values = get_label_values(label)

        shape_match = image_shape == label_shape
        affine_match = affines_match(image, label)

    return SubjectReport(
        subject_id=subject_id,
        split=split,
        image_shape=image_shape,
        image_spacing=image_spacing,
        image_orientation=image_orientation,
        image_dtype=str(image.get_data_dtype()),
        image_min=image_min,
        image_max=image_max,
        image_mean=image_mean,
        image_nonzero_voxels=image_nonzero,
        label_shape=label_shape,
        label_spacing=label_spacing,
        label_orientation=label_orientation,
        label_values=label_values,
        shape_match=shape_match,
        affine_match=affine_match,
    )


def validate_label_values(
    reports: list[SubjectReport],
) -> list[str]:
    """
    Check that labeled subjects contain only the expected labels.
    """
    errors: list[str] = []

    for report in reports:
        if report.label_values is None:
            continue

        unexpected = report.label_values - EXPECTED_LABELS

        if unexpected:
            errors.append(
                f"{report.subject_id}: unexpected labels {sorted(unexpected)}"
            )

        missing_expected = EXPECTED_LABELS - report.label_values

        if missing_expected:
            errors.append(
                f"{report.subject_id}: missing labels {sorted(missing_expected)}"
            )

    return errors


def validate_geometry(
    reports: list[SubjectReport],
) -> list[str]:
    """
    Check image/label geometry consistency.
    """
    errors: list[str] = []

    for report in reports:
        if report.shape_match is False:
            errors.append(
                f"{report.subject_id}: image and label shapes "
                f"do not match "
                f"({report.image_shape} vs {report.label_shape})"
            )

        if report.affine_match is False:
            errors.append(f"{report.subject_id}: image and label affines do not match")

    return errors


def print_subject_report(report: SubjectReport) -> None:
    """
    Print a readable report for one subject.
    """
    print(f"\n{report.subject_id} [{report.split}]")
    print("-" * 60)

    print(f"Image shape:       {report.image_shape}")
    print(f"Image spacing:     {tuple(round(x, 4) for x in report.image_spacing)}")
    print(f"Orientation:       {report.image_orientation}")
    print(f"Image dtype:       {report.image_dtype}")

    print(f"Intensity range:   {report.image_min:.3f} - {report.image_max:.3f}")
    print(f"Mean intensity:    {report.image_mean:.3f}")
    print(f"Nonzero voxels:    {report.image_nonzero_voxels:,}")

    if report.label_shape is not None:
        print(f"Label shape:       {report.label_shape}")
        print(
            "Label spacing:     "
            f"{tuple(round(x, 4) for x in report.label_spacing or ())}"
        )
        print(f"Label orientation: {report.label_orientation}")
        print(f"Label values:      {sorted(report.label_values or set())}")
        print(f"Shape match:       {report.shape_match}")
        print(f"Affine match:      {report.affine_match}")
    else:
        print("Segmentation:      not available")


def print_summary(
    reports: list[SubjectReport],
) -> None:
    """
    Print a dataset-level summary.
    """
    print("\n")
    print("=" * 70)
    print("IBSR-18 DATASET SUMMARY")
    print("=" * 70)

    split_counts: dict[str, int] = {}

    for report in reports:
        split_counts[report.split] = split_counts.get(report.split, 0) + 1

    for split, count in split_counts.items():
        print(f"{split:>10}: {count} subjects")

    print(f"{'total':>10}: {len(reports)} subjects")

    shapes = sorted({report.image_shape for report in reports})

    spacings = sorted(
        {tuple(round(value, 4) for value in report.image_spacing) for report in reports}
    )

    orientations = sorted({report.image_orientation for report in reports})

    print("\nImage shapes:")
    for shape in shapes:
        print(f"  {shape}")

    print("\nVoxel spacings:")
    for spacing in spacings:
        print(f"  {spacing}")

    print("\nOrientations:")
    for orientation in orientations:
        print(f"  {orientation}")

    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Validate the IBSR-18 dataset.")

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing IBSR subject folders.",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory containing train.txt, val.txt, and test.txt.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the IBSR-18 dataset validation.
    """
    args = parse_args()

    data_dir = args.data_dir
    splits_dir = args.splits_dir

    print("=" * 70)
    print("IBSR-18 DATASET VALIDATION")
    print("=" * 70)

    print(f"Data directory:   {data_dir}")
    print(f"Splits directory: {splits_dir}")

    # load_splits() also validates:
    #   - required split files
    #   - split sizes
    #   - duplicate subjects
    #   - subject leakage
    #   - total number of subjects
    splits = load_splits(splits_dir)

    reports: list[SubjectReport] = []

    for split_name, subject_ids in splits.items():
        for subject_id in subject_ids:
            report = validate_subject(
                data_dir=data_dir,
                subject_id=subject_id,
                split=split_name,
            )

            reports.append(report)

            print_subject_report(report)

    geometry_errors = validate_geometry(reports)
    label_errors = validate_label_values(reports)

    print_summary(reports)

    errors = geometry_errors + label_errors

    print("\nValidation result")
    print("=" * 70)

    if errors:
        print("FAILED")
        print("\nProblems detected:")

        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    print("PASSED")
    print("All dataset integrity checks passed.")


if __name__ == "__main__":
    main()
