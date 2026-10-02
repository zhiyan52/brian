"""
Analyze class distribution in the IBSR-18 segmentation labels.

This script examines the ground-truth segmentation masks and reports:

    - voxel count per class
    - percentage of voxels per class
    - number of subjects containing each class
    - per-subject class percentages
    - mean, minimum, and maximum class percentages

The analysis is useful for identifying class imbalance before
modifying the training sampler or loss function.

Only labeled subjects are analyzed. By default, this includes
the training split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from ibsr_unet.data.datasets import get_subject_paths
from ibsr_unet.data.splits import get_split_subjects

EXPECTED_LABELS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Analyze IBSR-18 segmentation class distribution."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the IBSR-18 subject folders.",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing train/val/test split files.",
    )

    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Labeled split to analyze.",
    )

    return parser.parse_args()


def analyze_subject(
    data_dir: Path,
    subject_id: str,
) -> dict[int, int]:
    """
    Count voxels belonging to each class for one subject.

    Parameters
    ----------
    data_dir:
        Root directory containing IBSR subject folders.

    subject_id:
        IBSR subject identifier.

    Returns
    -------
    dict[int, int]
        Mapping from class label to voxel count.
    """

    subject = get_subject_paths(
        data_dir=data_dir,
        subject_id=subject_id,
    )

    if subject.label is None:
        raise FileNotFoundError(f"No segmentation path available for {subject_id}.")

    if not subject.label.exists():
        raise FileNotFoundError(
            f"Segmentation file not found for {subject_id}: {subject.label}"
        )

    label_image = nib.load(str(subject.label))

    labels = np.asanyarray(label_image.dataobj)

    # Remove the singleton fourth dimension if present.
    labels = np.squeeze(labels)

    if labels.ndim != 3:
        raise ValueError(
            f"Expected a 3D segmentation for {subject_id}, "
            f"but received shape {labels.shape}."
        )

    unique_labels = np.unique(labels)

    unexpected_labels = [
        int(label) for label in unique_labels if int(label) not in EXPECTED_LABELS
    ]

    if unexpected_labels:
        raise ValueError(
            f"Unexpected labels found in {subject_id}: "
            f"{unexpected_labels}. "
            f"Expected labels: {EXPECTED_LABELS}."
        )

    return {
        class_index: int(np.count_nonzero(labels == class_index))
        for class_index in EXPECTED_LABELS
    }


def print_subject_results(
    subject_id: str,
    class_counts: dict[int, int],
) -> None:
    """Print class distribution for one subject."""

    total_voxels = sum(class_counts.values())

    print(f"\n{subject_id}")
    print("-" * 60)

    for class_index in EXPECTED_LABELS:
        count = class_counts[class_index]
        percentage = 100.0 * count / total_voxels if total_voxels > 0 else 0.0

        print(f"Class {class_index}: {count:>12,} voxels ({percentage:>7.3f}%)")


def print_summary(
    all_counts: dict[str, dict[int, int]],
) -> None:
    """Print aggregate class-distribution statistics."""

    if not all_counts:
        raise RuntimeError("No subjects were analyzed.")

    subjects = list(all_counts.keys())

    total_counts = {
        class_index: sum(
            subject_counts[class_index] for subject_counts in all_counts.values()
        )
        for class_index in EXPECTED_LABELS
    }

    total_voxels = sum(total_counts.values())

    # Calculate percentages separately for each subject.
    subject_percentages: dict[int, list[float]] = {
        class_index: [] for class_index in EXPECTED_LABELS
    }

    for subject_counts in all_counts.values():
        subject_total = sum(subject_counts.values())

        for class_index in EXPECTED_LABELS:
            percentage = (
                100.0 * subject_counts[class_index] / subject_total
                if subject_total > 0
                else 0.0
            )

            subject_percentages[class_index].append(percentage)

    print("\n" + "=" * 70)
    print("Overall Class Distribution")
    print("=" * 70)

    for class_index in EXPECTED_LABELS:
        count = total_counts[class_index]

        percentage = 100.0 * count / total_voxels if total_voxels > 0 else 0.0

        subjects_present = sum(
            count > 0
            for count in (
                all_counts[subject_id][class_index] for subject_id in subjects
            )
        )

        print(
            f"Class {class_index}: "
            f"{count:>12,} voxels "
            f"({percentage:>7.3f}%) | "
            f"present in "
            f"{subjects_present}/{len(subjects)} subjects"
        )

    print("\n" + "=" * 70)
    print("Per-Subject Class Percentage Statistics")
    print("=" * 70)

    print(f"{'Class':<10}{'Mean %':>12}{'Min %':>12}{'Max %':>12}")

    print("-" * 46)

    for class_index in EXPECTED_LABELS:
        percentages = np.asarray(
            subject_percentages[class_index],
            dtype=np.float64,
        )

        print(
            f"{class_index:<10}"
            f"{percentages.mean():>12.3f}"
            f"{percentages.min():>12.3f}"
            f"{percentages.max():>12.3f}"
        )

    print("=" * 70)


def main() -> None:
    """Run the label-distribution analysis."""

    args = parse_args()

    subject_ids = get_split_subjects(
        splits_dir=args.splits_dir,
        split=args.split,
    )

    print("=" * 70)
    print("IBSR-18 Label Distribution Analysis")
    print("=" * 70)
    print(f"Split:      {args.split}")
    print(f"Subjects:   {len(subject_ids)}")
    print(f"Data root:  {args.data_dir}")

    all_counts: dict[str, dict[int, int]] = {}

    for subject_id in subject_ids:
        class_counts = analyze_subject(
            data_dir=args.data_dir,
            subject_id=subject_id,
        )

        all_counts[subject_id] = class_counts

        print_subject_results(
            subject_id=subject_id,
            class_counts=class_counts,
        )

    print_summary(
        all_counts=all_counts,
    )


if __name__ == "__main__":
    main()
