"""
Utilities for reading and validating IBSR-18 subject splits.

The dataset is divided at the subject level into:
    - train: 10 subjects
    - validation: 5 subjects
    - test: 3 subjects

This module is responsible only for managing subject IDs and
preventing data leakage between train, validation, and test sets.
NIfTI loading and preprocessing are handled elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test"]


EXPECTED_SPLIT_SIZES: dict[SplitName, int] = {
    "train": 10,
    "val": 5,
    "test": 3,
}


class SplitError(ValueError):
    """Raised when the IBSR-18 split configuration is invalid."""


def read_split_file(split_file: Path) -> list[str]:
    """
    Read subject IDs from a split file.

    Blank lines and lines beginning with '#' are ignored.

    Parameters
    ----------
    split_file:
        Path to a text file containing one subject ID per line.

    Returns
    -------
    list[str]
        List of subject IDs.
    """
    split_file = Path(split_file)

    if not split_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    subjects: list[str] = []

    for line in split_file.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        subjects.append(line)

    if not subjects:
        raise SplitError(f"Split file is empty: {split_file}")

    if len(subjects) != len(set(subjects)):
        raise SplitError(f"Duplicate subject IDs found in: {split_file}")

    return subjects


def load_splits(
    splits_dir: Path,
) -> dict[SplitName, list[str]]:
    """
    Load train, validation, and test subject splits.

    Parameters
    ----------
    splits_dir:
        Directory containing train.txt, val.txt, and test.txt.

    Returns
    -------
    dict
        Dictionary containing the three subject splits.
    """
    splits_dir = Path(splits_dir)

    splits: dict[SplitName, list[str]] = {
        "train": read_split_file(splits_dir / "train.txt"),
        "val": read_split_file(splits_dir / "val.txt"),
        "test": read_split_file(splits_dir / "test.txt"),
    }

    validate_splits(splits)

    return splits


def validate_splits(
    splits: dict[SplitName, list[str]],
    expected_sizes: dict[SplitName, int] | None = None,
) -> None:
    """
    Validate IBSR-18 subject splits.

    Validation checks:
        1. All required split names are present.
        2. Each split has the expected number of subjects.
        3. No subject occurs in more than one split.
        4. Exactly 18 unique subjects are present.

    Parameters
    ----------
    splits:
        Dictionary containing train, validation, and test subjects.

    expected_sizes:
        Optional expected number of subjects in each split.

    Raises
    ------
    SplitError
        If the split configuration is invalid.
    """
    if expected_sizes is None:
        expected_sizes = EXPECTED_SPLIT_SIZES

    required_splits = {"train", "val", "test"}

    if set(splits) != required_splits:
        raise SplitError("Splits must contain exactly: train, val, test")

    for split_name, expected_size in expected_sizes.items():
        actual_size = len(splits[split_name])

        if actual_size != expected_size:
            raise SplitError(
                f"{split_name} split contains {actual_size} subjects; "
                f"expected {expected_size}."
            )

    all_subjects: list[str] = []

    for subjects in splits.values():
        all_subjects.extend(subjects)

    if len(all_subjects) != len(set(all_subjects)):
        duplicates = {
            subject for subject in all_subjects if all_subjects.count(subject) > 1
        }

        raise SplitError(
            f"Subject leakage detected between splits: {sorted(duplicates)}"
        )

    if len(set(all_subjects)) != 18:
        raise SplitError(
            f"Expected 18 unique IBSR subjects, found {len(set(all_subjects))}."
        )


def get_split_subjects(
    splits_dir: Path,
    split: SplitName,
) -> list[str]:
    """
    Return subject IDs belonging to a specific split.

    Parameters
    ----------
    splits_dir:
        Directory containing the split files.

    split:
        One of 'train', 'val', or 'test'.

    Returns
    -------
    list[str]
        Subject IDs for the requested split.
    """
    splits = load_splits(splits_dir)

    return splits[split]


def get_all_subjects(
    splits_dir: Path,
) -> list[str]:
    """
    Return all IBSR-18 subjects across train, validation, and test.
    """
    splits = load_splits(splits_dir)

    subjects: list[str] = []

    for split_subjects in splits.values():
        subjects.extend(split_subjects)

    return subjects


def print_split_summary(
    splits_dir: Path,
) -> None:
    """
    Print a human-readable summary of the dataset splits.
    """
    splits = load_splits(splits_dir)

    print("IBSR-18 Dataset Splits")
    print("=" * 40)

    for split_name, subjects in splits.items():
        print(f"{split_name:>5}: {len(subjects):2d} subjects | {', '.join(subjects)}")

    print("=" * 40)
    print(f"Total: {len(get_all_subjects(splits_dir))} subjects")
