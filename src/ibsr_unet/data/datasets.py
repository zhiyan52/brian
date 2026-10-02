"""
Dataset construction utilities for the IBSR-18 brain tissue segmentation
project.

This module is responsible only for constructing dataset records.

Preprocessing and augmentation are handled separately in ``transforms.py``.

Image and label sources
-----------------------
Standard experiments:

    image -> data/raw/<subject>/<subject>.nii.gz
    label -> data/raw/<subject>/<subject>_seg.nii.gz

N4 experiments:

    image -> data/processed/n4/<subject>/<subject>.nii.gz
    label -> data/raw/<subject>/<subject>_seg.nii.gz

The N4-corrected images are precomputed once and are therefore not corrected
again during every dataset access.

The public IBSR-18 test split contains images without segmentation labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from monai.data import Dataset

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IBSRSubject:
    """
    Description of one IBSR-18 subject.

    Parameters
    ----------
    subject_id:
        Subject identifier, for example ``IBSR_11``.

    image_path:
        Path to the MRI image.

    label_path:
        Path to the segmentation label, or ``None`` for unlabeled test data.
    """

    subject_id: str
    image_path: Path
    label_path: Path | None = None

    @property
    def has_label(self) -> bool:
        """Return whether this subject has a segmentation label."""

        return self.label_path is not None


# ---------------------------------------------------------------------------
# Split utilities
# ---------------------------------------------------------------------------


def _read_split_file(
    splits_dir: str | Path,
    split_name: str,
) -> list[str]:
    """
    Read subject IDs from a split file.

    Expected files:

        train.txt
        val.txt
        test.txt

    Blank lines and lines beginning with ``#`` are ignored.

    Parameters
    ----------
    splits_dir:
        Directory containing the split files.

    split_name:
        Split name: ``train``, ``val``, or ``test``.

    Returns
    -------
    list[str]
        Subject IDs in the requested split.

    Raises
    ------
    ValueError
        If an unsupported split name is supplied or duplicate subject IDs
        are found.

    FileNotFoundError
        If the split file does not exist.
    """

    if split_name not in {"train", "val", "test"}:
        raise ValueError(
            f"Unsupported split '{split_name}'. Expected one of: train, val, test."
        )

    splits_dir = Path(splits_dir)
    split_file = splits_dir / f"{split_name}.txt"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found:\n  {split_file}")

    subject_ids: list[str] = []

    with split_file.open("r", encoding="utf-8") as file:
        for line in file:
            subject_id = line.strip()

            if not subject_id or subject_id.startswith("#"):
                continue

            subject_ids.append(subject_id)

    if not subject_ids:
        raise ValueError(f"Split file is empty:\n  {split_file}")

    if len(subject_ids) != len(set(subject_ids)):
        duplicates = sorted(
            {
                subject_id
                for subject_id in subject_ids
                if subject_ids.count(subject_id) > 1
            }
        )

        raise ValueError(
            f"Duplicate subject IDs found in {split_file}:\n  {duplicates}"
        )

    return subject_ids


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def get_image_path(
    root_dir: str | Path,
    subject_id: str,
) -> Path:
    """
    Return the standard MRI image path for a subject.

    Parameters
    ----------
    root_dir:
        Root directory containing IBSR subject folders.

    subject_id:
        IBSR subject identifier.

    Returns
    -------
    pathlib.Path
        Path to the MRI NIfTI file.
    """

    root_dir = Path(root_dir)

    return root_dir / subject_id / f"{subject_id}.nii.gz"


def get_label_path(
    root_dir: str | Path,
    subject_id: str,
) -> Path:
    """
    Return the standard segmentation label path for a subject.

    Parameters
    ----------
    root_dir:
        Root directory containing IBSR subject folders.

    subject_id:
        IBSR subject identifier.

    Returns
    -------
    pathlib.Path
        Path to the segmentation NIfTI file.
    """

    root_dir = Path(root_dir)

    return root_dir / subject_id / f"{subject_id}_seg.nii.gz"


# ---------------------------------------------------------------------------
# Subject construction
# ---------------------------------------------------------------------------


def build_subject(
    subject_id: str,
    data_dir: str | Path,
    image_dir: str | Path | None = None,
    require_label: bool = True,
) -> IBSRSubject:
    """
    Build an ``IBSRSubject`` record.

    Parameters
    ----------
    subject_id:
        IBSR subject identifier.

    data_dir:
        Root directory containing the original IBSR data and labels.

        Labels are always read from this directory.

    image_dir:
        Optional directory containing alternative image files.

        When ``None``, images are read from ``data_dir``.

        For Experiment 4 this points to:

            data/processed/n4

    require_label:
        Whether a segmentation label is required.

        Set to ``False`` for the unlabeled test split.

    Returns
    -------
    IBSRSubject
        Subject description.

    Raises
    ------
    FileNotFoundError
        If a required image or label is missing.
    """

    data_dir = Path(data_dir)

    # Use the raw data directory unless an alternative image directory
    # has been explicitly supplied.
    image_root = Path(image_dir) if image_dir is not None else data_dir

    image_path = get_image_path(
        root_dir=image_root,
        subject_id=subject_id,
    )

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for {subject_id}:\n  {image_path}")

    label_path: Path | None = None

    if require_label:
        # Labels ALWAYS come from the original IBSR data directory.
        label_path = get_label_path(
            root_dir=data_dir,
            subject_id=subject_id,
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label not found for {subject_id}:\n  {label_path}"
            )

    return IBSRSubject(
        subject_id=subject_id,
        image_path=image_path,
        label_path=label_path,
    )


# ---------------------------------------------------------------------------
# Subject-list creation
# ---------------------------------------------------------------------------


def create_labeled_subjects(
    subject_ids: list[str],
    data_dir: str | Path,
    image_dir: str | Path | None = None,
) -> list[IBSRSubject]:
    """
    Create subject descriptions for a labeled split.
    """

    return [
        build_subject(
            subject_id=subject_id,
            data_dir=data_dir,
            image_dir=image_dir,
            require_label=True,
        )
        for subject_id in subject_ids
    ]


def create_unlabeled_subjects(
    subject_ids: list[str],
    data_dir: str | Path,
    image_dir: str | Path | None = None,
) -> list[IBSRSubject]:
    """
    Create subject descriptions for an unlabeled split.
    """

    return [
        build_subject(
            subject_id=subject_id,
            data_dir=data_dir,
            image_dir=image_dir,
            require_label=False,
        )
        for subject_id in subject_ids
    ]


# ---------------------------------------------------------------------------
# MONAI record conversion
# ---------------------------------------------------------------------------


def subject_to_record(
    subject: IBSRSubject,
) -> dict[str, str]:
    """
    Convert an ``IBSRSubject`` into a MONAI dataset record.

    Labeled subject:

        {
            "image": "...",
            "label": "..."
        }

    Unlabeled subject:

        {
            "image": "..."
        }
    """

    record: dict[str, str] = {
        "image": str(subject.image_path),
    }

    if subject.label_path is not None:
        record["label"] = str(subject.label_path)

    return record


def subjects_to_records(
    subjects: list[IBSRSubject],
) -> list[dict[str, str]]:
    """
    Convert multiple IBSR subjects into MONAI dataset records.
    """

    return [subject_to_record(subject) for subject in subjects]


# ---------------------------------------------------------------------------
# Generic MONAI dataset creation
# ---------------------------------------------------------------------------


def create_labeled_dataset(
    subject_ids: list[str],
    data_dir: str | Path,
    image_dir: str | Path | None = None,
    transform=None,
) -> Dataset:
    """
    Create a MONAI dataset for a labeled set of subjects.

    Images may optionally come from an alternative directory, while labels
    always come from ``data_dir``.
    """

    subjects = create_labeled_subjects(
        subject_ids=subject_ids,
        data_dir=data_dir,
        image_dir=image_dir,
    )

    records = subjects_to_records(subjects)

    return Dataset(
        data=records,
        transform=transform,
    )


def create_unlabeled_dataset(
    subject_ids: list[str],
    data_dir: str | Path,
    image_dir: str | Path | None = None,
    transform=None,
) -> Dataset:
    """
    Create a MONAI dataset for an unlabeled set of subjects.
    """

    subjects = create_unlabeled_subjects(
        subject_ids=subject_ids,
        data_dir=data_dir,
        image_dir=image_dir,
    )

    records = subjects_to_records(subjects)

    return Dataset(
        data=records,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Split-specific MONAI dataset creation
# ---------------------------------------------------------------------------


def create_train_dataset(
    data_dir: str | Path,
    splits_dir: str | Path,
    image_dir: str | Path | None = None,
    transform=None,
) -> Dataset:
    """
    Create the IBSR-18 training dataset.

    Parameters
    ----------
    data_dir:
        Original IBSR data directory.

        Labels are loaded from this directory.

    splits_dir:
        Directory containing ``train.txt``.

    image_dir:
        Optional alternative image directory.

        For Experiment 4 this should be:

            data/processed/n4

    transform:
        Training preprocessing and augmentation transform.

    Returns
    -------
    monai.data.Dataset
        Training dataset.
    """

    subject_ids = _read_split_file(
        splits_dir=splits_dir,
        split_name="train",
    )

    return create_labeled_dataset(
        subject_ids=subject_ids,
        data_dir=data_dir,
        image_dir=image_dir,
        transform=transform,
    )


def create_val_dataset(
    data_dir: str | Path,
    splits_dir: str | Path,
    image_dir: str | Path | None = None,
    transform=None,
) -> Dataset:
    """
    Create the IBSR-18 validation dataset.

    Parameters
    ----------
    data_dir:
        Original IBSR data directory.

    splits_dir:
        Directory containing ``val.txt``.

    image_dir:
        Optional alternative image directory.

        For Experiment 4 this should be:

            data/processed/n4

    transform:
        Validation preprocessing transform.

    Returns
    -------
    monai.data.Dataset
        Validation dataset.
    """

    subject_ids = _read_split_file(
        splits_dir=splits_dir,
        split_name="val",
    )

    return create_labeled_dataset(
        subject_ids=subject_ids,
        data_dir=data_dir,
        image_dir=image_dir,
        transform=transform,
    )


def create_test_dataset(
    data_dir: str | Path,
    splits_dir: str | Path,
    image_dir: str | Path | None = None,
    transform=None,
) -> Dataset:
    """
    Create the IBSR-18 test dataset.

    The public test split is unlabeled.

    Parameters
    ----------
    data_dir:
        Original IBSR data directory.

    splits_dir:
        Directory containing ``test.txt``.

    image_dir:
        Optional alternative image directory.

        For Experiment 4 this should be:

            data/processed/n4

    transform:
        Test preprocessing transform.

    Returns
    -------
    monai.data.Dataset
        Unlabeled test dataset.
    """

    subject_ids = _read_split_file(
        splits_dir=splits_dir,
        split_name="test",
    )

    return create_unlabeled_dataset(
        subject_ids=subject_ids,
        data_dir=data_dir,
        image_dir=image_dir,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Convenience inspection
# ---------------------------------------------------------------------------


def summarize_subjects(
    subjects: list[IBSRSubject],
) -> list[dict[str, str | bool]]:
    """
    Return a lightweight summary of dataset subjects.

    This is useful for debugging and smoke tests without loading
    the actual NIfTI volumes.
    """

    return [
        {
            "subject_id": subject.subject_id,
            "image": str(subject.image_path),
            "label": (
                str(subject.label_path) if subject.label_path is not None else ""
            ),
            "has_label": subject.has_label,
        }
        for subject in subjects
    ]
