"""
Smoke test for the IBSR-18 N4 bias-correction preprocessing pipeline.

This script verifies that:
1. The training configuration enables N4 bias-field correction.
2. The data module can be constructed successfully.
3. Train/validation/test datasets are created with the expected sizes.
4. N4 preprocessing runs without errors.
5. Training preprocessing produces:
   - image shape (1, D, H, W)
   - label shape (1, D, H, W)
   - float32 image
   - int64 label
   - image intensity range approximately [0, 1]
   - labels contained in {0, 1, 2, 3}
6. Validation preprocessing produces compatible outputs.
7. Image and label spatial dimensions match.

Run from the project root:

    python scripts/smoke_test_n4.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from ibsr_unet.data.datamodule import IBSRDataModule  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

EXPECTED_LABELS = {0, 1, 2, 3}

EXPECTED_TRAIN_SIZE = 10
EXPECTED_VAL_SIZE = 5
EXPECTED_TEST_SIZE = 3


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load the YAML training configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    return config


def get_first_sample(sample):
    """
    Extract the first sample when a transform returns a list.

    RandSpatialCropSamplesd may return multiple samples from one subject.
    The smoke test only needs to inspect one sample.
    """
    if isinstance(sample, list):
        if len(sample) == 0:
            raise RuntimeError("Training preprocessing returned an empty list.")

        return sample[0]

    return sample


def check_sample(
    sample: dict,
    sample_name: str,
    expected_image_shape_ndim: int = 4,
) -> None:
    """Validate the structure and contents of a preprocessed sample."""

    if not isinstance(sample, dict):
        raise TypeError(
            f"{sample_name} must be a dictionary, got {type(sample).__name__}."
        )

    if "image" not in sample:
        raise KeyError(f"{sample_name} is missing the 'image' key.")

    if "label" not in sample:
        raise KeyError(f"{sample_name} is missing the 'label' key.")

    image = sample["image"]
    label = sample["label"]

    print(f"Image type: {type(image).__name__}")
    print(f"Label type: {type(label).__name__}")
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Label shape: {tuple(label.shape)}")
    print(f"Image dtype: {image.dtype}")
    print(f"Label dtype: {label.dtype}")

    # -----------------------------------------------------------------------
    # Tensor checks
    # -----------------------------------------------------------------------

    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"{sample_name} image must be a torch.Tensor, got {type(image).__name__}."
        )

    if not isinstance(label, torch.Tensor):
        raise TypeError(
            f"{sample_name} label must be a torch.Tensor, got {type(label).__name__}."
        )

    # -----------------------------------------------------------------------
    # Dimensionality checks
    # -----------------------------------------------------------------------

    if image.ndim != expected_image_shape_ndim:
        raise AssertionError(
            f"{sample_name} image should have {expected_image_shape_ndim} "
            f"dimensions (C, D, H, W), got {image.ndim}."
        )

    if label.ndim != expected_image_shape_ndim:
        raise AssertionError(
            f"{sample_name} label should have {expected_image_shape_ndim} "
            f"dimensions (C, D, H, W), got {label.ndim}."
        )

    # -----------------------------------------------------------------------
    # Channel checks
    # -----------------------------------------------------------------------

    if image.shape[0] != 1:
        raise AssertionError(
            f"{sample_name} image should have one channel, "
            f"got shape {tuple(image.shape)}."
        )

    if label.shape[0] != 1:
        raise AssertionError(
            f"{sample_name} label should have one channel, "
            f"got shape {tuple(label.shape)}."
        )

    # -----------------------------------------------------------------------
    # Spatial shape checks
    # -----------------------------------------------------------------------

    if image.shape[1:] != label.shape[1:]:
        raise AssertionError(
            f"{sample_name} image and label spatial dimensions do not match: "
            f"{tuple(image.shape)} vs {tuple(label.shape)}."
        )

    # -----------------------------------------------------------------------
    # Dtype checks
    # -----------------------------------------------------------------------

    if image.dtype != torch.float32:
        raise AssertionError(
            f"{sample_name} image should be float32, got {image.dtype}."
        )

    if label.dtype != torch.int64:
        raise AssertionError(
            f"{sample_name} label should be int64/long, got {label.dtype}."
        )

    # -----------------------------------------------------------------------
    # Intensity checks
    # -----------------------------------------------------------------------

    image_min = float(image.min())
    image_max = float(image.max())

    print(f"Image range: {image_min:.4f} to {image_max:.4f}")

    # ScaleIntensityRangePercentilesd should produce values in [0, 1].
    tolerance = 1e-5

    if image_min < -tolerance or image_max > 1.0 + tolerance:
        raise AssertionError(
            f"{sample_name} image is not normalized to [0, 1]: "
            f"min={image_min}, max={image_max}."
        )

    # -----------------------------------------------------------------------
    # Label checks
    # -----------------------------------------------------------------------

    unique_labels = torch.unique(label).cpu().tolist()
    unique_labels = {int(value) for value in unique_labels}

    print(f"Unique labels: {sorted(unique_labels)}")

    unexpected_labels = unique_labels - EXPECTED_LABELS

    if unexpected_labels:
        raise AssertionError(
            f"{sample_name} contains unexpected labels: "
            f"{sorted(unexpected_labels)}. "
            f"Expected labels are a subset of {sorted(EXPECTED_LABELS)}."
        )

    # -----------------------------------------------------------------------
    # Final sample validation
    # -----------------------------------------------------------------------

    print(f"{sample_name} checks: PASSED")


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete N4 preprocessing smoke test."""

    print("=" * 72)
    print("IBSR-18 N4 Bias-Correction Smoke Test")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------------

    config = load_config(CONFIG_PATH)

    data_config = config.get("data", {})
    training_config = config.get("training", {})

    if not isinstance(data_config, dict):
        raise ValueError("'data' section must be a mapping.")

    if not isinstance(training_config, dict):
        raise ValueError("'training' section must be a mapping.")

    use_n4_bias_correction = bool(data_config.get("use_n4_bias_correction", False))

    print(f"N4 bias correction: {use_n4_bias_correction}")

    if not use_n4_bias_correction:
        raise AssertionError(
            "N4 bias correction is disabled in configs/train.yaml. "
            "Set data.use_n4_bias_correction to true."
        )

    # -----------------------------------------------------------------------
    # Read configuration values
    # -----------------------------------------------------------------------

    root_dir = data_config.get("root_dir", "data/raw")
    splits_dir = data_config.get("splits_dir", "data/splits")
    spacing = data_config.get("spacing")
    patch_size = tuple(data_config.get("patch_size", [96, 96, 96]))

    batch_size = int(training_config.get("batch_size", 1))
    num_samples = int(training_config.get("num_samples", 1))
    num_workers = int(training_config.get("num_workers", 0))
    pin_memory = bool(training_config.get("pin_memory", False))

    # -----------------------------------------------------------------------
    # Validate basic configuration
    # -----------------------------------------------------------------------

    if len(patch_size) != 3:
        raise AssertionError(f"patch_size must contain three values, got {patch_size}.")

    if any(int(size) <= 0 for size in patch_size):
        raise AssertionError(
            f"patch_size must contain positive values, got {patch_size}."
        )

    if batch_size < 1:
        raise AssertionError(f"batch_size must be >= 1, got {batch_size}.")

    if num_samples < 1:
        raise AssertionError(f"num_samples must be >= 1, got {num_samples}.")

    # -----------------------------------------------------------------------
    # Create data module
    # -----------------------------------------------------------------------

    datamodule = IBSRDataModule(
        data_dir=root_dir,
        splits_dir=splits_dir,
        patch_size=patch_size,
        num_samples=num_samples,
        target_spacing=spacing,
        use_n4_bias_correction=use_n4_bias_correction,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # -----------------------------------------------------------------------
    # Prepare datasets
    # -----------------------------------------------------------------------

    datamodule.setup()

    print("Datasets created successfully.")

    summary = datamodule.summary()

    print(summary)

    # -----------------------------------------------------------------------
    # Verify split sizes
    # -----------------------------------------------------------------------

    if len(datamodule.train_dataset) != EXPECTED_TRAIN_SIZE:
        raise AssertionError(
            f"Expected {EXPECTED_TRAIN_SIZE} training subjects, "
            f"got {len(datamodule.train_dataset)}."
        )

    if len(datamodule.val_dataset) != EXPECTED_VAL_SIZE:
        raise AssertionError(
            f"Expected {EXPECTED_VAL_SIZE} validation subjects, "
            f"got {len(datamodule.val_dataset)}."
        )

    if len(datamodule.test_dataset) != EXPECTED_TEST_SIZE:
        raise AssertionError(
            f"Expected {EXPECTED_TEST_SIZE} test subjects, "
            f"got {len(datamodule.test_dataset)}."
        )

    # -----------------------------------------------------------------------
    # Test training preprocessing
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("Testing training preprocessing")
    print("-" * 72)

    start_time = time.perf_counter()

    train_sample = datamodule.train_dataset[0]

    elapsed = time.perf_counter() - start_time

    print(f"Preprocessing time: {elapsed:.2f} seconds")

    train_sample = get_first_sample(train_sample)

    check_sample(
        train_sample,
        sample_name="Training sample",
    )

    # Training patches should match the configured patch size.
    train_image = train_sample["image"]

    if tuple(train_image.shape[1:]) != patch_size:
        raise AssertionError(
            "Training image spatial shape does not match patch_size: "
            f"expected {patch_size}, "
            f"got {tuple(train_image.shape[1:])}."
        )

    print("Training patch-size check: PASSED")

    # -----------------------------------------------------------------------
    # Test validation preprocessing
    # -----------------------------------------------------------------------

    print()
    print("-" * 72)
    print("Testing validation preprocessing")
    print("-" * 72)

    start_time = time.perf_counter()

    val_sample = datamodule.val_dataset[0]

    elapsed = time.perf_counter() - start_time

    print(f"Preprocessing time: {elapsed:.2f} seconds")

    val_sample = get_first_sample(val_sample)

    check_sample(
        val_sample,
        sample_name="Validation sample",
    )

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("SMOKE TEST PASSED")
    print("=" * 72)
    print(
        "N4 bias-field correction is successfully integrated into "
        "the training and validation pipelines."
    )
    print()
    print("Verified:")
    print("  ✓ N4 enabled in train.yaml")
    print("  ✓ Data module creation")
    print("  ✓ Train/validation/test split sizes")
    print("  ✓ N4 preprocessing execution")
    print("  ✓ Channel-first image/label format")
    print("  ✓ Training patch size")
    print("  ✓ Image/label spatial alignment")
    print("  ✓ Image dtype = float32")
    print("  ✓ Label dtype = int64")
    print("  ✓ Image intensity normalization to [0, 1]")
    print("  ✓ Labels contained in {0, 1, 2, 3}")


if __name__ == "__main__":
    main()
