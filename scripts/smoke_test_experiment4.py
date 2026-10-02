"""
Smoke test for the IBSR-18 3D U-Net Experiment 4 pipeline.

Experiment 4:
    Precomputed N4 bias-field correction
    + native voxel spacing
    + residual 3D U-Net

This script verifies the complete data and model pipeline without
starting model training.

Checks
------
1. Load the training configuration.
2. Verify precomputed N4 configuration.
3. Verify the raw and N4 directories exist.
4. Build the DataModule.
5. Verify dataset sizes.
6. Verify image/label paths.
7. Verify runtime N4 correction is disabled.
8. Load one transformed training sample.
9. Verify training patch shape, dtypes, and label values.
10. Build one DataLoader batch.
11. Verify DataLoader batch shape and dtypes.
12. Build the configured 3D U-Net.
13. Run one forward pass.
14. Verify the model output shape and number of classes.
15. Verify the model contains trainable parameters.

No optimizer step, backpropagation, or training epoch is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# ---------------------------------------------------------------------
# Make the project root importable when this script is run directly.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ruff: imports intentionally follow sys.path configuration.
from ibsr_unet.data.datamodule import IBSRDataModule  # noqa: E402
from ibsr_unet.models.unet import build_unet  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load the YAML training configuration."""

    print("\n[1/15] Loading configuration...")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found:\n{config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must contain a YAML mapping.")

    print(f"      PASS: {config_path}")

    return config


# ---------------------------------------------------------------------
# N4 configuration
# ---------------------------------------------------------------------


def check_n4_configuration(
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """Verify the Experiment 4 precomputed N4 configuration."""

    print("\n[2/15] Checking precomputed N4 configuration...")

    if "data" not in config:
        raise KeyError("Training configuration does not contain 'data'.")

    data_config = config["data"]

    use_precomputed_n4 = bool(
        data_config.get(
            "use_precomputed_n4",
            False,
        )
    )

    use_runtime_n4 = bool(
        data_config.get(
            "use_n4_bias_correction",
            False,
        )
    )

    n4_dir_value = data_config.get("n4_dir")

    if not use_precomputed_n4:
        raise AssertionError("Expected use_precomputed_n4=True for Experiment 4.")

    if use_runtime_n4:
        raise AssertionError("Runtime N4 must be disabled when using precomputed N4.")

    if n4_dir_value is None:
        raise AssertionError("n4_dir is missing from the configuration.")

    if "root_dir" not in data_config:
        raise KeyError("data.root_dir is missing from the configuration.")

    data_dir = PROJECT_ROOT / data_config["root_dir"]

    n4_dir = PROJECT_ROOT / n4_dir_value

    print("      PASS: use_precomputed_n4=True")

    print("      PASS: use_n4_bias_correction=False")

    return data_dir, n4_dir


def check_directories(
    data_dir: Path,
    n4_dir: Path,
) -> None:
    """Verify that the required data directories exist."""

    print("\n[3/15] Checking data directories...")

    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist:\n{data_dir}")

    if not n4_dir.exists():
        raise FileNotFoundError(f"Precomputed N4 directory does not exist:\n{n4_dir}")

    print(f"      PASS: raw data directory: {data_dir}")

    print(f"      PASS: N4 directory:        {n4_dir}")


# ---------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------


def build_datamodule(
    config: dict[str, Any],
) -> IBSRDataModule:
    """
    Build the DataModule using the same configuration logic as train.py.
    """

    print("\n[4/15] Building DataModule...")

    data_config = config["data"]
    training_config = config["training"]

    # -------------------------------------------------------------
    # Optional target spacing.
    # -------------------------------------------------------------

    spacing = data_config.get("spacing")

    target_spacing = (
        tuple(float(value) for value in spacing) if spacing is not None else None
    )

    # -------------------------------------------------------------
    # Training patch size.
    # -------------------------------------------------------------

    patch_size = tuple(int(value) for value in data_config["patch_size"])

    if len(patch_size) != 3:
        raise ValueError("data.patch_size must contain exactly three values.")

    # -------------------------------------------------------------
    # N4 configuration.
    # -------------------------------------------------------------

    use_n4_bias_correction = bool(
        data_config.get(
            "use_n4_bias_correction",
            False,
        )
    )

    n4_dir_value = data_config.get("n4_dir")

    n4_dir = PROJECT_ROOT / n4_dir_value if n4_dir_value is not None else None

    use_precomputed_n4 = bool(
        data_config.get(
            "use_precomputed_n4",
            False,
        )
    )

    # -------------------------------------------------------------
    # Directories.
    # -------------------------------------------------------------

    data_dir = PROJECT_ROOT / data_config["root_dir"]

    splits_dir = PROJECT_ROOT / data_config["splits_dir"]

    # -------------------------------------------------------------
    # Build DataModule.
    # -------------------------------------------------------------

    datamodule = IBSRDataModule(
        data_dir=data_dir,
        splits_dir=splits_dir,
        patch_size=patch_size,
        num_samples=int(
            training_config.get(
                "num_samples",
                1,
            )
        ),
        target_spacing=target_spacing,
        use_n4_bias_correction=(use_n4_bias_correction),
        n4_dir=n4_dir,
        use_precomputed_n4=(use_precomputed_n4),
        batch_size=int(training_config["batch_size"]),
        num_workers=int(
            training_config.get(
                "num_workers",
                0,
            )
        ),
        pin_memory=bool(
            training_config.get(
                "pin_memory",
                True,
            )
        ),
    )

    datamodule.setup()

    summary = datamodule.summary()

    print("      PASS: DataModule setup completed.")

    print(f"      Train size: {summary['train_size']}")

    print(f"      Val size:   {summary['val_size']}")

    print(f"      Test size:  {summary['test_size']}")

    return datamodule


def check_dataset_sizes(
    datamodule: IBSRDataModule,
) -> None:
    """Verify the expected IBSR-18 split sizes."""

    print("\n[5/15] Checking dataset sizes...")

    if datamodule.train_dataset is None:
        raise RuntimeError("Training dataset was not created.")

    if datamodule.val_dataset is None:
        raise RuntimeError("Validation dataset was not created.")

    if datamodule.test_dataset is None:
        raise RuntimeError("Test dataset was not created.")

    train_size = len(datamodule.train_dataset)

    val_size = len(datamodule.val_dataset)

    test_size = len(datamodule.test_dataset)

    expected = (
        10,
        5,
        3,
    )

    actual = (
        train_size,
        val_size,
        test_size,
    )

    if actual != expected:
        raise AssertionError(
            "Unexpected dataset sizes.\n"
            f"Expected: train={expected[0]}, "
            f"val={expected[1]}, "
            f"test={expected[2]}\n"
            f"Got:      train={actual[0]}, "
            f"val={actual[1]}, "
            f"test={actual[2]}"
        )

    print("      PASS: train=10, val=5, test=3.")


# ---------------------------------------------------------------------
# Dataset records
# ---------------------------------------------------------------------


def check_dataset_records(
    datamodule: IBSRDataModule,
    n4_dir: Path,
    data_dir: Path,
) -> None:
    """
    Verify that training images come from precomputed N4 and
    labels come from the original raw data.
    """

    print("\n[6/15] Checking dataset records...")

    if datamodule.train_dataset is None:
        raise RuntimeError("Training dataset was not created.")

    if len(datamodule.train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    first_record = datamodule.train_dataset.data[0]

    if "image" not in first_record:
        raise KeyError("Training record does not contain 'image'.")

    if "label" not in first_record:
        raise KeyError("Training record does not contain 'label'.")

    image_path = Path(first_record["image"])

    label_path = Path(first_record["label"])

    expected_image_root = n4_dir.resolve()

    expected_label_root = data_dir.resolve()

    if expected_image_root not in image_path.resolve().parents:
        raise AssertionError(
            "Training image is not coming from "
            "the precomputed N4 directory:\n"
            f"  {image_path}"
        )

    if expected_label_root not in label_path.resolve().parents:
        raise AssertionError(
            f"Training label is not coming from the raw data directory:\n  {label_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(f"N4 image does not exist:\n{image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label does not exist:\n{label_path}")

    print("      PASS: training image uses precomputed N4.")

    print(f"      Image: {image_path}")

    print("      PASS: training label uses raw data.")

    print(f"      Label: {label_path}")


# ---------------------------------------------------------------------
# Transform configuration
# ---------------------------------------------------------------------


def check_transform_configuration(
    datamodule: IBSRDataModule,
) -> None:
    """Verify that Experiment 4 uses precomputed N4 only."""

    print("\n[7/15] Checking transform configuration...")

    if datamodule.use_precomputed_n4 is not True:
        raise AssertionError("DataModule does not have use_precomputed_n4=True.")

    if datamodule.use_n4_bias_correction is not False:
        raise AssertionError("Runtime N4 correction is enabled.")

    print("      PASS: precomputed N4 enabled.")

    print("      PASS: runtime N4 disabled.")


# ---------------------------------------------------------------------
# Training sample
# ---------------------------------------------------------------------


def check_training_sample(
    datamodule: IBSRDataModule,
    expected_patch_size: tuple[int, int, int],
) -> None:
    """
    Load and inspect one transformed training sample.

    RandSpatialCropSamplesd returns a list of dictionaries.
    Even with num_samples=1, the output is:

        [
            {
                "image": ...,
                "label": ...
            }
        ]

    This function unwraps the first sample before checking it.
    """

    print("\n[8/15] Loading one transformed training sample...")

    if datamodule.train_dataset is None:
        raise RuntimeError("Training dataset is not available.")

    sample = datamodule.train_dataset[0]

    # -------------------------------------------------------------
    # RandSpatialCropSamplesd returns a list.
    # -------------------------------------------------------------

    if isinstance(sample, list):
        if len(sample) == 0:
            raise RuntimeError("Training transform returned an empty list.")

        sample = sample[0]

    if not isinstance(sample, dict):
        raise TypeError(
            "Expected the transformed training sample "
            "to be a dictionary, or a list containing "
            "dictionaries.\n"
            f"Got: {type(sample).__name__}"
        )

    if "image" not in sample:
        raise KeyError("Training sample does not contain 'image'.")

    if "label" not in sample:
        raise KeyError("Training sample does not contain 'label'.")

    image = sample["image"]
    label = sample["label"]

    if not isinstance(
        image,
        torch.Tensor,
    ):
        image = torch.as_tensor(image)

    if not isinstance(
        label,
        torch.Tensor,
    ):
        label = torch.as_tensor(label)

    print(f"      Image shape: {tuple(image.shape)}")

    print(f"      Label shape: {tuple(label.shape)}")

    print(f"      Image dtype: {image.dtype}")

    print(f"      Label dtype: {label.dtype}")

    # -------------------------------------------------------------
    # Expected shape:
    #
    #     [C, X, Y, Z]
    #
    # where C=1 and X/Y/Z=96.
    # -------------------------------------------------------------

    expected_image_shape = (
        1,
        *expected_patch_size,
    )

    if tuple(image.shape) != expected_image_shape:
        raise AssertionError(
            "Unexpected transformed image shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Got:      {tuple(image.shape)}"
        )

    if tuple(label.shape) != expected_image_shape:
        raise AssertionError(
            "Unexpected transformed label shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Got:      {tuple(label.shape)}"
        )

    # -------------------------------------------------------------
    # Dtypes.
    # -------------------------------------------------------------

    if image.dtype != torch.float32:
        raise AssertionError(f"Expected image dtype torch.float32, got {image.dtype}.")

    if label.dtype != torch.int64:
        raise AssertionError(f"Expected label dtype torch.int64, got {label.dtype}.")

    # -------------------------------------------------------------
    # Image validity.
    # -------------------------------------------------------------

    if not torch.isfinite(image).all():
        raise AssertionError("Training image contains NaN or infinite values.")

    # -------------------------------------------------------------
    # Label validity.
    # -------------------------------------------------------------

    unique_labels = torch.unique(label).cpu().tolist()

    invalid_labels = [
        value
        for value in unique_labels
        if value
        not in {
            0,
            1,
            2,
            3,
        }
    ]

    if invalid_labels:
        raise AssertionError(f"Invalid label values found: {invalid_labels}")

    print(f"      Label values: {unique_labels}")

    print("      PASS: transformed sample is valid.")


# ---------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------


def check_dataloader(
    datamodule: IBSRDataModule,
    expected_patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Build one training DataLoader batch and verify its shape.
    """

    print("\n[9/15] Building one DataLoader batch...")

    loader = datamodule.train_dataloader()

    batch = next(iter(loader))

    if "image" not in batch:
        raise KeyError("Training batch does not contain 'image'.")

    if "label" not in batch:
        raise KeyError("Training batch does not contain 'label'.")

    image = batch["image"]
    label = batch["label"]

    print(f"      Batch image shape: {tuple(image.shape)}")

    print(f"      Batch label shape: {tuple(label.shape)}")

    # DataLoader adds the batch dimension:
    #
    #     [B, C, X, Y, Z]
    #
    # With batch_size=1:
    #
    #     [1, 1, 96, 96, 96]

    expected_shape = (
        1,
        1,
        *expected_patch_size,
    )

    if tuple(image.shape) != expected_shape:
        raise AssertionError(
            "Unexpected DataLoader image shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(image.shape)}"
        )

    if tuple(label.shape) != expected_shape:
        raise AssertionError(
            "Unexpected DataLoader label shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(label.shape)}"
        )

    if image.dtype != torch.float32:
        raise AssertionError(
            f"Expected DataLoader image dtype torch.float32, got {image.dtype}."
        )

    if label.dtype != torch.int64:
        raise AssertionError(
            f"Expected DataLoader label dtype torch.int64, got {label.dtype}."
        )

    print("      PASS: DataLoader batch is valid.")

    return image


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def build_model(
    config: dict[str, Any],
) -> torch.nn.Module:
    """
    Build the configured 3D U-Net.
    """

    if "model" not in config:
        raise KeyError("Training configuration does not contain 'model'.")

    model_config = config["model"]

    channels = tuple(int(value) for value in model_config["channels"])

    strides = tuple(
        int(value)
        for value in model_config.get(
            "strides",
            [2, 2, 2, 2],
        )
    )

    model = build_unet(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        channels=channels,
        strides=strides,
        num_res_units=int(
            model_config.get(
                "num_res_units",
                2,
            )
        ),
    )

    return model


def check_model_configuration(
    config: dict[str, Any],
) -> torch.nn.Module:
    """Build and inspect the configured model."""

    print("\n[10/15] Building the configured 3D U-Net...")

    model = build_model(config)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    if num_parameters <= 0:
        raise AssertionError("Model contains no parameters.")

    print(f"      Parameters: {num_parameters:,}")

    print("      PASS: model configuration is valid.")

    return model


def check_model_parameters(
    model: torch.nn.Module,
) -> None:
    """Verify that the model has trainable parameters."""

    print("\n[11/15] Checking trainable model parameters...")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    if trainable_parameters <= 0:
        raise AssertionError("Model contains no trainable parameters.")

    print(f"      Trainable parameters: {trainable_parameters:,}")

    print("      PASS: trainable parameters exist.")


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------


def check_model_device(
    model: torch.nn.Module,
) -> torch.device:
    """Select the available computation device."""

    print("\n[12/15] Selecting computation device...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"      Device: {device}")

    if device.type == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    print("      PASS: computation device selected.")

    return device


def check_model_forward(
    config: dict[str, Any],
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run one forward pass through the 3D U-Net.
    """

    print("\n[13/15] Running one 3D U-Net forward pass...")

    model = model.to(device)

    image = image.to(device)

    model.eval()

    with torch.inference_mode():
        output = model(image)

    print(f"      Input shape:  {tuple(image.shape)}")

    print(f"      Output shape: {tuple(output.shape)}")

    if not isinstance(
        output,
        torch.Tensor,
    ):
        raise TypeError("Model output is not a torch.Tensor.")

    if not torch.isfinite(output).all():
        raise AssertionError("Model output contains NaN or infinite values.")

    print("      PASS: forward pass completed.")

    return output


def check_model_output(
    config: dict[str, Any],
    image: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Verify model output shape and number of classes.
    """

    print("\n[14/15] Checking model output...")

    expected_channels = int(config["model"]["out_channels"])

    expected_shape = (
        image.shape[0],
        expected_channels,
        image.shape[2],
        image.shape[3],
        image.shape[4],
    )

    if tuple(output.shape) != expected_shape:
        raise AssertionError(
            "Unexpected model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(output.shape)}"
        )

    if output.shape[1] != expected_channels:
        raise AssertionError(
            "Unexpected number of output classes.\n"
            f"Expected: {expected_channels}\n"
            f"Got:      {output.shape[1]}"
        )

    print(f"      Output classes: {output.shape[1]}")

    print("      PASS: model output shape is valid.")


def check_model_parameter_count(
    model: torch.nn.Module,
) -> None:
    """Report the final model parameter count."""

    print("\n[15/15] Final model parameter check...")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    if trainable_parameters <= 0:
        raise AssertionError("No trainable parameters found.")

    print(f"      Total parameters:     {total_parameters:,}")

    print(f"      Trainable parameters: {trainable_parameters:,}")

    print("      PASS: model parameter count is valid.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Run the complete Experiment 4 smoke test."""

    print("\n" + "=" * 70)
    print("IBSR-18 3D U-NET SMOKE TEST")
    print("=" * 70)

    try:
        # -------------------------------------------------------------
        # 1. Configuration
        # -------------------------------------------------------------

        config = load_config(CONFIG_PATH)

        # -------------------------------------------------------------
        # 2. N4 configuration
        # -------------------------------------------------------------

        data_dir, n4_dir = check_n4_configuration(config)

        # -------------------------------------------------------------
        # 3. Directories
        # -------------------------------------------------------------

        check_directories(
            data_dir=data_dir,
            n4_dir=n4_dir,
        )

        # -------------------------------------------------------------
        # 4. DataModule
        # -------------------------------------------------------------

        datamodule = build_datamodule(config)

        # -------------------------------------------------------------
        # 5. Dataset sizes
        # -------------------------------------------------------------

        check_dataset_sizes(datamodule)

        # -------------------------------------------------------------
        # 6. Dataset records
        # -------------------------------------------------------------

        check_dataset_records(
            datamodule=datamodule,
            n4_dir=n4_dir,
            data_dir=data_dir,
        )

        # -------------------------------------------------------------
        # 7. Transform configuration
        # -------------------------------------------------------------

        check_transform_configuration(datamodule)

        # -------------------------------------------------------------
        # 8. Training sample
        # -------------------------------------------------------------

        patch_size = tuple(int(value) for value in config["data"]["patch_size"])

        check_training_sample(
            datamodule=datamodule,
            expected_patch_size=patch_size,
        )

        # -------------------------------------------------------------
        # 9. DataLoader
        # -------------------------------------------------------------

        image = check_dataloader(
            datamodule=datamodule,
            expected_patch_size=patch_size,
        )

        # -------------------------------------------------------------
        # 10. Model
        # -------------------------------------------------------------

        model = check_model_configuration(config)

        # -------------------------------------------------------------
        # 11. Parameters
        # -------------------------------------------------------------

        check_model_parameters(model)

        # -------------------------------------------------------------
        # 12. Device
        # -------------------------------------------------------------

        device = check_model_device(model)

        # -------------------------------------------------------------
        # 13. Forward pass
        # -------------------------------------------------------------

        output = check_model_forward(
            config=config,
            model=model,
            image=image,
            device=device,
        )

        # -------------------------------------------------------------
        # 14. Output shape
        # -------------------------------------------------------------

        check_model_output(
            config=config,
            image=image,
            output=output,
        )

        # -------------------------------------------------------------
        # 15. Final parameter check
        # -------------------------------------------------------------

        check_model_parameter_count(model)

    except Exception as exc:
        print("\n" + "=" * 70)
        print("SMOKE TEST FAILED")
        print("=" * 70)

        print(f"\n{type(exc).__name__}: {exc}")

        raise

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)

    print("\nNo training was performed.")
    print("No optimizer step was performed.")
    print("No checkpoint was created.")


if __name__ == "__main__":
    main()
