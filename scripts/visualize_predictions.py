"""
Generate qualitative visualizations of IBSR-18 segmentation predictions.

This script:
1. Loads the selected Experiment 1 checkpoint.
2. Runs inference on the validation subjects.
3. Uses the established sliding-window inference configuration.
4. Inverts the validation spatial preprocessing so predictions are
   restored to the original native NIfTI geometry.
5. Saves predicted segmentations as NIfTI files.
6. Generates axial, coronal, and sagittal qualitative comparisons.

Selected model:
    Experiment 1
    Residual 3D U-Net
    Native spacing
    No N4 preprocessing
    Best validation mean foreground Dice: 0.8840
    Best epoch: 99

Expected output:

outputs/
├── predictions/
│   ├── IBSR_11/
│   │   └── prediction.nii.gz
│   ├── IBSR_12/
│   │   └── prediction.nii.gz
│   ├── IBSR_13/
│   │   └── prediction.nii.gz
│   ├── IBSR_14/
│   │   └── prediction.nii.gz
│   └── IBSR_17/
│       └── prediction.nii.gz
│
└── visualizations/
    ├── IBSR_11_prediction.png
    ├── IBSR_12_prediction.png
    ├── IBSR_13_prediction.png
    ├── IBSR_14_prediction.png
    └── IBSR_17_prediction.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.transforms import Invertd

from ibsr_unet.data.datamodule import IBSRDataModule
from ibsr_unet.inference.predictor import sliding_window_predict
from ibsr_unet.models.unet import build_unet
from ibsr_unet.visualization.plots import plot_prediction_comparison

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt"

PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"

VISUALIZATION_DIR = PROJECT_ROOT / "outputs" / "visualizations"


# ---------------------------------------------------------------------
# Expected selected experiment
# ---------------------------------------------------------------------

EXPECTED_EXPERIMENT = 1
EXPECTED_BEST_DICE = 0.9185
EXPECTED_BEST_EPOCH = 391


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load the YAML configuration file.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Expected the YAML configuration to contain a dictionary.")

    return config


# ---------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------


def build_datamodule(
    config: dict[str, Any],
) -> IBSRDataModule:
    """
    Build the IBSR-18 data module using the same preprocessing
    configuration as Experiment 1.

    Experiment 1:
        - native spacing
        - no runtime N4
        - no precomputed N4
    """

    data_config = config["data"]
    training_config = config["training"]

    # -------------------------------------------------------------
    # Resolve data paths.
    # -------------------------------------------------------------

    data_dir = PROJECT_ROOT / data_config["root_dir"]

    splits_dir = PROJECT_ROOT / data_config["splits_dir"]

    # -------------------------------------------------------------
    # Native spacing.
    #
    # Experiment 1 does not resample the images.
    # -------------------------------------------------------------

    spacing_config = data_config.get("spacing")

    if spacing_config is None:
        target_spacing = None
    else:
        target_spacing = tuple(float(value) for value in spacing_config)

    # -------------------------------------------------------------
    # Patch size.
    # -------------------------------------------------------------

    patch_size = tuple(
        int(value)
        for value in data_config.get(
            "patch_size",
            [96, 96, 96],
        )
    )

    # -------------------------------------------------------------
    # Data-loader configuration.
    # -------------------------------------------------------------

    num_samples = int(
        training_config.get(
            "num_samples",
            1,
        )
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            1,
        )
    )

    num_workers = int(
        training_config.get(
            "num_workers",
            0,
        )
    )

    pin_memory = bool(
        training_config.get(
            "pin_memory",
            torch.cuda.is_available(),
        )
    )

    # -------------------------------------------------------------
    # Create data module.
    # -------------------------------------------------------------

    return IBSRDataModule(
        data_dir=data_dir,
        splits_dir=splits_dir,
        patch_size=patch_size,
        num_samples=num_samples,
        target_spacing=target_spacing,
        use_n4_bias_correction=False,
        n4_dir=None,
        use_precomputed_n4=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def build_model(
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """
    Build the 3D U-Net using the architecture specified in train.yaml.
    """

    model_config = config["model"]

    model = build_unet(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        channels=tuple(int(value) for value in model_config["channels"]),
        strides=tuple(int(value) for value in model_config["strides"]),
        num_res_units=int(model_config["num_res_units"]),
    )

    return model.to(device)


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load the trained model checkpoint.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Expected checkpoint to contain a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


# ---------------------------------------------------------------------
# Subject information
# ---------------------------------------------------------------------


def get_subject_info_from_dataset(
    dataset: Any,
    index: int,
) -> tuple[str, Path]:
    """
    Retrieve the subject ID and reference MRI path from the original
    MONAI dataset item.

    The validation DataLoader does not include a `subject_id` field in
    its batches. The underlying MONAI Dataset retains the original
    sample dictionaries in `dataset.data`.

    Returns
    -------
    tuple[str, Path]
        Subject ID and original reference MRI path.
    """

    if not hasattr(dataset, "data"):
        raise AttributeError(
            "Validation dataset does not expose the expected `data` attribute."
        )

    dataset_data = dataset.data

    if not isinstance(
        dataset_data,
        (list, tuple),
    ):
        raise TypeError("Expected validation dataset.data to be a list or tuple.")

    if index < 0 or index >= len(dataset_data):
        raise IndexError(f"Dataset index {index} is out of range.")

    data_item = dataset_data[index]

    if not isinstance(
        data_item,
        dict,
    ):
        raise TypeError(
            "Expected each validation dataset item to be a dictionary, "
            f"but received {type(data_item).__name__}."
        )

    image_path = data_item.get("image")

    if image_path is None:
        raise KeyError("Validation dataset item does not contain an 'image' path.")

    if isinstance(
        image_path,
        (list, tuple),
    ):
        if len(image_path) != 1:
            raise ValueError(
                "Expected exactly one image path for a validation sample, "
                f"but found {len(image_path)}."
            )

        image_path = image_path[0]

    image_path = Path(str(image_path))

    if not image_path.exists():
        raise FileNotFoundError(
            f"Reference MRI from dataset item does not exist: {image_path}"
        )

    subject_id = image_path.parent.name

    if not subject_id.startswith("IBSR_"):
        raise ValueError(
            f"Could not determine a valid IBSR subject ID from image path: {image_path}"
        )

    return (
        subject_id,
        image_path,
    )


# ---------------------------------------------------------------------
# Prediction inversion
# ---------------------------------------------------------------------


def invert_prediction_to_native_space(
    prediction: torch.Tensor,
    batch: dict[str, Any],
    validation_transforms: Any,
) -> torch.Tensor:
    """
    Invert the validation spatial preprocessing applied to the MRI.

    The validation pipeline contains CropForegroundd, which changes
    the spatial dimensions before inference. MONAI records the spatial
    operations in the image MetaTensor metadata.

    This function uses MONAI Invertd to restore the prediction to the
    original native NIfTI geometry.

    Parameters
    ----------
    prediction:
        Predicted class labels with shape:

            [B, X, Y, Z]

        for the current batch.

    batch:
        Validation batch containing the transformed image and its
        MONAI metadata.

    validation_transforms:
        The exact Compose transform used by the validation dataset.

    Returns
    -------
    torch.Tensor
        Prediction restored to native spatial dimensions with shape:

            [B, X, Y, Z]
    """

    if prediction.ndim != 4:
        raise ValueError(
            "Expected prediction to have shape [B,X,Y,Z], "
            f"but received {tuple(prediction.shape)}."
        )

    # -------------------------------------------------------------
    # Attach the prediction to the transformed batch.
    #
    # Invertd uses the transform history recorded on the original
    # image (`orig_keys="image"`) to undo the spatial operations.
    # -------------------------------------------------------------

    inversion_batch = dict(batch)

    inversion_batch["prediction"] = prediction.to(dtype=torch.float32)

    # -------------------------------------------------------------
    # Invert the validation transforms.
    #
    # nearest_interp=True is essential because this is a discrete
    # segmentation map.
    # -------------------------------------------------------------

    inverter = Invertd(
        keys="prediction",
        transform=validation_transforms,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    )

    inversion_batch = inverter(inversion_batch)

    restored_prediction = inversion_batch["prediction"]

    if isinstance(
        restored_prediction,
        torch.Tensor,
    ):
        restored_prediction = restored_prediction.detach().cpu()
    else:
        restored_prediction = torch.as_tensor(restored_prediction)

    # -------------------------------------------------------------
    # Invertd normally returns [B,1,X,Y,Z] when operating with
    # channel-first metadata. Remove the singleton channel.
    # -------------------------------------------------------------

    if restored_prediction.ndim == 5 and restored_prediction.shape[1] == 1:
        restored_prediction = restored_prediction[:, 0]

    if restored_prediction.ndim != 4:
        raise ValueError(
            f"Unexpected inverted prediction shape: {tuple(restored_prediction.shape)}"
        )

    # -------------------------------------------------------------
    # Convert back to discrete class labels.
    # -------------------------------------------------------------

    restored_prediction = torch.round(restored_prediction).to(dtype=torch.uint8)

    return restored_prediction


# ---------------------------------------------------------------------
# NIfTI saving
# ---------------------------------------------------------------------


def save_prediction_nifti(
    prediction: np.ndarray,
    reference_path: Path,
    output_path: Path,
) -> None:
    """
    Save a predicted segmentation as NIfTI using the reference MRI's
    spatial metadata.

    The prediction is saved as uint8 because the segmentation contains
    discrete class labels:

        0 = Background
        1 = CSF
        2 = GM
        3 = WM
    IBSR raw images may have a trailing singleton dimension:

        (X, Y, Z, 1)

    whereas the predicted segmentation is naturally:

        (X, Y, Z)

    The singleton dimension is therefore ignored when validating the
    spatial geometry.

    """

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference MRI not found: {reference_path}")

    reference = nib.load(str(reference_path))

    prediction = np.asarray(
        prediction,
        dtype=np.uint8,
    )

    # -------------------------------------------------------------
    # Normalize reference geometry.
    #
    # Raw IBSR files can be:
    #
    #     (X, Y, Z, 1)
    #
    # while the segmentation is:
    #
    #     (X, Y, Z)
    #
    # The trailing singleton dimension is not spatial.
    # -------------------------------------------------------------

    reference_shape = reference.shape

    if len(reference_shape) == 4 and reference_shape[-1] == 1:
        reference_spatial_shape = reference_shape[:3]
    elif len(reference_shape) == 3:
        reference_spatial_shape = reference_shape
    else:
        raise ValueError(
            "Unexpected reference MRI shape: "
            f"{reference_shape}. "
            "Expected (X,Y,Z) or (X,Y,Z,1)."
        )

    # -------------------------------------------------------------
    # Verify spatial geometry.
    # -------------------------------------------------------------

    if prediction.shape != reference_spatial_shape:
        raise ValueError(
            "Prediction/reference spatial shape mismatch: "
            f"prediction={prediction.shape}, "
            f"reference_spatial={reference_spatial_shape}, "
            f"reference_raw={reference_shape}"
        )

    # -------------------------------------------------------------
    # Preserve the reference affine and header.
    # -------------------------------------------------------------

    header = reference.header.copy()

    header.set_data_dtype(np.uint8)

    # Ensure the header describes the 3D prediction rather than the
    # raw image's trailing singleton dimension.
    header.set_data_shape(prediction.shape)

    prediction_image = nib.Nifti1Image(
        prediction,
        affine=reference.affine,
        header=header,
    )

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        prediction_image,
        str(output_path),
    )


# ---------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------


def tensor_to_numpy(
    value: Any,
) -> np.ndarray:
    """
    Convert a tensor-like value to a NumPy array.
    """

    if isinstance(
        value,
        torch.Tensor,
    ):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def remove_singleton_channel(
    array: np.ndarray,
) -> np.ndarray:
    """
    Remove a leading singleton channel dimension if present.
    """

    if array.ndim == 4 and array.shape[0] == 1:
        return array[0]

    return array


# ---------------------------------------------------------------------
# Visualization generation
# ---------------------------------------------------------------------


def generate_visualizations(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    prediction_dir: Path,
    visualization_dir: Path,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    validation_transforms: Any,
) -> None:
    """
    Run inference on the validation set, restore predictions to native
    space, save NIfTI predictions, and generate qualitative figures.
    """

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    dataset = dataloader.dataset

    if not hasattr(
        dataset,
        "data",
    ):
        raise AttributeError(
            "Validation DataLoader dataset does not expose `.data`. "
            "Cannot determine subject IDs and reference image paths."
        )

    print()
    print("Generating qualitative results...")
    print("-" * 70)

    print("Inference configuration:")
    print(f"  ROI size:         {roi_size}")
    print(f"  SW batch size:    {sw_batch_size}")
    print(f"  Overlap:          {overlap}")
    print("  Target spacing:   native")
    print("  N4 preprocessing: disabled")
    print("  Spatial inversion: enabled")

    print("-" * 70)

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            # -----------------------------------------------------
            # Subject ID and original reference MRI.
            # -----------------------------------------------------

            subject_id, reference_path = get_subject_info_from_dataset(
                dataset=dataset,
                index=index,
            )

            print(f"Processing {subject_id}...")

            # -----------------------------------------------------
            # Move image to device.
            # -----------------------------------------------------

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            # -----------------------------------------------------
            # Ground truth.
            #
            # This is still in transformed/cropped validation space,
            # which is exactly the space used for inference.
            # -----------------------------------------------------

            label_np = remove_singleton_channel(tensor_to_numpy(batch["label"])[0])

            # -----------------------------------------------------
            # Transformed MRI volume.
            #
            # Used for the qualitative visualization. This is the
            # same preprocessed volume seen by the model.
            # -----------------------------------------------------

            image_np = images[0, 0].detach().cpu().numpy()

            # -----------------------------------------------------
            # Sliding-window inference.
            # -----------------------------------------------------

            predictions = sliding_window_predict(
                model=model,
                images=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )

            # -----------------------------------------------------
            # Convert logits to class labels.
            # -----------------------------------------------------

            prediction_labels = torch.argmax(
                predictions,
                dim=1,
            )

            # prediction_labels:
            #     [B, X, Y, Z]
            #
            # At this point the prediction is still in cropped
            # validation space.
            # -----------------------------------------------------

            transformed_prediction = prediction_labels.detach().cpu()

            print(
                f"  Transformed prediction shape: {tuple(transformed_prediction.shape)}"
            )

            # -----------------------------------------------------
            # Restore prediction to native image space.
            # -----------------------------------------------------

            native_prediction = invert_prediction_to_native_space(
                prediction=transformed_prediction,
                batch=batch,
                validation_transforms=validation_transforms,
            )

            prediction_np = native_prediction[0].numpy().astype(np.uint8)

            print(f"  Native prediction shape:      {prediction_np.shape}")

            print(
                f"  Reference MRI shape:          {nib.load(str(reference_path)).shape}"
            )

            # -----------------------------------------------------
            # Validate prediction labels.
            # -----------------------------------------------------

            unique_labels = np.unique(prediction_np)

            if not np.all(
                np.isin(
                    unique_labels,
                    [0, 1, 2, 3],
                )
            ):
                raise ValueError(
                    f"{subject_id}: unexpected prediction labels: {unique_labels}"
                )

            print(f"  Prediction labels: {unique_labels.tolist()}")

            # -----------------------------------------------------
            # Save native-space prediction.
            # -----------------------------------------------------

            subject_prediction_dir = prediction_dir / subject_id

            prediction_path = subject_prediction_dir / "prediction.nii.gz"

            save_prediction_nifti(
                prediction=prediction_np,
                reference_path=reference_path,
                output_path=prediction_path,
            )

            print(f"  Saved prediction: {prediction_path}")

            # -----------------------------------------------------
            # Generate qualitative figure.
            #
            # The plotting function expects image, label, and
            # prediction to have matching spatial dimensions.
            #
            # Therefore, for qualitative visualization we use the
            # transformed validation-space image/label/prediction.
            # -----------------------------------------------------

            transformed_prediction_np = (
                transformed_prediction[0].numpy().astype(np.uint8)
            )

            visualization_path = visualization_dir / f"{subject_id}_prediction.png"

            plot_prediction_comparison(
                image=image_np,
                label=label_np,
                prediction=transformed_prediction_np,
                subject_id=subject_id,
                output_path=visualization_path,
            )

            print(f"  Saved visualization: {visualization_path}")

            print()

    print("-" * 70)

    print("Qualitative evaluation complete.")

    print()
    print(f"Predictions:     {prediction_dir}")
    print(f"Visualizations:  {visualization_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """
    Main entry point.
    """

    # -------------------------------------------------------------
    # Select device.
    # -------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # -------------------------------------------------------------
    # Load configuration.
    # -------------------------------------------------------------

    config = load_config(CONFIG_PATH)

    # -------------------------------------------------------------
    # Build data module.
    # -------------------------------------------------------------

    data_module = build_datamodule(config)

    data_module.setup()

    validation_loader = data_module.val_dataloader()

    # -------------------------------------------------------------
    # Get the exact validation transform used by the dataset.
    #
    # This must be the same transform object that generated the
    # metadata used for inversion.
    # -------------------------------------------------------------

    if data_module.val_dataset is None:
        raise RuntimeError("Validation dataset was not initialized.")

    validation_transforms = data_module.val_dataset.transform

    if validation_transforms is None:
        raise RuntimeError(
            "Validation dataset does not have a transform. "
            "Cannot invert spatial preprocessing."
        )

    # -------------------------------------------------------------
    # Verify validation dataset.
    # -------------------------------------------------------------

    if len(data_module.val_dataset.data) == 0:
        raise RuntimeError("Validation dataset is empty.")

    print()
    print(f"Validation subjects: {len(data_module.val_dataset.data)}")

    # -------------------------------------------------------------
    # Build model.
    # -------------------------------------------------------------

    model = build_model(
        config=config,
        device=device,
    )

    # -------------------------------------------------------------
    # Load selected checkpoint.
    # -------------------------------------------------------------

    print()
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    checkpoint_dice = checkpoint.get("val_mean_dice")

    print(f"Checkpoint epoch: {checkpoint_epoch}")

    if checkpoint_dice is not None:
        print(f"Checkpoint validation mean Dice: {float(checkpoint_dice):.4f}")

    # -------------------------------------------------------------
    # Verify selected checkpoint.
    #
    # Do not silently visualize a different checkpoint.
    # -------------------------------------------------------------

    if checkpoint_epoch != EXPECTED_BEST_EPOCH:
        raise RuntimeError(
            "The checkpoint is not the selected Experiment 1 "
            f"checkpoint. Expected epoch {EXPECTED_BEST_EPOCH}, "
            f"but found epoch {checkpoint_epoch}."
        )

    if checkpoint_dice is None:
        raise RuntimeError(
            "Checkpoint does not contain 'val_mean_dice'. "
            "Cannot verify that this is the selected checkpoint."
        )

    if not np.isclose(
        float(checkpoint_dice),
        EXPECTED_BEST_DICE,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Checkpoint validation Dice does not match the "
            "selected Experiment 1 model. "
            f"Expected {EXPECTED_BEST_DICE:.4f}, "
            f"found {float(checkpoint_dice):.4f}."
        )

    print()
    print("Selected model verified:")
    print(f"  Experiment:          {EXPECTED_EXPERIMENT}")
    print("  Architecture:        Residual 3D U-Net")
    print(f"  Best epoch:          {EXPECTED_BEST_EPOCH}")
    print(f"  Mean foreground Dice: {EXPECTED_BEST_DICE:.4f}")
    print("  Preprocessing:       native spacing, no N4")

    # -------------------------------------------------------------
    # Inference configuration.
    # -------------------------------------------------------------

    inference_config = config.get(
        "inference",
        {},
    )

    roi_size = tuple(
        int(value)
        for value in inference_config.get(
            "roi_size",
            config["data"].get(
                "patch_size",
                [96, 96, 96],
            ),
        )
    )

    sw_batch_size = int(
        inference_config.get(
            "sw_batch_size",
            1,
        )
    )

    overlap = float(
        inference_config.get(
            "overlap",
            0.25,
        )
    )

    # -------------------------------------------------------------
    # Generate results.
    # -------------------------------------------------------------

    generate_visualizations(
        model=model,
        dataloader=validation_loader,
        device=device,
        prediction_dir=PREDICTION_DIR,
        visualization_dir=VISUALIZATION_DIR,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        validation_transforms=validation_transforms,
    )


if __name__ == "__main__":
    main()
