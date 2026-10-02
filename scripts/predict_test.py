"""
Run unlabeled test-set inference for the selected IBSR-18 model.

This script:
    - loads the selected model checkpoint
    - uses the same configuration and preprocessing as evaluation
    - runs sliding-window inference on the IBSR-18 test set
    - restores predictions to native image space
    - saves predictions as native-space NIfTI files

IBSR-18 test subjects:
    IBSR_02
    IBSR_10
    IBSR_15

The test set does not contain ground-truth segmentation labels,
so no quantitative performance metrics are computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

# Reuse the validated configuration, DataModule, and checkpoint
# loading logic from the existing evaluation pipeline.
from evaluate import (
    build_datamodule,
    load_config,
    load_model,
)

from ibsr_unet.inference import sliding_window_predict

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions" / "test"


# ---------------------------------------------------------------------
# IBSR-18 test subjects.
# ---------------------------------------------------------------------

TEST_SUBJECTS = {
    "IBSR_02",
    "IBSR_10",
    "IBSR_15",
}


def get_device() -> torch.device:
    """Select CUDA when available."""

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("CUDA is not available. Using CPU.")

    return torch.device("cpu")


def get_subject_info(
    dataset: Any,
    index: int,
) -> tuple[str, Path]:
    """
    Recover the subject ID and original MRI path.

    The MONAI Dataset stores the original data dictionaries in
    ``dataset.data``.
    """

    if not hasattr(dataset, "data"):
        raise AttributeError(
            "Test dataset does not expose the expected 'data' attribute."
        )

    data_item = dataset.data[index]

    image_path = data_item.get("image")

    if image_path is None:
        raise KeyError(f"Test dataset item {index} does not contain an image path.")

    image_path = Path(str(image_path))

    subject_id = image_path.parent.name

    if not subject_id.startswith("IBSR_"):
        raise ValueError(
            f"Could not determine IBSR subject ID from image path: {image_path}"
        )

    return subject_id, image_path


# ---------------------------------------------------------------------
# Metadata diagnostics
# ---------------------------------------------------------------------


def print_image_metadata(
    batch: dict[str, Any],
) -> None:
    """
    Print MONAI image metadata relevant to spatial restoration.

    This is intentionally diagnostic. Different MONAI versions can
    store CropForeground information differently.
    """

    image = batch["image"]

    print()
    print("  Image metadata diagnostics:")

    if not hasattr(image, "meta"):
        print("    Image does not expose MetaTensor metadata.")
        return

    meta = image.meta

    for key, value in meta.items():
        # Crop-related metadata is especially important.
        key_lower = str(key).lower()

        if (
            "crop" in key_lower
            or "foreground" in key_lower
            or "spatial" in key_lower
            or "original" in key_lower
            or "filename" in key_lower
            or "affine" in key_lower
        ):
            print(f"    {key}: {value}")

    # MONAI may keep transform history separately.
    if hasattr(image, "applied_operations"):
        operations = image.applied_operations

        print(f"    applied_operations count: {len(operations)}")

        for operation_index, operation in enumerate(operations):
            if not isinstance(
                operation,
                dict,
            ):
                print(f"    operation[{operation_index}]: {operation}")
                continue

            class_name = operation.get(
                "class",
                operation.get(
                    "class_name",
                    "<unknown>",
                ),
            )

            print(f"    operation[{operation_index}] class: {class_name}")

            # CropForegroundd information is normally stored
            # somewhere inside the operation dictionary.
            operation_text = repr(operation).lower()

            if (
                "crop" in operation_text
                or "foreground" in operation_text
                or "roi" in operation_text
                or "spatial" in operation_text
            ):
                print(f"      details: {operation}")


def get_crop_coordinates_from_metadata(
    batch: dict[str, Any],
) -> (
    tuple[
        tuple[int, int, int],
        tuple[int, int, int],
    ]
    | None
):
    """
    Recover the exact CropForeground coordinates recorded by MONAI.

    For the MONAI version used in this project, CropForeground is
    stored approximately as:

        {
            "class": "CropForeground",
            "orig_size": (X, Y, Z),
            "extra_info": {
                "cropped": [
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    z_start,
                    z_end,
                ],
                ...
            },
        }

    The values in ``cropped`` specify how many voxels were removed
    from the beginning and end of each spatial dimension.

    Returns
    -------
    tuple or None
        ``(start, end)`` coordinates of the retained foreground
        region in native spatial coordinates.
    """

    image = batch["image"]

    if not hasattr(
        image,
        "applied_operations",
    ):
        return None

    for operation in image.applied_operations:
        if not isinstance(
            operation,
            dict,
        ):
            continue

        operation_class = operation.get(
            "class",
            operation.get(
                "class_name",
                "",
            ),
        )

        if operation_class != "CropForeground":
            continue

        # ---------------------------------------------------------
        # Original spatial dimensions before CropForeground.
        # ---------------------------------------------------------

        orig_size = operation.get("orig_size")

        if orig_size is None:
            return None

        try:
            orig_size = tuple(int(value) for value in orig_size)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if len(orig_size) != 3:
            return None

        # ---------------------------------------------------------
        # Crop information is stored inside extra_info.
        # ---------------------------------------------------------

        extra_info = operation.get("extra_info")

        if not isinstance(
            extra_info,
            dict,
        ):
            return None

        cropped = extra_info.get("cropped")

        if cropped is None:
            return None

        try:
            cropped = tuple(int(value) for value in cropped)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if len(cropped) != 6:
            return None

        # ---------------------------------------------------------
        # MONAI ordering:
        #
        # cropped = [
        #     x_start,
        #     x_end,
        #     y_start,
        #     y_end,
        #     z_start,
        #     z_end,
        # ]
        #
        # These are amounts removed from each side.
        # ---------------------------------------------------------

        start = (
            cropped[0],
            cropped[2],
            cropped[4],
        )

        end = (
            orig_size[0] - cropped[1],
            orig_size[1] - cropped[3],
            orig_size[2] - cropped[5],
        )

        return start, end

    return None


# ---------------------------------------------------------------------
# Native-space restoration
# ---------------------------------------------------------------------


def invert_prediction_to_native_space(
    prediction: torch.Tensor,
    batch: dict[str, Any],
    reference_path: Path,
) -> torch.Tensor:
    """
    Restore a foreground-cropped prediction to native MRI space.

    The selected Experiment 1 preprocessing is:

        1. Load image
        2. Ensure channel-first
        3. Orient to RAS
        4. Native spacing
        5. Normalize intensity
        6. CropForegroundd(source_key="image")

    The function first attempts to recover the exact crop
    coordinates recorded by MONAI.

    If MONAI does not expose those coordinates, the function raises
    a detailed diagnostic error rather than silently producing a
    potentially misaligned NIfTI.

    Parameters
    ----------
    prediction:
        Model prediction with shape:

            (B, 1, X_crop, Y_crop, Z_crop)

    batch:
        Transformed MONAI batch.

    reference_path:
        Original native-space MRI path.

    Returns
    -------
    torch.Tensor
        Native-space prediction with shape:

            (B, X_native, Y_native, Z_native)
    """

    if prediction.ndim != 5:
        raise ValueError(
            "Expected prediction with shape "
            "(B, 1, X, Y, Z), but received "
            f"{tuple(prediction.shape)}."
        )

    if prediction.shape[1] != 1:
        raise ValueError(
            "Expected exactly one prediction channel, "
            f"but received shape {tuple(prediction.shape)}."
        )

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference MRI not found: {reference_path}")

    # -------------------------------------------------------------
    # Load native MRI.
    # -------------------------------------------------------------

    reference = nib.load(str(reference_path))

    native_image = np.asarray(
        reference.dataobj,
        dtype=np.float32,
    )

    if native_image.ndim == 4 and native_image.shape[-1] == 1:
        native_image = native_image[..., 0]

    if native_image.ndim != 3:
        raise ValueError(
            "Expected native MRI to have three spatial "
            f"dimensions, but received "
            f"shape {native_image.shape}."
        )

    native_shape = native_image.shape

    # -------------------------------------------------------------
    # Recover MONAI's actual foreground crop coordinates.
    # -------------------------------------------------------------

    crop_coordinates = get_crop_coordinates_from_metadata(batch)

    if crop_coordinates is None:
        print_image_metadata(batch)

        raise ValueError(
            "Could not recover the exact CropForegroundd "
            "coordinates from MONAI metadata or applied "
            "operations.\n\n"
            f"  Prediction spatial shape: "
            f"{tuple(prediction.shape[2:])}\n"
            f"  Native MRI shape: "
            f"{native_shape}\n"
            f"  Reference MRI: "
            f"{reference_path}\n\n"
            "The metadata diagnostics above show the transform "
            "information available from this MONAI version."
        )

    start, end = crop_coordinates

    # -------------------------------------------------------------
    # Validate coordinates.
    # -------------------------------------------------------------

    if len(start) != 3 or len(end) != 3:
        raise ValueError(
            "Recovered crop coordinates are not 3-dimensional.\n"
            f"  start={start}\n"
            f"  end={end}"
        )

    for axis in range(3):
        if start[axis] < 0:
            raise ValueError(f"Invalid crop start coordinate: {start}")

        if end[axis] > native_shape[axis]:
            raise ValueError(
                "Crop end coordinate exceeds native image "
                "dimensions.\n"
                f"  end={end}\n"
                f"  native_shape={native_shape}"
            )

        if end[axis] <= start[axis]:
            raise ValueError(f"Invalid crop interval.\n  start={start}\n  end={end}")

    expected_crop_shape = tuple(end[axis] - start[axis] for axis in range(3))

    actual_prediction_shape = tuple(prediction.shape[2:])

    print(f"  MONAI crop start: {start}")

    print(f"  MONAI crop end:   {end}")

    print(f"  Crop shape:       {expected_crop_shape}")

    # -------------------------------------------------------------
    # Critical geometry check.
    # -------------------------------------------------------------

    if actual_prediction_shape != expected_crop_shape:
        raise ValueError(
            "Prediction/MONAI crop geometry mismatch.\n"
            f"  Prediction spatial shape: "
            f"{actual_prediction_shape}\n"
            f"  MONAI crop shape: "
            f"{expected_crop_shape}\n"
            f"  Native MRI shape: "
            f"{native_shape}\n"
            f"  Crop start: "
            f"{start}\n"
            f"  Crop end:   "
            f"{end}\n"
            f"  MRI: "
            f"{reference_path}"
        )

    # -------------------------------------------------------------
    # Restore prediction to native space.
    # -------------------------------------------------------------

    restored_predictions = []

    for batch_index in range(prediction.shape[0]):
        cropped_prediction = prediction[
            batch_index,
            0,
        ]

        restored = torch.zeros(
            native_shape,
            dtype=torch.uint8,
            device=cropped_prediction.device,
        )

        restored[
            start[0] : end[0],
            start[1] : end[1],
            start[2] : end[2],
        ] = torch.round(cropped_prediction).to(dtype=torch.uint8)

        restored_predictions.append(restored)

    return torch.stack(
        restored_predictions,
        dim=0,
    )


# ---------------------------------------------------------------------
# NIfTI saving
# ---------------------------------------------------------------------


def save_prediction_nifti(
    prediction: np.ndarray,
    reference_path: Path,
    output_path: Path,
) -> None:
    """
    Save a prediction using the native MRI spatial metadata.

    The prediction is stored as:

        (X, Y, Z)

    with uint8 segmentation labels.
    """

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference MRI not found: {reference_path}")

    reference = nib.load(str(reference_path))

    prediction = np.asarray(
        prediction,
        dtype=np.uint8,
    )

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

    if prediction.shape != reference_spatial_shape:
        raise ValueError(
            "Prediction/reference spatial shape "
            "mismatch: "
            f"prediction={prediction.shape}, "
            f"reference_spatial="
            f"{reference_spatial_shape}, "
            f"reference_raw={reference_shape}"
        )

    # Use native MRI affine and header.
    header = reference.header.copy()

    header.set_data_dtype(np.uint8)

    header.set_data_shape(prediction.shape)

    prediction_image = nib.Nifti1Image(
        prediction,
        affine=reference.affine,
        header=header,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        prediction_image,
        str(output_path),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Run unlabeled test-set inference."""

    config_path = DEFAULT_CONFIG
    checkpoint_path = DEFAULT_CHECKPOINT

    print("=" * 60)
    print("IBSR-18 TEST-SET INFERENCE")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    config = load_config(config_path)

    print(f"Configuration: {config_path}")

    print(f"Checkpoint:    {checkpoint_path}")

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = get_device()

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    datamodule = build_datamodule(config)

    datamodule.setup()

    if datamodule.test_dataset is None:
        raise RuntimeError("Test dataset was not initialized.")

    test_dataset = datamodule.test_dataset

    # -----------------------------------------------------------------
    # Preprocessing summary
    # -----------------------------------------------------------------

    data_summary = datamodule.summary()

    print()
    print("Inference data configuration:")

    print(f"  Precomputed N4: {data_summary['use_precomputed_n4']}")

    print(f"  Runtime N4:     {data_summary['use_n4_bias_correction']}")

    print(f"  Target spacing: {data_summary['target_spacing']}")

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = load_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # -----------------------------------------------------------------
    # Inference configuration
    # -----------------------------------------------------------------

    inference_config = config.get(
        "inference",
        {},
    )

    roi_size = tuple(
        int(value)
        for value in inference_config.get(
            "roi_size",
            config["data"]["patch_size"],
        )
    )

    if len(roi_size) != 3:
        raise ValueError("inference.roi_size must contain exactly three values.")

    sw_batch_size = int(
        inference_config.get(
            "sw_batch_size",
            1,
        )
    )

    overlap = float(
        inference_config.get(
            "overlap",
            0.50,
        )
    )

    print()
    print("Inference configuration:")

    print(f"  ROI size:       {roi_size}")

    print(f"  SW batch size:  {sw_batch_size}")

    print(f"  Overlap:        {overlap}")

    # -----------------------------------------------------------------
    # Test subjects
    # -----------------------------------------------------------------

    print()
    print("Test subjects:")

    for subject_id in sorted(TEST_SUBJECTS):
        print(f"  - {subject_id}")

    print("=" * 60)

    # -----------------------------------------------------------------
    # Test inference
    # -----------------------------------------------------------------

    print()
    print("Running unlabeled test inference...")
    print("-" * 60)

    processed_subjects = 0

    test_dataloader = datamodule.test_dataloader()

    with torch.inference_mode():
        for index, batch in enumerate(test_dataloader):
            subject_id, reference_path = get_subject_info(
                test_dataset,
                index,
            )

            # ---------------------------------------------------------
            # Safety check.
            # ---------------------------------------------------------

            if subject_id not in TEST_SUBJECTS:
                continue

            print(f"{subject_id}...")

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            # ---------------------------------------------------------
            # Sliding-window inference.
            # ---------------------------------------------------------

            predictions = sliding_window_predict(
                model=model,
                images=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )

            # ---------------------------------------------------------
            # Convert logits to discrete segmentation labels.
            # ---------------------------------------------------------

            predictions = torch.argmax(
                predictions,
                dim=1,
                keepdim=True,
            )

            print(f"  Transformed prediction shape: {tuple(predictions.shape)}")

            # ---------------------------------------------------------
            # Restore prediction to native MRI space.
            # ---------------------------------------------------------

            native_prediction = invert_prediction_to_native_space(
                prediction=predictions,
                batch=batch,
                reference_path=reference_path,
            )

            prediction_np = native_prediction[0].cpu().numpy().astype(np.uint8)

            print(f"  Native prediction shape: {prediction_np.shape}")

            # ---------------------------------------------------------
            # Report predicted labels.
            # ---------------------------------------------------------

            labels = np.unique(prediction_np)

            print(f"  Prediction labels: {labels.tolist()}")

            # ---------------------------------------------------------
            # Save native-space NIfTI.
            # ---------------------------------------------------------

            output_path = OUTPUT_DIR / subject_id / f"{subject_id}_pred.nii.gz"

            save_prediction_nifti(
                prediction=prediction_np,
                reference_path=reference_path,
                output_path=output_path,
            )

            print(f"  Reference MRI: {reference_path}")

            print(f"  Saved prediction: {output_path}")

            processed_subjects += 1

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print()
    print("=" * 60)
    print("Test inference complete.")

    print(f"Processed subjects: {processed_subjects}/{len(TEST_SUBJECTS)}")

    print(f"Output directory: {OUTPUT_DIR}")

    print()

    print(
        "Note: The IBSR test set has no ground-truth "
        "segmentations in this project, so these predictions "
        "are not accompanied by quantitative Dice evaluation."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
