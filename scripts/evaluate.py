"""
Evaluate a trained IBSR-18 3D U-Net.

This script:
    - loads the best model checkpoint
    - runs sliding-window inference on the validation set
    - optionally applies left-right flip test-time augmentation (TTA)
    - computes Dice for each segmentation class
    - reports per-subject Dice scores using actual IBSR subject IDs
    - reports mean Dice for each class
    - reports mean Dice across foreground classes

IBSR-18 label mapping:
    0 = Background
    1 = CSF
    2 = GM
    3 = WM

The test set is intentionally not used here because it does not
contain ground-truth segmentation labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from ibsr_unet.data.datamodule import IBSRDataModule
from ibsr_unet.evaluation.metrics import dice_per_class
from ibsr_unet.inference import (
    left_right_flip_tta,
    sliding_window_predict,
)
from ibsr_unet.models.unet import build_unet

# ---------------------------------------------------------------------
# IBSR-18 segmentation class names.
#
# The label values are defined by the dataset:
#
#     0 = Background
#     1 = CSF
#     2 = GM
#     3 = WM
# ---------------------------------------------------------------------
CLASS_NAMES = {
    0: "Background",
    1: "CSF",
    2: "GM",
    3: "WM",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a trained IBSR-18 3D U-Net.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best_model.pt"),
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help=(
            "Enable left-right flip test-time augmentation "
            "and average original/flipped logits."
        ),
    )

    return parser.parse_args()


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load configuration from YAML."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def get_device() -> torch.device:
    """Select CUDA when available."""

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("CUDA is not available. Using CPU.")

    return torch.device("cpu")


def get_subject_id(
    batch: dict[str, Any],
) -> str:
    """
    Extract the IBSR subject ID from a DataLoader batch.

    The preferred source is an explicit ``subject_id`` field.

    If that field is not present, the subject ID is recovered from
    the image metadata stored by MONAI. For example:

        data/processed/n4/IBSR_11/IBSR_11.nii.gz

    produces:

        IBSR_11

    This fallback is necessary because MONAI's DataLoader collation
    does not always preserve custom metadata fields such as
    ``subject_id`` at the top level of the batch.
    """

    # -------------------------------------------------------------
    # Preferred: explicit subject_id field.
    # -------------------------------------------------------------
    if "subject_id" in batch:
        subject_id = batch["subject_id"]

        if isinstance(subject_id, (list, tuple)):
            if len(subject_id) == 0:
                raise ValueError("Batch contains an empty subject_id.")

            subject_id = subject_id[0]

        if isinstance(subject_id, torch.Tensor):
            if subject_id.numel() != 1:
                raise ValueError(
                    "Expected subject_id tensor to contain exactly one value."
                )

            subject_id = subject_id.item()

        return str(subject_id)

    # -------------------------------------------------------------
    # Fallback: recover subject ID from MONAI image metadata.
    # -------------------------------------------------------------
    if "image" in batch:
        image = batch["image"]

        if hasattr(image, "meta"):
            filename = image.meta.get("filename_or_obj")

            if isinstance(filename, (list, tuple)):
                if len(filename) == 0:
                    filename = None
                else:
                    filename = filename[0]

            if filename is not None:
                filename_path = Path(str(filename))

                # Expected structure:
                #
                #   .../<subject>/<subject>.nii.gz
                #
                # Therefore the parent directory is the
                # IBSR subject identifier.
                subject_id = filename_path.parent.name

                if subject_id.startswith("IBSR_"):
                    return subject_id

    raise KeyError(
        "Could not determine subject ID. "
        "Expected either 'subject_id' in the batch or "
        "image metadata containing 'filename_or_obj'."
    )


def build_datamodule(
    config: dict[str, Any],
) -> IBSRDataModule:
    """
    Build the IBSR-18 DataModule.

    The evaluation DataModule uses the same image preprocessing
    configuration as training. This is especially important for
    Experiment 4, where validation images must come from the
    precomputed N4 directory.
    """

    data_config = config["data"]
    training_config = config["training"]

    spacing = data_config.get("spacing")

    target_spacing = (
        tuple(float(value) for value in spacing) if spacing is not None else None
    )

    patch_size = tuple(int(value) for value in data_config["patch_size"])

    # -------------------------------------------------------------
    # N4 configuration.
    #
    # Experiment 4:
    #
    #     use_precomputed_n4 = True
    #     use_n4_bias_correction = False
    #
    # This ensures that N4 is applied exactly once.
    # -------------------------------------------------------------
    use_n4_bias_correction = bool(
        data_config.get(
            "use_n4_bias_correction",
            False,
        )
    )

    use_precomputed_n4 = bool(
        data_config.get(
            "use_precomputed_n4",
            False,
        )
    )

    n4_dir_value = data_config.get("n4_dir")

    n4_dir = Path(n4_dir_value) if n4_dir_value is not None else None

    if use_precomputed_n4 and n4_dir is None:
        raise ValueError("data.n4_dir must be provided when use_precomputed_n4=True.")

    return IBSRDataModule(
        data_dir=Path(data_config["root_dir"]),
        splits_dir=Path(data_config["splits_dir"]),
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
        batch_size=1,
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


def load_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Build the model and load the trained checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    num_classes: int,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    use_tta: bool = False,
) -> torch.Tensor:
    """
    Evaluate the model and return mean per-class Dice scores.

    Dice is computed independently for each validation subject and
    then averaged subject-wise.

    When ``use_tta`` is enabled, the model is evaluated using
    conservative left-right flip test-time augmentation. The
    original and flipped-back raw logits are averaged before Dice
    computation.

    Parameters
    ----------
    model:
        Trained segmentation model.

    dataloader:
        Validation DataLoader.

    device:
        Torch device used for inference.

    num_classes:
        Number of segmentation classes.

    roi_size:
        Sliding-window inference ROI.

    sw_batch_size:
        Number of sliding-window patches processed together.

    overlap:
        Sliding-window overlap.

    use_tta:
        Whether to enable left-right flip TTA.

    Returns
    -------
    torch.Tensor
        Tensor containing one Dice score per class.

        For IBSR-18:

            index 0 = Background
            index 1 = CSF
            index 2 = GM
            index 3 = WM
    """

    class_scores: list[torch.Tensor] = []

    for batch in dataloader:
        # -------------------------------------------------------------
        # Retrieve the actual IBSR subject identifier.
        # -------------------------------------------------------------
        subject_id = get_subject_id(batch)

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        # -------------------------------------------------------------
        # Inference.
        #
        # Standard evaluation:
        #
        #     one sliding-window prediction
        #
        # TTA evaluation:
        #
        #     original prediction
        #              +
        #     left-right flipped prediction
        #              ↓
        #     flip augmented prediction back
        #              ↓
        #     average raw logits
        #
        # dice_per_class() performs argmax internally, so raw logits
        # are passed directly to the metric function.
        # -------------------------------------------------------------
        if use_tta:
            predictions = left_right_flip_tta(
                model=model,
                images=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )
        else:
            predictions = sliding_window_predict(
                model=model,
                images=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )

        # -------------------------------------------------------------
        # Compute Dice independently for each class.
        # -------------------------------------------------------------
        scores = dice_per_class(
            prediction=predictions,
            target=labels,
            num_classes=num_classes,
            include_background=True,
        )

        class_scores.append(scores.cpu())

        # -------------------------------------------------------------
        # Print per-subject results using the actual IBSR subject ID
        # and semantic class names.
        # -------------------------------------------------------------
        formatted_scores = []

        for class_index, score in enumerate(scores):
            class_name = CLASS_NAMES.get(
                class_index,
                f"Class {class_index}",
            )

            formatted_scores.append(f"{class_name}={score.item():.4f}")

        print(f"{subject_id}: " + " | ".join(formatted_scores))

    if not class_scores:
        raise RuntimeError("Validation DataLoader produced no batches.")

    # Stack subject-level scores:
    #
    #     [num_subjects, num_classes]
    #
    # and average across subjects.
    return torch.stack(class_scores).mean(dim=0)


def main() -> None:
    """Run model evaluation."""

    args = parse_args()

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------
    config = load_config(args.config)

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------
    device = get_device()

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------
    datamodule = build_datamodule(config)

    datamodule.setup()

    # -------------------------------------------------------------
    # Print preprocessing configuration so that it is explicit
    # which image source is being evaluated.
    # -------------------------------------------------------------
    data_summary = datamodule.summary()

    print("\nEvaluation data configuration:")
    print(f"  Precomputed N4: {data_summary['use_precomputed_n4']}")
    print(f"  Runtime N4:     {data_summary['use_n4_bias_correction']}")

    if data_summary["use_precomputed_n4"]:
        print(f"  N4 directory:   {data_summary['n4_dir']}")

    print(f"  Target spacing: {data_summary['target_spacing']}")

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model = load_model(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # -------------------------------------------------------------
    # Inference configuration
    # -------------------------------------------------------------
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
            0.25,
        )
    )

    num_classes = int(config["model"]["out_channels"])

    # -------------------------------------------------------------
    # Print inference configuration.
    # -------------------------------------------------------------
    print("\nInference configuration:")
    print(f"  ROI size:        {roi_size}")
    print(f"  SW batch size:   {sw_batch_size}")
    print(f"  SW overlap:      {overlap:.2f}")
    print(f"  TTA:             {'left-right flip' if args.tta else 'disabled'}")

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------
    if args.tta:
        print("\nRunning validation evaluation with left-right flip TTA...")
    else:
        print("\nRunning validation evaluation...")

    print("-" * 60)

    mean_class_dice = evaluate(
        model=model,
        dataloader=datamodule.val_dataloader(),
        device=device,
        num_classes=num_classes,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        use_tta=args.tta,
    )

    # -------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------
    print("\n" + "=" * 60)

    if args.tta:
        print("Validation Dice Results — Left-Right Flip TTA")
    else:
        print("Validation Dice Results")

    print("=" * 60)

    for class_index, score in enumerate(mean_class_dice):
        class_name = CLASS_NAMES.get(
            class_index,
            f"Class {class_index}",
        )

        print(f"{class_name:<10}: {score.item():.4f}")

    # -------------------------------------------------------------
    # Mean foreground Dice.
    #
    # Background is excluded because it occupies the majority of
    # the image and would artificially inflate the overall score.
    #
    # Foreground classes:
    #     CSF
    #     GM
    #     WM
    # -------------------------------------------------------------
    foreground_dice = mean_class_dice[1:].mean()

    print("-" * 60)

    print(f"Mean foreground Dice: {foreground_dice.item():.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
