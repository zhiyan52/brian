"""
Train a 3D U-Net on the IBSR-18 brain tissue segmentation dataset.

This script is the main training entry point for the project.

Pipeline
--------
1. Load configuration from YAML.
2. Set random seeds.
3. Select the training device.
4. Create the IBSR-18 DataModule.
5. Build the 3D U-Net.
6. Create the Dice + Cross-Entropy loss.
7. Create the AdamW optimizer.
8. Configure per-class Dice validation.
9. Train the model.

The data preprocessing pipeline, including optional N4 bias-field
correction, is controlled through the YAML configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW

from ibsr_unet.data.datamodule import IBSRDataModule
from ibsr_unet.evaluation.metrics import dice_per_class
from ibsr_unet.models.unet import build_unet
from ibsr_unet.training.losses import DiceCrossEntropyLoss
from ibsr_unet.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train a 3D U-Net on IBSR-18.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the training configuration YAML file.",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load the training configuration from YAML.

    Parameters
    ----------
    config_path:
        Path to the training configuration file.

    Returns
    -------
    dict
        Parsed configuration.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must contain a YAML mapping.")

    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        return device

    print("CUDA is not available. Using CPU.")

    return torch.device("cpu")


def build_datamodule(
    config: dict[str, Any],
) -> IBSRDataModule:
    """Build the IBSR-18 DataModule from configuration."""

    data_config = config["data"]
    training_config = config["training"]

    # ---------------------------------------------------------------
    # Target spacing
    # ---------------------------------------------------------------
    spacing = data_config.get("spacing")

    target_spacing = (
        tuple(float(value) for value in spacing) if spacing is not None else None
    )

    # ---------------------------------------------------------------
    # Training patch size
    # ---------------------------------------------------------------
    patch_size = tuple(int(value) for value in data_config["patch_size"])

    # ---------------------------------------------------------------
    # Optional N4 bias-field correction
    # ---------------------------------------------------------------
    use_n4_bias_correction = bool(
        data_config.get(
            "use_n4_bias_correction",
            False,
        )
    )

    # ---------------------------------------------------------------
    # Precomputed N4 images
    # ---------------------------------------------------------------
    n4_dir = data_config.get("n4_dir")

    n4_dir = Path(n4_dir) if n4_dir is not None else None

    use_precomputed_n4 = bool(
        data_config.get(
            "use_precomputed_n4",
            False,
        )
    )

    return IBSRDataModule(
        data_dir=Path(data_config["root_dir"]),
        splits_dir=Path(data_config["splits_dir"]),
        patch_size=patch_size,
        num_samples=int(training_config.get("num_samples", 1)),
        target_spacing=target_spacing,
        use_n4_bias_correction=use_n4_bias_correction,
        n4_dir=n4_dir,
        use_precomputed_n4=use_precomputed_n4,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(training_config.get("pin_memory", True)),
    )


def build_model(
    config: dict[str, Any],
) -> torch.nn.Module:
    """Build the 3D U-Net from configuration."""

    model_config = config["model"]

    channels = tuple(int(value) for value in model_config["channels"])

    strides = tuple(
        int(value)
        for value in model_config.get(
            "strides",
            [2, 2, 2, 2],
        )
    )

    return build_unet(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        channels=channels,
        strides=strides,
        num_res_units=int(model_config.get("num_res_units", 2)),
    )


def build_loss(
    config: dict[str, Any],
) -> DiceCrossEntropyLoss:
    """Build the Dice + Cross-Entropy loss."""

    loss_config = config["loss"]

    return DiceCrossEntropyLoss(
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        include_background=bool(
            loss_config.get(
                "include_background",
                True,
            )
        ),
        class_weights=loss_config.get("class_weights"),
    )


def build_metric(
    config: dict[str, Any],
):
    """
    Build the validation per-class Dice metric.

    Label mapping
    -------------
    0 = background
    1 = CSF
    2 = GM
    3 = WM
    """

    num_classes = int(config["model"]["out_channels"])

    def metric(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return dice_per_class(
            prediction,
            target,
            num_classes=num_classes,
            include_background=True,
        )

    return metric


def print_configuration(
    config: dict[str, Any],
    device: torch.device,
) -> None:
    """Print the main experiment configuration."""

    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]

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

    print("\n" + "=" * 60)
    print("IBSR-18 3D U-Net Training")
    print("=" * 60)

    print(f"Device:             {device}")
    print(f"Data root:          {data_config['root_dir']}")
    print(f"Patch size:         {data_config['patch_size']}")

    spacing = data_config.get("spacing")

    if spacing is None:
        print("Target spacing:     native")
    else:
        print(f"Target spacing:     {spacing}")

    print(f"N4 bias correction: {use_n4_bias_correction}")

    print(f"Precomputed N4:     {use_precomputed_n4}")

    print(f"N4 directory:       {data_config.get('n4_dir')}")

    print(f"Batch size:         {training_config['batch_size']}")

    print(f"Num samples:        {training_config.get('num_samples', 1)}")

    print(f"Learning rate:      {training_config['learning_rate']}")

    print(f"Weight decay:       {training_config.get('weight_decay', 0.0)}")

    print(f"Epochs:             {training_config['epochs']}")

    print(f"Channels:           {model_config['channels']}")

    print(f"Residual units:     {model_config.get('num_res_units', 2)}")

    print("=" * 60 + "\n")


def main() -> None:
    """Run the IBSR-18 training pipeline."""

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------
    args = parse_args()

    config = load_config(args.config)

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------
    seed = int(config.get("seed", 42))

    set_seed(seed)

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------
    device = get_device()

    print_configuration(
        config=config,
        device=device,
    )

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    datamodule = build_datamodule(config)

    datamodule.setup()

    print("DataModule:")
    print(datamodule.summary())

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = build_model(config)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"\nModel parameters: {num_parameters:,}")

    # ---------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------
    loss_fn = build_loss(config)

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------
    training_config = config["training"]

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(
            training_config.get(
                "weight_decay",
                0.0,
            )
        ),
    )

    # ---------------------------------------------------------------
    # Validation metric
    # ---------------------------------------------------------------
    metric_fn = build_metric(config)

    # ---------------------------------------------------------------
    # Sliding-window inference settings
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Checkpoint directory
    # ---------------------------------------------------------------
    output_config = config.get(
        "output",
        {},
    )

    checkpoint_dir = Path(
        output_config.get(
            "checkpoint_dir",
            "outputs/checkpoints",
        )
    )

    # ---------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=int(training_config["epochs"]),
        checkpoint_dir=checkpoint_dir,
        metric_fn=metric_fn,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------
    history = trainer.fit(
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------
    print("\nTraining complete.")

    if history["val_mean_dice"]:
        best_dice = max(history["val_mean_dice"])

        print(f"Best validation mean Dice: {best_dice:.4f}")

    print(f"Best checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
