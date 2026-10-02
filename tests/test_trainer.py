"""Tests for the training engine."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from ibsr_unet.evaluation.metrics import mean_dice
from ibsr_unet.training import Trainer


class TinySegmentationModel(nn.Module):
    """Small model used only for testing the training loop."""

    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinySegmentationLoss(nn.Module):
    """Simple loss used to test the trainer independently."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return prediction.mean()


def create_test_loader() -> DataLoader:
    """Create a minimal segmentation DataLoader."""

    images = torch.randn(
        2,
        1,
        16,
        16,
        16,
    )

    labels = torch.randint(
        0,
        4,
        (2, 1, 16, 16, 16),
    )

    dataset = TensorDataset(
        images,
        labels,
    )

    # The real DataModule returns dictionaries.
    return DataLoader(
        [
            {
                "image": image,
                "label": label,
            }
            for image, label in dataset
        ],
        batch_size=1,
    )


def test_trainer_fit(tmp_path: Path) -> None:
    """Trainer should complete training and save a checkpoint."""

    model = TinySegmentationModel()

    loss_fn = TinySegmentationLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        num_epochs=2,
        checkpoint_dir=tmp_path,
        metric_fn=lambda prediction, target: mean_dice(
            prediction,
            target,
            num_classes=4,
            include_background=False,
        ),
        roi_size=(16, 16, 16),
        sw_batch_size=1,
        overlap=0.25,
    )

    train_loader = create_test_loader()
    val_loader = create_test_loader()

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    assert "train_loss" in history
    assert "val_loss" in history
    assert "val_mean_dice" in history

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_mean_dice"]) == 2

    assert trainer.best_val_dice > float("-inf")

    assert (tmp_path / "best_model.pt").exists()
