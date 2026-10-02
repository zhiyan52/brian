"""Tests for the inference utilities."""

import torch
from torch import nn

from ibsr_unet.inference import sliding_window_predict


class TinySegmentationModel(nn.Module):
    """Small model used to test sliding-window inference."""

    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_sliding_window_predict() -> None:
    """Sliding-window inference should preserve the input volume size."""

    model = TinySegmentationModel()
    model.eval()

    images = torch.randn(
        1,
        1,
        32,
        32,
        32,
    )

    with torch.no_grad():
        predictions = sliding_window_predict(
            model=model,
            images=images,
            roi_size=(16, 16, 16),
            sw_batch_size=1,
            overlap=0.25,
        )

    assert predictions.shape == (
        1,
        4,
        32,
        32,
        32,
    )

    assert predictions.dtype == torch.float32
