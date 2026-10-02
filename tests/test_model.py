"""Tests for the 3D U-Net model."""

import torch

from ibsr_unet.models import build_unet


def test_unet_output_shape() -> None:
    """The model should preserve spatial dimensions."""

    model = build_unet(
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    x = torch.randn(1, 1, 96, 96, 96)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 4, 96, 96, 96)


def test_unet_output_channels() -> None:
    """The model should produce one logit channel per class."""

    model = build_unet(
        in_channels=1,
        out_channels=4,
    )

    x = torch.randn(1, 1, 96, 96, 96)

    with torch.no_grad():
        y = model(x)

    assert y.shape[1] == 4
