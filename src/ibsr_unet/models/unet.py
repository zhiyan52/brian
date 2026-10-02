"""
3D U-Net model for IBSR-18 brain tissue segmentation.

The baseline model uses MONAI's configurable 3D U-Net implementation.
The network predicts four classes:

    0: background
    1: tissue class
    2: tissue class
    3: tissue class

The exact semantic mapping of labels 1-3 should be documented once
verified against the IBSR-18 dataset documentation.
"""

from __future__ import annotations

from collections.abc import Sequence

from monai.networks.nets import UNet
from torch import nn


def build_unet(
    in_channels: int = 1,
    out_channels: int = 4,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
) -> nn.Module:
    """
    Build a 3D U-Net for multi-class brain tissue segmentation.

    Parameters
    ----------
    in_channels:
        Number of input image channels. IBSR-18 uses one T1-weighted
        MRI volume, so the default is 1.

    out_channels:
        Number of output segmentation classes. IBSR-18 contains
        background plus three tissue labels, so the default is 4.

    channels:
        Number of feature channels at each U-Net level.

    strides:
        Downsampling stride at each encoder level.

    num_res_units:
        Number of residual convolutional units in each block.
        Setting this to 2 provides a stronger baseline than a
        single-convolution block while remaining computationally
        manageable.

    Returns
    -------
    nn.Module
        Configured MONAI 3D U-Net.
    """

    if len(strides) != len(channels) - 1:
        raise ValueError(
            "The number of strides must equal len(channels) - 1. "
            f"Received {len(strides)} strides and {len(channels)} channels."
        )

    if in_channels < 1:
        raise ValueError("in_channels must be >= 1.")

    if out_channels < 2:
        raise ValueError("out_channels must be >= 2.")

    if num_res_units < 0:
        raise ValueError("num_res_units must be >= 0.")

    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=num_res_units,
    )
