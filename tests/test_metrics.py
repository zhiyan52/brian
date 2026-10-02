"""Tests for segmentation metrics."""

import torch

from ibsr_unet.evaluation import (
    dice_per_class,
    mean_dice,
)


def test_perfect_segmentation() -> None:
    """Perfect predictions should produce Dice = 1."""

    target = torch.zeros(
        1,
        4,
        4,
        4,
        dtype=torch.long,
    )

    target[:, :2] = 1
    target[:, 2:] = 2

    prediction = torch.full(
        (1, 3, 4, 4, 4),
        -10.0,
    )

    prediction[:, 0][target == 0] = 10.0
    prediction[:, 1][target == 1] = 10.0
    prediction[:, 2][target == 2] = 10.0

    scores = dice_per_class(
        prediction=prediction,
        target=target,
        num_classes=3,
        include_background=True,
    )

    assert torch.allclose(
        scores,
        torch.ones(3),
    )


def test_mean_dice_excludes_background() -> None:
    """Mean Dice should use tissue classes when background is excluded."""

    target = torch.zeros(
        1,
        4,
        4,
        4,
        dtype=torch.long,
    )

    target[:, :2] = 1
    target[:, 2:] = 2

    prediction = torch.full(
        (1, 3, 4, 4, 4),
        -10.0,
    )

    prediction[:, 0][target == 0] = 10.0
    prediction[:, 1][target == 1] = 10.0
    prediction[:, 2][target == 2] = 10.0

    score = mean_dice(
        prediction=prediction,
        target=target,
        num_classes=3,
        include_background=False,
    )

    assert torch.isclose(
        score,
        torch.tensor(1.0),
    )


def test_metric_accepts_channel_target() -> None:
    """Metrics should accept targets with a singleton channel."""

    prediction = torch.randn(
        1,
        4,
        8,
        8,
        8,
    )

    target = torch.randint(
        0,
        4,
        (1, 1, 8, 8, 8),
    )

    score = mean_dice(
        prediction=prediction,
        target=target,
        num_classes=4,
    )

    assert score.ndim == 0
    assert torch.isfinite(score)


def test_metric_values_are_bounded() -> None:
    """Dice scores should remain within [0, 1]."""

    prediction = torch.randn(
        2,
        4,
        8,
        8,
        8,
    )

    target = torch.randint(
        0,
        4,
        (2, 8, 8, 8),
    )

    scores = dice_per_class(
        prediction=prediction,
        target=target,
        num_classes=4,
    )

    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)
