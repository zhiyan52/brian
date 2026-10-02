"""
Training and validation engine for IBSR-18 3D brain tissue segmentation.

The Trainer is responsible for:
    - model optimization
    - training/validation loops
    - metric tracking
    - per-class Dice tracking
    - sliding-window validation
    - checkpointing the best model

Evaluation metrics are injected through ``metric_fn`` so that the
training engine remains reusable for other segmentation tasks.

Training uses random patches, while validation uses sliding-window
inference to process complete 3D volumes without requiring the entire
volume to fit into GPU memory at once.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ibsr_unet.inference import sliding_window_predict

MetricFn = Callable[[Tensor, Tensor], Tensor]


class Trainer:
    """Train and validate a segmentation model."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        num_epochs: int = 1,
        checkpoint_dir: Path = Path("outputs/checkpoints"),
        metric_fn: MetricFn | None = None,
        roi_size: tuple[int, int, int] = (96, 96, 96),
        sw_batch_size: int = 1,
        overlap: float = 0.25,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Segmentation model.

        loss_fn:
            Training/validation loss function.

        optimizer:
            PyTorch optimizer.

        device:
            Device used for training and validation.

        num_epochs:
            Number of training epochs.

        checkpoint_dir:
            Directory where model checkpoints are saved.

        metric_fn:
            Optional validation metric function.

            The function should accept ``(prediction, target)`` and
            return either:

                - a scalar tensor, or
                - a tensor containing one metric per class.

            For IBSR-18, the metric function returns per-class Dice
            scores for background, CSF, GM, and WM.

        roi_size:
            Spatial size of each sliding-window inference patch.

        sw_batch_size:
            Number of sliding-window patches processed simultaneously.

            A value of 1 is recommended for GPUs with limited VRAM.

        overlap:
            Fraction of overlap between neighboring inference windows.
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer

        self.device = device
        self.num_epochs = num_epochs

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metric_fn = metric_fn

        # Sliding-window inference settings.
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

        # Dice is maximized, unlike validation loss.
        self.best_val_dice = float("-inf")

        # Used as a fallback if no validation metric is provided.
        self.best_val_loss = float("inf")

        # -------------------------------------------------------------
        # Training history.
        #
        # ``val_per_class_dice`` stores a list for every epoch.
        #
        # For IBSR-18:
        #     index 0 -> background
        #     index 1 -> CSF
        #     index 2 -> GM
        #     index 3 -> WM
        # -------------------------------------------------------------
        self.history: dict[str, list[Any]] = {
            "train_loss": [],
            "val_loss": [],
            "val_mean_dice": [],
            "val_per_class_dice": [],
        }

    def train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """
        Run one training epoch.

        Training operates on randomly sampled 3D patches.

        Returns
        -------
        float
            Mean training loss over the epoch.
        """
        self.model.train()

        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True,
            )

            # Training uses the sampled patches directly.
            predictions = self.model(images)

            loss = self.loss_fn(
                predictions,
                labels,
            )

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Training DataLoader produced no batches.")

        return running_loss / num_batches

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> tuple[float, Tensor | None]:
        """
        Run validation using sliding-window inference.

        Validation volumes are processed using overlapping windows
        instead of passing the complete volume through the network
        at once.

        Returns
        -------
        tuple[float, Tensor | None]
            Validation loss and validation metric.

            If a metric function is provided, the second value
            contains the averaged metric. For IBSR-18 this is a
            vector containing Dice for:

                [background, CSF, GM, WM]

        Notes
        -----
        Metrics are averaged across validation subjects.
        """
        self.model.eval()

        running_loss = 0.0

        # This can hold either a scalar metric or a vector of
        # per-class metrics.
        running_metric: Tensor | None = None

        num_batches = 0

        for batch in val_loader:
            images = batch["image"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            # ---------------------------------------------------------
            # Sliding-window inference.
            #
            # This allows the complete validation volume to be
            # evaluated without requiring the entire volume to fit
            # into GPU memory.
            # ---------------------------------------------------------
            predictions = sliding_window_predict(
                model=self.model,
                images=images,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                overlap=self.overlap,
            )

            loss = self.loss_fn(
                predictions,
                labels,
            )

            running_loss += loss.item()

            # ---------------------------------------------------------
            # Validation metric.
            #
            # ``metric_fn`` may return:
            #
            #     scalar -> one metric
            #     vector -> per-class metrics
            # ---------------------------------------------------------
            if self.metric_fn is not None:
                metric = self.metric_fn(
                    predictions,
                    labels,
                )

                metric = metric.detach().float()

                if running_metric is None:
                    running_metric = torch.zeros_like(
                        metric,
                        dtype=torch.float32,
                    )

                running_metric += metric

            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Validation DataLoader produced no batches.")

        val_loss = running_loss / num_batches

        if self.metric_fn is None:
            return val_loss, None

        if running_metric is None:
            raise RuntimeError(
                "Metric function was provided, but no metric "
                "was accumulated during validation."
            )

        val_metric = running_metric / num_batches

        return val_loss, val_metric

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_mean_dice: float | None,
        filename: str = "best_model.pt",
    ) -> Path:
        """
        Save a training checkpoint.

        Parameters
        ----------
        epoch:
            Epoch at which the checkpoint was created.

        val_loss:
            Validation loss.

        val_mean_dice:
            Validation mean foreground Dice score.

        filename:
            Checkpoint filename.
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_mean_dice": val_mean_dice,
            "best_val_dice": self.best_val_dice,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[Any]]:
        """
        Train the model for the configured number of epochs.

        The best checkpoint is selected according to validation mean
        foreground Dice when a metric function is provided.

        For IBSR-18, mean foreground Dice is calculated from:

            CSF + GM + WM

        Background is excluded from the mean.
        """
        for epoch in range(
            1,
            self.num_epochs + 1,
        ):
            # ---------------------------------------------------------
            # Training
            # ---------------------------------------------------------
            train_loss = self.train_one_epoch(train_loader)

            # ---------------------------------------------------------
            # Validation
            # ---------------------------------------------------------
            val_loss, val_per_class_dice = self.validate(val_loader)

            # ---------------------------------------------------------
            # Calculate mean foreground Dice.
            #
            # Expected metric ordering:
            #
            #     0 = background
            #     1 = CSF
            #     2 = GM
            #     3 = WM
            #
            # Background is excluded from the mean.
            # ---------------------------------------------------------
            if val_per_class_dice is not None:
                if val_per_class_dice.ndim == 0:
                    # Support scalar metrics for Trainer reuse.
                    val_mean_dice = val_per_class_dice.item()
                else:
                    if val_per_class_dice.numel() < 2:
                        raise RuntimeError(
                            "Per-class metric must contain at least "
                            "one foreground class."
                        )

                    val_mean_dice = val_per_class_dice[1:].mean().item()

            else:
                val_mean_dice = None

            # ---------------------------------------------------------
            # Store training history.
            # ---------------------------------------------------------
            self.history["train_loss"].append(train_loss)

            self.history["val_loss"].append(val_loss)

            if val_mean_dice is not None:
                self.history["val_mean_dice"].append(val_mean_dice)

            if val_per_class_dice is not None:
                self.history["val_per_class_dice"].append(
                    val_per_class_dice.cpu().tolist()
                )

            # ---------------------------------------------------------
            # Print epoch summary.
            # ---------------------------------------------------------
            if val_per_class_dice is not None:
                # -----------------------------------------------------
                # Per-class Dice.
                #
                # These indices correspond to:
                #     0 = background
                #     1 = CSF
                #     2 = GM
                #     3 = WM
                # -----------------------------------------------------
                if val_per_class_dice.numel() >= 4:
                    csf_dice = val_per_class_dice[1].item()

                    gm_dice = val_per_class_dice[2].item()

                    wm_dice = val_per_class_dice[3].item()

                    print(
                        f"Epoch "
                        f"{epoch:03d}/{self.num_epochs:03d} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | "
                        f"CSF={csf_dice:.4f} | "
                        f"GM={gm_dice:.4f} | "
                        f"WM={wm_dice:.4f} | "
                        f"mean_dice={val_mean_dice:.4f}"
                    )

                else:
                    print(
                        f"Epoch "
                        f"{epoch:03d}/{self.num_epochs:03d} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | "
                        f"mean_dice={val_mean_dice:.4f}"
                    )

            else:
                print(
                    f"Epoch "
                    f"{epoch:03d}/{self.num_epochs:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f}"
                )

            # ---------------------------------------------------------
            # Save the best checkpoint.
            # ---------------------------------------------------------
            if val_mean_dice is not None:
                # Dice is maximized.
                is_best = val_mean_dice > self.best_val_dice

                if is_best:
                    self.best_val_dice = val_mean_dice

                    checkpoint_path = self.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        val_mean_dice=val_mean_dice,
                        filename="best_model.pt",
                    )

                    print(f"  Saved best checkpoint: {checkpoint_path}")

            else:
                # Fallback: minimize validation loss when no metric
                # function has been supplied.
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    checkpoint_path = self.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        val_mean_dice=None,
                        filename="best_model.pt",
                    )

                    print(f"  Saved best checkpoint: {checkpoint_path}")

        return self.history
