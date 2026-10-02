"""
DataModule for the IBSR-18 brain MRI segmentation project.

This module connects:
    - subject splits
    - datasets
    - preprocessing transforms
    - PyTorch DataLoaders

The DataModule keeps data-loading logic out of the training script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monai.data import Dataset, list_data_collate
from torch.utils.data import DataLoader

from ibsr_unet.data.datasets import (
    create_test_dataset,
    create_train_dataset,
    create_val_dataset,
)
from ibsr_unet.data.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)


class IBSRDataModule:
    """
    Manage IBSR-18 datasets and DataLoaders.

    Parameters
    ----------
    data_dir:
        Directory containing the IBSR-18 subject folders.

    splits_dir:
        Directory containing train.txt, val.txt, and test.txt.

    patch_size:
        Spatial size of training patches.

    num_samples:
        Number of patches sampled from each training subject.

    target_spacing:
        Optional target voxel spacing. If None, native spacing is preserved.

    use_n4_bias_correction:
        Whether to apply N4 bias-field correction at runtime.

        This should be False when using precomputed N4 images.

    n4_dir:
        Directory containing precomputed N4-corrected MRI images.

    use_precomputed_n4:
        Whether to load MRI images from the precomputed N4 directory.

    batch_size:
        Number of samples processed by the DataLoader at once.

    num_workers:
        Number of worker processes used by DataLoaders.

    pin_memory:
        Whether DataLoader workers pin CPU memory for faster GPU transfer.

    persistent_workers:
        Whether worker processes remain alive between epochs.
    """

    def __init__(
        self,
        data_dir: Path,
        splits_dir: Path,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        num_samples: int = 2,
        target_spacing: tuple[float, float, float] | None = None,
        use_n4_bias_correction: bool = False,
        n4_dir: Path | None = None,
        use_precomputed_n4: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.splits_dir = Path(splits_dir)

        self.patch_size = patch_size
        self.num_samples = num_samples
        self.target_spacing = target_spacing

        self.use_n4_bias_correction = use_n4_bias_correction

        self.n4_dir = Path(n4_dir) if n4_dir is not None else None
        self.use_precomputed_n4 = use_precomputed_n4

        if self.use_precomputed_n4 and self.n4_dir is None:
            raise ValueError("n4_dir must be provided when use_precomputed_n4=True.")

        if self.use_precomputed_n4 and self.use_n4_bias_correction:
            raise ValueError(
                "Runtime N4 bias correction must be disabled when "
                "using precomputed N4 images."
            )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # persistent_workers requires at least one worker.
        self.persistent_workers = persistent_workers and num_workers > 0

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self) -> None:
        """Create train, validation, and test datasets."""

        # Use the precomputed N4 directory as the image source when
        # enabled. Labels always remain in the original data directory.
        image_dir = self.n4_dir if self.use_precomputed_n4 else None

        self.train_dataset = create_train_dataset(
            data_dir=self.data_dir,
            splits_dir=self.splits_dir,
            image_dir=image_dir,
            transform=get_train_transforms(
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_spacing=self.target_spacing,
                use_n4_bias_correction=self.use_n4_bias_correction,
            ),
        )

        self.val_dataset = create_val_dataset(
            data_dir=self.data_dir,
            splits_dir=self.splits_dir,
            image_dir=image_dir,
            transform=get_val_transforms(
                target_spacing=self.target_spacing,
                use_n4_bias_correction=self.use_n4_bias_correction,
            ),
        )

        self.test_dataset = create_test_dataset(
            data_dir=self.data_dir,
            splits_dir=self.splits_dir,
            image_dir=image_dir,
            transform=get_test_transforms(
                target_spacing=self.target_spacing,
                use_n4_bias_correction=self.use_n4_bias_correction,
            ),
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader."""

        if self.train_dataset is None:
            raise RuntimeError("DataModule has not been set up. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=list_data_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader."""

        if self.val_dataset is None:
            raise RuntimeError("DataModule has not been set up. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader."""

        if self.test_dataset is None:
            raise RuntimeError("DataModule has not been set up. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary of the DataModule configuration."""

        return {
            "data_dir": str(self.data_dir),
            "splits_dir": str(self.splits_dir),
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "target_spacing": self.target_spacing,
            "use_n4_bias_correction": (self.use_n4_bias_correction),
            "n4_dir": (str(self.n4_dir) if self.n4_dir is not None else None),
            "use_precomputed_n4": (self.use_precomputed_n4),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "train_size": (
                len(self.train_dataset) if self.train_dataset is not None else None
            ),
            "val_size": (
                len(self.val_dataset) if self.val_dataset is not None else None
            ),
            "test_size": (
                len(self.test_dataset) if self.test_dataset is not None else None
            ),
        }
