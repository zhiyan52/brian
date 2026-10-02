"""
Preprocessing and augmentation transforms for the IBSR-18 dataset.

The IBSR-18 images may appear in two forms:

1. Original/raw NIfTI images:
       (X, Y, Z, 1)

2. Precomputed N4-corrected images:
       (X, Y, Z)

Both forms are converted to MONAI's channel-first convention:

       (1, X, Y, Z)

The preprocessing pipeline supports:
    - NIfTI loading
    - channel-first conversion
    - RAS orientation
    - optional isotropic resampling
    - optional N4 bias-field correction
    - intensity normalization
    - foreground cropping
    - random training patches
    - random spatial augmentation

For the main N4 experiment, N4 correction is performed offline and
the corrected images are loaded from data/processed/n4. Runtime N4
correction should therefore be disabled.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandSpatialCropSamplesd,
    Spacingd,
)


class EnsureSingleChannelFirstd(MapTransform):
    """
    Ensure a single-channel 3D image is channel-first.

    Supported input layouts:

        [X, Y, Z]
            -> [1, X, Y, Z]

        [X, Y, Z, 1]
            -> [1, X, Y, Z]

        [1, X, Y, Z]
            -> unchanged

    The transformation operates directly on the loaded tensor/MetaTensor
    so that MONAI metadata is preserved.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self,
        data: Mapping[Hashable, Any],
    ) -> dict[Hashable, Any]:
        d = dict(data)

        for key in self.key_iterator(d):
            value = d[key]

            if not hasattr(value, "ndim"):
                raise TypeError(
                    f"Expected a tensor-like object for '{key}', "
                    f"but received {type(value).__name__}."
                )

            if value.ndim == 3:
                # Precomputed N4:
                # [X, Y, Z] -> [1, X, Y, Z]
                d[key] = value.unsqueeze(0)

            elif value.ndim == 4:
                if value.shape[-1] == 1:
                    # Raw IBSR:
                    # [X, Y, Z, 1] -> [1, X, Y, Z]
                    d[key] = value.permute(3, 0, 1, 2)

                elif value.shape[0] == 1:
                    # Already channel-first.
                    d[key] = value

                else:
                    raise ValueError(
                        f"Expected a single-channel 3D image for '{key}', "
                        f"but received shape {tuple(value.shape)}."
                    )

            else:
                raise ValueError(
                    f"Expected a 3D or 4D image for '{key}', "
                    f"but received shape {tuple(value.shape)}."
                )

        return d


class N4BiasFieldCorrectiond(MapTransform):
    """
    Apply N4 bias-field correction using SimpleITK.

    This transform is intended for runtime N4 correction and is useful
    for testing or experiments.

    For the main N4 experiment, prefer precomputing N4-corrected images
    once and loading them from data/processed/n4.
    """

    def __init__(
        self,
        keys: KeysCollection,
        shrink_factor: int = 4,
        maximum_number_of_iterations: tuple[int, ...] = (50,) * 8,
        convergence_threshold: float = 0.001,
        number_of_control_points: tuple[int, int, int] = (4, 4, 4),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        if shrink_factor < 1:
            raise ValueError("shrink_factor must be >= 1.")

        if not maximum_number_of_iterations:
            raise ValueError("maximum_number_of_iterations must not be empty.")

        if convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be > 0.")

        if len(number_of_control_points) != 3:
            raise ValueError("number_of_control_points must contain three values.")

        self.shrink_factor = shrink_factor
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.convergence_threshold = convergence_threshold
        self.number_of_control_points = number_of_control_points

    @staticmethod
    def _to_numpy_3d(
        image: Any,
    ) -> tuple[np.ndarray, str]:
        """
        Convert an image to a 3D NumPy array.

        Returns
        -------
        array:
            3D float32 NumPy array.

        layout:
            Original layout identifier used to restore the result.
        """

        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)

        if array.ndim == 3:
            return (
                array.astype(np.float32),
                "xyz",
            )

        if array.ndim == 4:
            if array.shape[-1] == 1:
                return (
                    array[..., 0].astype(np.float32),
                    "xyz1",
                )

            if array.shape[0] == 1:
                return (
                    array[0].astype(np.float32),
                    "1xyz",
                )

        raise ValueError(
            "N4BiasFieldCorrectiond expects a single-channel "
            "3D image with shape [X,Y,Z], [X,Y,Z,1], or "
            "[1,X,Y,Z]. "
            f"Received shape {array.shape}."
        )

    @staticmethod
    def _restore_layout(
        array: np.ndarray,
        layout: str,
        reference: Any,
    ) -> Any:
        """
        Restore the original channel layout and tensor type.
        """

        if layout == "xyz":
            restored = array

        elif layout == "xyz1":
            restored = array[..., np.newaxis]

        elif layout == "1xyz":
            restored = array[np.newaxis, ...]

        else:
            raise ValueError(f"Unknown layout: {layout}")

        if isinstance(reference, torch.Tensor):
            tensor = torch.from_numpy(np.ascontiguousarray(restored))

            tensor = tensor.to(device=reference.device)

            if reference.dtype.is_floating_point:
                tensor = tensor.to(dtype=reference.dtype)

            return tensor

        return restored

    def _apply_n4(
        self,
        image: Any,
    ) -> Any:
        """
        Apply N4 correction to one image.
        """

        array, layout = self._to_numpy_3d(image)

        # Remove invalid values.
        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # N4 operates on a SimpleITK image.
        sitk_image = sitk.GetImageFromArray(array)

        # Use non-zero foreground as the N4 mask.
        mask_array = (array > 0).astype(np.uint8)

        sitk_mask = sitk.GetImageFromArray(mask_array)

        # If no foreground exists, leave the image unchanged.
        if not np.any(mask_array):
            return image

        # Shrink for faster N4 computation.
        if self.shrink_factor > 1:
            sitk_image_shrunk = sitk.Shrink(
                sitk_image,
                [self.shrink_factor] * 3,
            )

            sitk_mask_shrunk = sitk.Shrink(
                sitk_mask,
                [self.shrink_factor] * 3,
            )
        else:
            sitk_image_shrunk = sitk_image
            sitk_mask_shrunk = sitk_mask

        sitk_image_shrunk = sitk.Cast(
            sitk_image_shrunk,
            sitk.sitkFloat32,
        )

        sitk_mask_shrunk = sitk.Cast(
            sitk_mask_shrunk,
            sitk.sitkUInt8,
        )

        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()

        n4_filter.SetMaximumNumberOfIterations(list(self.maximum_number_of_iterations))

        n4_filter.SetConvergenceThreshold(self.convergence_threshold)

        n4_filter.SetNumberOfControlPoints(list(self.number_of_control_points))

        corrected = n4_filter.Execute(
            sitk_image_shrunk,
            sitk_mask_shrunk,
        )

        # Restore the original image size.
        corrected = sitk.Resample(
            corrected,
            sitk_image,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )

        corrected_array = sitk.GetArrayFromImage(corrected).astype(np.float32)

        return self._restore_layout(
            corrected_array,
            layout,
            image,
        )

    def __call__(
        self,
        data: Mapping[Hashable, Any],
    ) -> dict[Hashable, Any]:
        d = dict(data)

        for key in self.key_iterator(d):
            d[key] = self._apply_n4(d[key])

        return d


def get_bias_correction_transforms(
    keys: tuple[str, ...] = ("image",),
) -> Compose:
    """
    Return the runtime N4 bias-field correction pipeline.
    """

    return Compose(
        [
            N4BiasFieldCorrectiond(
                keys=keys,
                shrink_factor=4,
                maximum_number_of_iterations=(50,) * 8,
                convergence_threshold=0.001,
                number_of_control_points=(4, 4, 4),
            ),
        ]
    )


def get_train_transforms(
    patch_size: tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 1,
    target_spacing: tuple[float, float, float] | None = None,
    use_n4_bias_correction: bool = False,
) -> Compose:
    """
    Build the training preprocessing and augmentation pipeline.

    Parameters
    ----------
    patch_size:
        Spatial dimensions of randomly sampled training patches.

    num_samples:
        Number of patches sampled from each subject.

    target_spacing:
        Optional target voxel spacing. If None, native spacing
        is preserved.

    use_n4_bias_correction:
        Whether to perform N4 bias correction at runtime.

        Set this to False when using precomputed N4 images.
    """

    if len(patch_size) != 3:
        raise ValueError("patch_size must contain exactly three values.")

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    transforms: list[Any] = [
        # Load the NIfTI file and retain MONAI metadata.
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
        ),
        # Handles both raw IBSR [X,Y,Z,1] and precomputed
        # N4 [X,Y,Z] images.
        EnsureSingleChannelFirstd(
            keys=["image", "label"],
        ),
        # Standardize orientation.
        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
        ),
    ]

    # Optional resampling.
    if target_spacing is not None:
        if len(target_spacing) != 3:
            raise ValueError("target_spacing must contain exactly three values.")

        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            )
        )

    # Runtime N4 is only used when explicitly requested.
    #
    # Experiment 4 should use:
    #     use_n4_bias_correction=False
    #
    # because its images have already been corrected offline.
    if use_n4_bias_correction:
        transforms.append(
            N4BiasFieldCorrectiond(
                keys=["image"],
                shrink_factor=4,
                maximum_number_of_iterations=(50,) * 8,
                convergence_threshold=0.001,
                number_of_control_points=(4, 4, 4),
            )
        )

    transforms.extend(
        [
            # Normalize MRI intensities using the non-zero brain
            # region.
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
            ),
            # Remove empty background around the brain.
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
            ),
            # Sample fixed-size 3D training patches.
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=patch_size,
                num_samples=num_samples,
                random_size=False,
            ),
            # Random left/right-style spatial flips.
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=2,
            ),
            # Ensure final dtypes.
            EnsureTyped(
                keys=["image", "label"],
                dtype=(torch.float32, torch.int64),
            ),
        ]
    )

    return Compose(transforms)


def get_val_transforms(
    target_spacing: tuple[float, float, float] | None = None,
    use_n4_bias_correction: bool = False,
) -> Compose:
    """
    Build the validation preprocessing pipeline.

    Validation data is not randomly cropped or augmented.

    Parameters
    ----------
    target_spacing:
        Optional target voxel spacing.

    use_n4_bias_correction:
        Whether to perform N4 correction at runtime.
    """

    transforms: list[Any] = [
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
        ),
        EnsureSingleChannelFirstd(
            keys=["image", "label"],
        ),
        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
        ),
    ]

    if target_spacing is not None:
        if len(target_spacing) != 3:
            raise ValueError("target_spacing must contain exactly three values.")

        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            )
        )

    if use_n4_bias_correction:
        transforms.append(
            N4BiasFieldCorrectiond(
                keys=["image"],
                shrink_factor=4,
                maximum_number_of_iterations=(50,) * 8,
                convergence_threshold=0.001,
                number_of_control_points=(4, 4, 4),
            )
        )

    transforms.extend(
        [
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
            ),
            EnsureTyped(
                keys=["image", "label"],
                dtype=(torch.float32, torch.int64),
            ),
        ]
    )

    return Compose(transforms)


def get_test_transforms(
    target_spacing: tuple[float, float, float] | None = None,
    use_n4_bias_correction: bool = False,
) -> Compose:
    """
    Build the test preprocessing pipeline.

    Test subjects do not have labels, so only the image key
    is processed.
    """

    transforms: list[Any] = [
        LoadImaged(
            keys=["image"],
            image_only=False,
        ),
        EnsureSingleChannelFirstd(
            keys=["image"],
        ),
        Orientationd(
            keys=["image"],
            axcodes="RAS",
        ),
    ]

    if target_spacing is not None:
        if len(target_spacing) != 3:
            raise ValueError("target_spacing must contain exactly three values.")

        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=target_spacing,
                mode="bilinear",
            )
        )

    if use_n4_bias_correction:
        transforms.append(
            N4BiasFieldCorrectiond(
                keys=["image"],
                shrink_factor=4,
                maximum_number_of_iterations=(50,) * 8,
                convergence_threshold=0.001,
                number_of_control_points=(4, 4, 4),
            )
        )

    transforms.extend(
        [
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
            ),
            EnsureTyped(
                keys=["image"],
                dtype=torch.float32,
            ),
        ]
    )

    return Compose(transforms)
