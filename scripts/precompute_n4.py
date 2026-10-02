from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "n4"
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "n4_debug"


# ============================================================
# N4 configuration
# ============================================================

# Downsample image before N4 fitting for efficiency.
SHRINK_FACTOR = 4

# Four fitting levels, 50 iterations per level.
N4_MAX_ITERATIONS = [50, 50, 50, 50]

# N4 convergence threshold.
N4_CONVERGENCE_THRESHOLD = 0.001

# Number of B-spline control points.
N4_NUMBER_OF_CONTROL_POINTS = [4, 4, 4]


# ============================================================
# Subject discovery
# ============================================================


def get_subjects() -> list[str]:
    """Return all IBSR subject directories in data/raw."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory does not exist:\n{RAW_DIR}")

    subjects = sorted(
        path.name
        for path in RAW_DIR.iterdir()
        if path.is_dir() and path.name.startswith("IBSR_")
    )

    return subjects


# ============================================================
# Loading
# ============================================================


def load_image(subject: str) -> sitk.Image:
    """Load the raw IBSR image."""

    image_path = RAW_DIR / subject / f"{subject}.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(f"Raw image not found:\n{image_path}")

    image = sitk.ReadImage(str(image_path))

    print("\nInput image:")
    print(f"  {image_path}")
    print(f"  Size:       {image.GetSize()}")
    print(f"  Spacing:    {image.GetSpacing()}")
    print(f"  Origin:     {image.GetOrigin()}")
    print(f"  Direction:  {image.GetDirection()}")
    print(f"  Pixel type: {image.GetPixelIDTypeAsString()}")

    return image


# ============================================================
# Mask
# ============================================================


def create_foreground_mask(image: sitk.Image) -> sitk.Image:
    """
    Create a binary foreground mask from non-zero voxels.

    IBSR images are skull-stripped, so non-zero voxels provide
    the foreground mask used by N4.
    """

    mask = sitk.Cast(
        image > 0,
        sitk.sitkUInt8,
    )

    return mask


# ============================================================
# Statistics
# ============================================================


def get_masked_values(
    image: sitk.Image,
    mask: sitk.Image,
) -> np.ndarray:
    """
    Return finite image values inside the foreground mask.
    """

    image_array = sitk.GetArrayViewFromImage(image)
    mask_array = sitk.GetArrayViewFromImage(mask)

    values = image_array[mask_array > 0]

    values = values[np.isfinite(values)]

    return values.astype(np.float64)


def print_image_statistics(
    image: sitk.Image,
    name: str,
    mask: sitk.Image | None = None,
) -> None:
    """Print intensity statistics."""

    if mask is not None:
        image_array = sitk.GetArrayViewFromImage(image)
        mask_array = sitk.GetArrayViewFromImage(mask)

        values = image_array[mask_array > 0]
    else:
        values = sitk.GetArrayViewFromImage(image).reshape(-1)

    values = values[np.isfinite(values)]

    if values.size == 0:
        print(f"\n{name}: no finite values.")
        return

    p = np.percentile(
        values,
        [1, 5, 25, 50, 75, 95, 99, 99.9],
    )

    print(f"\n{name}:")
    print(f"  Voxels:    {values.size:,}")
    print(f"  Min:       {values.min():.6f}")
    print(f"  Max:       {values.max():.6f}")
    print(f"  Mean:      {values.mean():.6f}")
    print(f"  Std:       {values.std():.6f}")
    print(f"  P01:       {p[0]:.6f}")
    print(f"  P05:       {p[1]:.6f}")
    print(f"  P25:       {p[2]:.6f}")
    print(f"  Median:    {p[3]:.6f}")
    print(f"  P75:       {p[4]:.6f}")
    print(f"  P95:       {p[5]:.6f}")
    print(f"  P99:       {p[6]:.6f}")
    print(f"  P99.9:     {p[7]:.6f}")


def print_bias_field_statistics(
    bias_field: sitk.Image,
    mask: sitk.Image,
) -> None:
    """
    Print statistics for the multiplicative N4 bias field.

    IMPORTANT:
    Both the bias field and mask are converted to NumPy arrays
    before boolean indexing.
    """

    bias_array = sitk.GetArrayViewFromImage(bias_field).astype(np.float64)

    mask_array = sitk.GetArrayViewFromImage(mask)

    foreground = mask_array > 0

    values = bias_array[foreground]

    values = values[np.isfinite(values)]

    if values.size == 0:
        print("\nBias field: no finite foreground values found.")
        return

    p = np.percentile(
        values,
        [1, 5, 25, 50, 75, 95, 99, 99.9],
    )

    mean = values.mean()
    std = values.std()

    cv = std / mean if mean != 0 else np.nan

    print("\n" + "=" * 65)
    print("N4 MULTIPLICATIVE BIAS FIELD")
    print("=" * 65)

    print(f"Foreground voxels: {values.size:,}")
    print(f"Min:               {values.min():.6f}")
    print(f"Max:               {values.max():.6f}")
    print(f"Mean:              {mean:.6f}")
    print(f"Std:               {std:.6f}")
    print(f"CV:                {cv:.6f}")

    print("\nPercentiles:")
    print(f"  P01:             {p[0]:.6f}")
    print(f"  P05:             {p[1]:.6f}")
    print(f"  P25:             {p[2]:.6f}")
    print(f"  Median:          {p[3]:.6f}")
    print(f"  P75:             {p[4]:.6f}")
    print(f"  P95:             {p[5]:.6f}")
    print(f"  P99:             {p[6]:.6f}")
    print(f"  P99.9:           {p[7]:.6f}")

    print("\nInterpretation:")
    print("  The bias field is a multiplicative intensity correction field.")
    print("  Only values inside the foreground mask are relevant for diagnosis.")

    if np.any(values <= 0):
        print(
            "\nWARNING: Non-positive values detected inside the foreground bias field."
        )

    if not np.all(np.isfinite(values)):
        print("\nWARNING: Non-finite values detected inside the foreground bias field.")

    print("=" * 65)


# ============================================================
# Validation
# ============================================================


def validate_image_geometry(
    image: sitk.Image,
    reference: sitk.Image,
    name: str,
) -> None:
    """Validate image geometry against a reference image."""

    if image.GetSize() != reference.GetSize():
        raise RuntimeError(
            f"{name} size mismatch.\n"
            f"Expected: {reference.GetSize()}\n"
            f"Actual:   {image.GetSize()}"
        )

    if not np.allclose(
        image.GetSpacing(),
        reference.GetSpacing(),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            f"{name} spacing mismatch.\n"
            f"Expected: {reference.GetSpacing()}\n"
            f"Actual:   {image.GetSpacing()}"
        )

    if not np.allclose(
        image.GetOrigin(),
        reference.GetOrigin(),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(f"{name} origin mismatch.")

    if not np.allclose(
        image.GetDirection(),
        reference.GetDirection(),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(f"{name} direction mismatch.")


def validate_finite(
    image: sitk.Image,
    name: str,
) -> None:
    """Check that an image contains only finite values."""

    array = sitk.GetArrayViewFromImage(image)

    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains NaN or Inf values.")


def validate_image(
    image: sitk.Image,
    reference: sitk.Image,
    name: str,
) -> None:
    """Validate geometry and finite values."""

    validate_image_geometry(
        image,
        reference,
        name,
    )

    validate_finite(
        image,
        name,
    )

    print(f"{name} validation: PASSED")


# ============================================================
# N4 correction
# ============================================================


def run_n4(
    image: sitk.Image,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Run N4 bias-field correction.

    Returns
    -------
    corrected_image:
        Full-resolution N4-corrected image.

    bias_field:
        Full-resolution multiplicative bias field.
    """

    print("\n" + "=" * 65)
    print("N4 BIAS-FIELD CORRECTION")
    print("=" * 65)

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    image_float = sitk.Cast(
        image,
        sitk.sitkFloat32,
    )

    # --------------------------------------------------------
    # Foreground mask
    # --------------------------------------------------------

    mask = create_foreground_mask(image)

    # --------------------------------------------------------
    # Shrink for N4 fitting
    # --------------------------------------------------------

    dimension = image.GetDimension()

    shrink_factors = [SHRINK_FACTOR] * dimension

    image_shrunk = sitk.Shrink(
        image_float,
        shrink_factors,
    )

    mask_shrunk = sitk.Shrink(
        mask,
        shrink_factors,
    )

    print(f"\nOriginal size: {image.GetSize()}")

    print(f"Shrunk size:   {image_shrunk.GetSize()}")

    print(f"Shrink factor: {SHRINK_FACTOR}")

    # --------------------------------------------------------
    # Configure N4
    # --------------------------------------------------------

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    corrector.SetMaximumNumberOfIterations(N4_MAX_ITERATIONS)

    corrector.SetConvergenceThreshold(N4_CONVERGENCE_THRESHOLD)

    corrector.SetNumberOfControlPoints(N4_NUMBER_OF_CONTROL_POINTS)

    print(f"\nMaximum iterations: {N4_MAX_ITERATIONS}")

    print(f"Convergence threshold: {N4_CONVERGENCE_THRESHOLD}")

    print(f"Control points: {N4_NUMBER_OF_CONTROL_POINTS}")

    # --------------------------------------------------------
    # Execute N4
    # --------------------------------------------------------

    print("\nStarting N4 fitting...")

    start_time = time.time()

    corrector.Execute(
        image_shrunk,
        mask_shrunk,
    )

    elapsed = time.time() - start_time

    print(f"N4 fitting completed in {elapsed:.2f} seconds.")

    # --------------------------------------------------------
    # Recover full-resolution log bias field
    # --------------------------------------------------------

    log_bias_field = corrector.GetLogBiasFieldAsImage(image_float)

    # --------------------------------------------------------
    # Convert log bias field to multiplicative field
    # --------------------------------------------------------

    bias_field = sitk.Exp(log_bias_field)

    # --------------------------------------------------------
    # Correct original full-resolution image
    #
    # corrected = exp(log(image) - log_bias_field)
    #
    # Protect against log(0).
    # --------------------------------------------------------

    image_array = sitk.GetArrayFromImage(image_float)

    image_array = np.maximum(
        image_array.astype(np.float32),
        1e-6,
    )

    safe_image = sitk.GetImageFromArray(
        image_array,
        isVector=False,
    )

    safe_image.CopyInformation(image_float)

    log_image = sitk.Log(safe_image)

    corrected_image = sitk.Exp(log_image - log_bias_field)

    # --------------------------------------------------------
    # Restore exact zero background
    # --------------------------------------------------------

    corrected_image = sitk.Mask(
        corrected_image,
        mask,
    )

    corrected_image = sitk.Cast(
        corrected_image,
        sitk.sitkFloat32,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_image(
        corrected_image,
        image,
        "Corrected image",
    )

    validate_image(
        bias_field,
        image,
        "Bias field",
    )

    return corrected_image, bias_field


# ============================================================
# Saving
# ============================================================


def save_outputs(
    subject: str,
    corrected_image: sitk.Image,
    bias_field: sitk.Image,
) -> None:
    """Save corrected image and debug bias field."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEBUG_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    corrected_path = OUTPUT_DIR / f"{subject}_n4.nii.gz"

    bias_path = DEBUG_OUTPUT_DIR / f"{subject}_bias_field.nii.gz"

    sitk.WriteImage(
        corrected_image,
        str(corrected_path),
    )

    sitk.WriteImage(
        bias_field,
        str(bias_path),
    )

    print("\nSaved files:")

    print(f"  Corrected image:\n    {corrected_path}")

    print(f"  Bias field:\n    {bias_path}")


# ============================================================
# Process subject
# ============================================================


def process_subject(
    subject: str,
    overwrite: bool = False,
) -> None:
    """Run N4 preprocessing for one subject."""

    corrected_path = OUTPUT_DIR / f"{subject}_n4.nii.gz"

    if corrected_path.exists() and not overwrite:
        print(f"\nSkipping {subject}: output already exists.")

        print("Use --overwrite to regenerate.")

        return

    print("\n")
    print("#" * 70)
    print(f"# SUBJECT: {subject}")
    print("#" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = load_image(subject)

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask = create_foreground_mask(image)

    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    print_image_statistics(
        image,
        "RAW IMAGE — FOREGROUND",
        mask,
    )

    # --------------------------------------------------------
    # N4
    # --------------------------------------------------------

    corrected_image, bias_field = run_n4(image)

    # --------------------------------------------------------
    # Corrected statistics
    # --------------------------------------------------------

    print_image_statistics(
        corrected_image,
        "N4 CORRECTED IMAGE — FOREGROUND",
        mask,
    )

    # --------------------------------------------------------
    # Bias-field diagnostics
    # --------------------------------------------------------

    print_bias_field_statistics(
        bias_field,
        mask,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        subject,
        corrected_image,
        bias_field,
    )

    print(f"\nFinished {subject}.")


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Precompute SimpleITK N4 bias-field corrected IBSR-18 volumes.")
    )

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help=("Process a single subject, e.g. --subject IBSR_01"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Overwrite existing N4 outputs."),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("IBSR-18 N4 PRECOMPUTATION")
    print("=" * 70)

    print(f"\nProject root:\n  {PROJECT_ROOT}")

    print(f"\nRaw directory:\n  {RAW_DIR}")

    print(f"\nOutput directory:\n  {OUTPUT_DIR}")

    print(f"\nDebug directory:\n  {DEBUG_OUTPUT_DIR}")

    print("\nN4 configuration:")
    print(f"  Shrink factor:        {SHRINK_FACTOR}")
    print(f"  Maximum iterations:   {N4_MAX_ITERATIONS}")
    print(f"  Convergence threshold: {N4_CONVERGENCE_THRESHOLD}")
    print(f"  Control points:       {N4_NUMBER_OF_CONTROL_POINTS}")

    # --------------------------------------------------------
    # Single subject
    # --------------------------------------------------------

    if args.subject is not None:
        process_subject(
            args.subject,
            overwrite=args.overwrite,
        )

    # --------------------------------------------------------
    # All subjects
    # --------------------------------------------------------

    else:
        subjects = get_subjects()

        if not subjects:
            raise RuntimeError(f"No IBSR subjects found in:\n{RAW_DIR}")

        print(f"\nFound {len(subjects)} subjects:")

        for subject in subjects:
            print(f"  {subject}")

        for subject in subjects:
            process_subject(
                subject,
                overwrite=args.overwrite,
            )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
