from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
N4_DIR = PROJECT_ROOT / "data" / "processed" / "n4"
BIAS_DIR = PROJECT_ROOT / "data" / "processed" / "n4_debug"

REPORT_DIR = PROJECT_ROOT / "outputs" / "qc"
CSV_PATH = REPORT_DIR / "n4_qc_report.csv"


# =====================================================================
# QC THRESHOLDS
# =====================================================================

# N4 relationship:
#
#     corrected = raw / bias
#
# Therefore:
#
#     corrected * bias ≈ raw
#
MAX_RECONSTRUCTION_REL_ERROR = 1e-4

# Bias-field smoothness.
# This is a review criterion rather than a hard rejection criterion.
MAX_BIAS_GRADIENT_P95 = 0.05


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================


def percentile(values: np.ndarray, q: float) -> float:
    """Return a percentile as a regular Python float."""
    return float(np.percentile(values, q))


def load_3d_array(
    path: Path,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """
    Load a NIfTI image and normalize a trailing singleton dimension.

    IBSR raw images are stored as:
        (256, 128, 256, 1)

    while the N4 output is stored as:
        (256, 128, 256)

    For QC purposes only, (X, Y, Z, 1) is converted to (X, Y, Z).
    No files are modified.
    """

    image = nib.load(path)

    array = np.asarray(
        image.dataobj,
        dtype=np.float32,
    )

    if array.ndim == 4:
        if array.shape[-1] != 1:
            raise ValueError(
                f"Expected a singleton fourth dimension, "
                f"but found shape {array.shape} in {path}"
            )

        array = np.squeeze(
            array,
            axis=-1,
        )

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D image after normalization, "
            f"but found shape {array.shape} in {path}"
        )

    return image, array


def geometry_matches(
    image_a: nib.Nifti1Image,
    array_a: np.ndarray,
    image_b: nib.Nifti1Image,
    array_b: np.ndarray,
    atol: float = 1e-5,
) -> bool:
    """
    Check shape, spacing, and affine compatibility.

    The shape comparison uses normalized 3D arrays so that
    (X, Y, Z, 1) and (X, Y, Z) are treated as equivalent.
    """

    if array_a.shape != array_b.shape:
        return False

    spacing_a = np.asarray(
        image_a.header.get_zooms()[:3],
        dtype=np.float64,
    )

    spacing_b = np.asarray(
        image_b.header.get_zooms()[:3],
        dtype=np.float64,
    )

    if not np.allclose(
        spacing_a,
        spacing_b,
        atol=atol,
        rtol=0,
    ):
        return False

    if not np.allclose(
        image_a.affine,
        image_b.affine,
        atol=atol,
        rtol=0,
    ):
        return False

    return True


def foreground_stats(
    values: np.ndarray,
    foreground: np.ndarray,
    prefix: str,
) -> dict:
    """Calculate intensity statistics inside the foreground."""

    foreground_values = values[foreground]

    if foreground_values.size == 0:
        return {
            f"{prefix}_voxels": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p01": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
            f"{prefix}_p999": np.nan,
        }

    return {
        f"{prefix}_voxels": int(foreground_values.size),
        f"{prefix}_min": float(np.min(foreground_values)),
        f"{prefix}_max": float(np.max(foreground_values)),
        f"{prefix}_mean": float(np.mean(foreground_values)),
        f"{prefix}_std": float(np.std(foreground_values)),
        f"{prefix}_p01": percentile(foreground_values, 1),
        f"{prefix}_p05": percentile(foreground_values, 5),
        f"{prefix}_p25": percentile(foreground_values, 25),
        f"{prefix}_median": percentile(foreground_values, 50),
        f"{prefix}_p75": percentile(foreground_values, 75),
        f"{prefix}_p95": percentile(foreground_values, 95),
        f"{prefix}_p99": percentile(foreground_values, 99),
        f"{prefix}_p999": percentile(foreground_values, 99.9),
    }


def calculate_reconstruction_error(
    raw: np.ndarray,
    n4: np.ndarray,
    bias: np.ndarray,
    foreground: np.ndarray,
) -> dict:
    """
    Verify:

        N4_corrected * bias_field ≈ raw

    The calculation is performed in float64.
    """

    raw_fg = raw[foreground].astype(np.float64)

    n4_fg = n4[foreground].astype(np.float64)

    bias_fg = bias[foreground].astype(np.float64)

    reconstructed = n4_fg * bias_fg

    valid = np.abs(raw_fg) > 1e-8

    relative_error = np.zeros_like(raw_fg)

    relative_error[valid] = np.abs(reconstructed[valid] - raw_fg[valid]) / np.abs(
        raw_fg[valid]
    )

    absolute_error = np.abs(reconstructed - raw_fg)

    return {
        "reconstruction_rel_mean": float(np.mean(relative_error)),
        "reconstruction_rel_median": float(np.median(relative_error)),
        "reconstruction_rel_p95": percentile(
            relative_error,
            95,
        ),
        "reconstruction_rel_p99": percentile(
            relative_error,
            99,
        ),
        "reconstruction_rel_max": float(np.max(relative_error)),
        "reconstruction_abs_mean": float(np.mean(absolute_error)),
        "reconstruction_abs_max": float(np.max(absolute_error)),
    }


def calculate_bias_smoothness(
    bias: np.ndarray,
    foreground: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict:
    """
    Calculate spatial gradients of the multiplicative bias field.

    NIfTI array axes:
        axis 0 = x
        axis 1 = y
        axis 2 = z

    Therefore:
        spacing = (x, y, z)

    The foreground mask is eroded before measuring gradients so that
    the foreground/background boundary does not dominate the result.
    """

    interior_mask = binary_erosion(
        foreground,
        structure=np.ones(
            (3, 3, 3),
            dtype=bool,
        ),
        iterations=1,
        border_value=0,
    )

    if np.count_nonzero(interior_mask) == 0:
        interior_mask = foreground

    gradient_x, gradient_y, gradient_z = np.gradient(
        bias.astype(np.float64),
        spacing[0],
        spacing[1],
        spacing[2],
    )

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)

    values = gradient_magnitude[interior_mask]

    return {
        "bias_gradient_median": percentile(
            values,
            50,
        ),
        "bias_gradient_p95": percentile(
            values,
            95,
        ),
        "bias_gradient_p99": percentile(
            values,
            99,
        ),
        "bias_gradient_rms": float(np.sqrt(np.mean(values**2))),
        "bias_gradient_max": float(np.max(values)),
    }


# =====================================================================
# SUBJECT QC
# =====================================================================


def qc_subject(
    subject: str,
) -> dict:
    # -------------------------------------------------------------
    # Actual project structure
    #
    # raw:
    #   data/raw/IBSR_01/IBSR_01.nii.gz
    #
    # N4:
    #   data/processed/n4/IBSR_01/IBSR_01.nii.gz
    #
    # bias:
    #   data/processed/n4_debug/IBSR_01_bias_field.nii.gz
    # -------------------------------------------------------------

    raw_path = RAW_DIR / subject / f"{subject}.nii.gz"

    n4_path = N4_DIR / subject / f"{subject}.nii.gz"

    bias_path = BIAS_DIR / f"{subject}_bias_field.nii.gz"

    print()
    print("=" * 70)
    print(f"QC: {subject}")
    print("=" * 70)

    print("Raw:")
    print(f"  {raw_path}")

    print("N4:")
    print(f"  {n4_path}")

    print("Bias field:")
    print(f"  {bias_path}")

    result = {
        "subject": subject,
        "status": "PASS",
        "geometry_pass": False,
        "finite_pass": False,
        "voxel_count_pass": False,
        "reconstruction_pass": False,
        "bias_range_pass": False,
        "smoothness_pass": False,
        "failure_reason": "",
    }

    # -------------------------------------------------------------
    # FILE EXISTENCE
    # -------------------------------------------------------------

    missing_files = []

    for path in (
        raw_path,
        n4_path,
        bias_path,
    ):
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        result["status"] = "FAIL"
        result["failure_reason"] = "Missing file(s)"

        print()
        print("FAIL: missing file(s):")

        for path in missing_files:
            print(f"  {path}")

        return result

    # -------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------

    try:
        raw_img, raw = load_3d_array(raw_path)

        n4_img, n4 = load_3d_array(n4_path)

        bias_img, bias = load_3d_array(bias_path)

    except Exception as exc:
        result["status"] = "FAIL"
        result["failure_reason"] = f"Could not load NIfTI file: {exc}"

        print()
        print(f"FAIL: could not load NIfTI files: {exc}")

        return result

    print()
    print("Normalized shapes:")
    print(f"  Raw:   {raw.shape}")
    print(f"  N4:    {n4.shape}")
    print(f"  Bias:  {bias.shape}")

    # -------------------------------------------------------------
    # GEOMETRY
    # -------------------------------------------------------------

    geometry_pass = geometry_matches(
        raw_img,
        raw,
        n4_img,
        n4,
    ) and geometry_matches(
        raw_img,
        raw,
        bias_img,
        bias,
    )

    result["geometry_pass"] = geometry_pass

    print()
    print(f"Geometry:              {'PASS' if geometry_pass else 'FAIL'}")

    if not geometry_pass:
        result["status"] = "FAIL"
        result["failure_reason"] = "Geometry mismatch"

        return result

    # -------------------------------------------------------------
    # FOREGROUND MASK
    # -------------------------------------------------------------

    foreground = raw > 0

    raw_foreground_voxels = int(np.count_nonzero(foreground))

    n4_foreground = n4 > 0

    n4_foreground_voxels = int(np.count_nonzero(n4_foreground))

    voxel_count_pass = (
        raw_foreground_voxels == n4_foreground_voxels and raw_foreground_voxels > 0
    )

    result["voxel_count_pass"] = voxel_count_pass

    print(
        f"Foreground voxels:    "
        f"{raw_foreground_voxels:,} raw / "
        f"{n4_foreground_voxels:,} N4 "
        f"{'PASS' if voxel_count_pass else 'FAIL'}"
    )

    if not voxel_count_pass:
        result["status"] = "FAIL"
        result["failure_reason"] = "Foreground voxel count changed"

    # -------------------------------------------------------------
    # FINITE VALUES
    # -------------------------------------------------------------

    raw_finite = bool(np.all(np.isfinite(raw)))

    n4_finite = bool(np.all(np.isfinite(n4)))

    bias_finite = bool(np.all(np.isfinite(bias)))

    finite_pass = raw_finite and n4_finite and bias_finite

    result["finite_pass"] = finite_pass

    print(f"Finite values:        {'PASS' if finite_pass else 'FAIL'}")

    if not finite_pass:
        result["status"] = "FAIL"
        result["failure_reason"] = "NaN or Inf detected"

        return result

    # -------------------------------------------------------------
    # INTENSITY STATISTICS
    # -------------------------------------------------------------

    result.update(
        foreground_stats(
            raw,
            foreground,
            "raw",
        )
    )

    result.update(
        foreground_stats(
            n4,
            foreground,
            "n4",
        )
    )

    result.update(
        foreground_stats(
            bias,
            foreground,
            "bias",
        )
    )

    bias_foreground = bias[foreground]

    bias_mean = float(np.mean(bias_foreground))

    bias_std = float(np.std(bias_foreground))

    bias_median = float(np.median(bias_foreground))

    bias_cv = bias_std / bias_mean if bias_mean != 0 else np.inf

    result["bias_cv"] = float(bias_cv)

    # A multiplicative bias field should be
    # strictly positive in the foreground.
    #
    # The upper bound is deliberately generous and is only intended
    # as a sanity check, not a claim about a universal N4 limit.
    bias_range_pass = (
        float(np.min(bias_foreground)) > 0 and float(np.max(bias_foreground)) < 2.0
    )

    result["bias_range_pass"] = bias_range_pass

    print()
    print("RAW foreground:")
    print(f"  Mean:       {result['raw_mean']:.4f}")
    print(f"  Median:     {result['raw_median']:.4f}")
    print(f"  P01-P99:    {result['raw_p01']:.4f} - {result['raw_p99']:.4f}")

    print("N4 foreground:")
    print(f"  Mean:       {result['n4_mean']:.4f}")
    print(f"  Median:     {result['n4_median']:.4f}")
    print(f"  P01-P99:    {result['n4_p01']:.4f} - {result['n4_p99']:.4f}")

    print("Bias field:")
    print(f"  Min:        {result['bias_min']:.6f}")
    print(f"  Max:        {result['bias_max']:.6f}")
    print(f"  Median:     {bias_median:.6f}")
    print(f"  Mean:       {bias_mean:.6f}")
    print(f"  Std:        {bias_std:.6f}")
    print(f"  CV:         {bias_cv:.6f}")

    print(f"Bias range:           {'PASS' if bias_range_pass else 'FAIL'}")

    if not bias_range_pass:
        result["status"] = "FAIL"
        result["failure_reason"] = "Invalid bias-field range"

    # -------------------------------------------------------------
    # RECONSTRUCTION
    # -------------------------------------------------------------

    reconstruction = calculate_reconstruction_error(
        raw=raw,
        n4=n4,
        bias=bias,
        foreground=foreground,
    )

    result.update(reconstruction)

    reconstruction_pass = (
        result["reconstruction_rel_p99"] <= MAX_RECONSTRUCTION_REL_ERROR
    )

    result["reconstruction_pass"] = reconstruction_pass

    print()
    print("Reconstruction: N4 × bias ≈ raw")

    print(f"  Mean relative error:   {result['reconstruction_rel_mean']:.3e}")

    print(f"  Median relative error: {result['reconstruction_rel_median']:.3e}")

    print(f"  P95 relative error:     {result['reconstruction_rel_p95']:.3e}")

    print(f"  P99 relative error:     {result['reconstruction_rel_p99']:.3e}")

    print(f"  Max relative error:     {result['reconstruction_rel_max']:.3e}")

    print(f"  Mean absolute error:    {result['reconstruction_abs_mean']:.3e}")

    print(f"  Max absolute error:     {result['reconstruction_abs_max']:.3e}")

    print(f"Reconstruction:       {'PASS' if reconstruction_pass else 'FAIL'}")

    if not reconstruction_pass:
        result["status"] = "FAIL"
        result["failure_reason"] = "Reconstruction error exceeds threshold"

    # -------------------------------------------------------------
    # BIAS SMOOTHNESS
    # -------------------------------------------------------------

    spacing = tuple(float(x) for x in raw_img.header.get_zooms()[:3])

    smoothness = calculate_bias_smoothness(
        bias=bias,
        foreground=foreground,
        spacing=spacing,
    )

    result.update(smoothness)

    smoothness_pass = result["bias_gradient_p95"] <= MAX_BIAS_GRADIENT_P95

    result["smoothness_pass"] = smoothness_pass

    print()
    print("Bias-field spatial smoothness:")

    print(f"  Median |gradient|: {result['bias_gradient_median']:.6f}")

    print(f"  P95 |gradient|:    {result['bias_gradient_p95']:.6f}")

    print(f"  P99 |gradient|:    {result['bias_gradient_p99']:.6f}")

    print(f"  RMS |gradient|:    {result['bias_gradient_rms']:.6f}")

    print(f"  Max |gradient|:    {result['bias_gradient_max']:.6f}")

    print(f"Smoothness:           {'PASS' if smoothness_pass else 'REVIEW'}")

    # Smoothness remains a review criterion.
    # It does not automatically invalidate otherwise valid N4 output.

    # -------------------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------------------

    hard_checks = [
        geometry_pass,
        finite_pass,
        voxel_count_pass,
        reconstruction_pass,
        bias_range_pass,
    ]

    if all(hard_checks):
        result["status"] = "PASS"
        result["failure_reason"] = ""

    else:
        result["status"] = "FAIL"

        if not result["failure_reason"]:
            result["failure_reason"] = "One or more hard QC checks failed"

    print()
    print(f"OVERALL:              {result['status']}")

    return result


# =====================================================================
# CSV REPORT
# =====================================================================


def save_csv(
    results: list[dict],
) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted({key for result in results for key in result.keys()})

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("QC report saved to:")
    print(f"  {CSV_PATH}")


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run automated QC on precomputed IBSR-18 N4 images.")
    )

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help=("Run QC for one subject, e.g. IBSR_01."),
    )

    args = parser.parse_args()

    if args.subject:
        subjects = [args.subject]

    else:
        subjects = [f"IBSR_{i:02d}" for i in range(1, 19)]

    print("=" * 70)
    print("IBSR-18 N4 QUALITY CONTROL")
    print("=" * 70)

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Raw directory:")
    print(f"  {RAW_DIR}")

    print()
    print("N4 directory:")
    print(f"  {N4_DIR}")

    print()
    print("Bias-field directory:")
    print(f"  {BIAS_DIR}")

    print()
    print("QC report:")
    print(f"  {CSV_PATH}")

    print()
    print("Thresholds:")
    print(f"  Reconstruction P99 relative error <= {MAX_RECONSTRUCTION_REL_ERROR:.1e}")

    print(f"  Bias gradient P95 <= {MAX_BIAS_GRADIENT_P95:.3f} (review criterion)")

    print()
    print(f"Subjects to check: {len(subjects)}")

    results = []

    for subject in subjects:
        results.append(qc_subject(subject))

    save_csv(results)

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    passed = [result for result in results if result["status"] == "PASS"]

    failed = [result for result in results if result["status"] == "FAIL"]

    print()
    print("=" * 70)
    print("QC SUMMARY")
    print("=" * 70)

    print()
    print(f"Subjects checked:     {len(results)}")

    print(f"Passed:               {len(passed)}")

    print(f"Failed:               {len(failed)}")

    if failed:
        print()
        print("Subjects requiring attention:")

        for result in failed:
            print(
                f"  {result['subject']}: "
                f"{result.get('failure_reason', 'Unknown reason')}"
            )

    print()

    if not failed:
        print("ALL HARD QC CHECKS PASSED.")

        print()
        print("The precomputed N4 dataset is ready for Experiment 4 training.")

    else:
        print("QC FAILED for one or more subjects.")

        print("Do NOT start Experiment 4 until the failures have been investigated.")


if __name__ == "__main__":
    main()
