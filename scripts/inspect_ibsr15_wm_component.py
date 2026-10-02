from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MRI_PATH = PROJECT_ROOT / "data/raw/IBSR_15/IBSR_15.nii.gz"
PRED_PATH = PROJECT_ROOT / "outputs/predictions/test/IBSR_15/IBSR_15_pred.nii.gz"

OUTPUT_DIR = PROJECT_ROOT / "outputs/qc/ibsr15_wm_component"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load MRI and prediction
# ---------------------------------------------------------

mri_img = nib.load(MRI_PATH)
pred_img = nib.load(PRED_PATH)

mri = np.asarray(mri_img.dataobj)
pred = np.asarray(pred_img.dataobj)

# Remove singleton dimensions if present.
mri = np.squeeze(mri)
pred = np.squeeze(pred)

if mri.shape != pred.shape:
    raise ValueError(f"MRI and prediction shapes differ: {mri.shape} vs {pred.shape}")


# ---------------------------------------------------------
# Find WM connected components
# ---------------------------------------------------------

wm_mask = pred == 3

structure = np.ones((3, 3, 3), dtype=np.uint8)

components, num_components = ndimage.label(
    wm_mask,
    structure=structure,
)

component_sizes = np.bincount(components.ravel())

# Ignore background component.
component_sizes[0] = 0

component_ids = np.argsort(component_sizes)[::-1]

# Largest component = index 0
# Second largest = index 1
# 540-voxel component = identify explicitly
target_id = component_ids[2]

target_size = component_sizes[target_id]

print(f"Number of WM components: {num_components}")
print(f"Target component ID: {target_id}")
print(f"Target component size: {target_size} voxels")

if target_size != 540:
    print(
        "WARNING: The third-largest WM component is not 540 voxels. "
        "Check the component ordering."
    )

component_mask = components == target_id


# ---------------------------------------------------------
# Find component centroid
# ---------------------------------------------------------

coords = np.argwhere(component_mask)

centroid = coords.mean(axis=0)

print(f"Component centroid: {centroid}")

cx, cy, cz = np.round(centroid).astype(int)

print(f"Inspection center: x={cx}, y={cy}, z={cz}")


# ---------------------------------------------------------
# MRI display normalization
# ---------------------------------------------------------

nonzero = mri[mri > 0]

vmin = np.percentile(nonzero, 1)
vmax = np.percentile(nonzero, 99)

print(f"MRI display range: {vmin:.2f} - {vmax:.2f}")


# ---------------------------------------------------------
# Helper to create overlays
# ---------------------------------------------------------


def make_slice_figure(axis, index, output_path):
    """
    axis:
        0 = sagittal
        1 = coronal
        2 = axial
    """

    if axis == 0:
        mri_slice = mri[index, :, :]
        pred_slice = pred[index, :, :]
        comp_slice = component_mask[index, :, :]
        orientation = "Sagittal"
        xlabel = "z"
        ylabel = "y"

    elif axis == 1:
        mri_slice = mri[:, index, :]
        pred_slice = pred[:, index, :]
        comp_slice = component_mask[:, index, :]
        orientation = "Coronal"
        xlabel = "z"
        ylabel = "x"

    else:
        mri_slice = mri[:, :, index]
        pred_slice = pred[:, :, index]
        comp_slice = component_mask[:, :, index]
        orientation = "Axial"
        xlabel = "y"
        ylabel = "x"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -----------------------------------------------------
    # MRI
    # -----------------------------------------------------

    axes[0].imshow(
        np.rot90(mri_slice),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(f"{orientation} MRI\nslice {index}")

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    axes[1].imshow(
        np.rot90(mri_slice),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].imshow(
        np.rot90(pred_slice),
        alpha=0.35,
        vmin=0,
        vmax=3,
    )

    axes[1].set_title(f"{orientation} prediction\nslice {index}")

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

    # -----------------------------------------------------
    # Target WM component
    # -----------------------------------------------------

    axes[2].imshow(
        np.rot90(mri_slice),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    component_display = np.ma.masked_where(
        ~comp_slice,
        comp_slice,
    )

    axes[2].imshow(
        np.rot90(component_display),
        alpha=0.85,
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(f"{orientation} — 540-voxel WM component\nslice {index}")

    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Generate central views
# ---------------------------------------------------------

make_slice_figure(
    axis=0,
    index=cx,
    output_path=OUTPUT_DIR / "sagittal_center.png",
)

make_slice_figure(
    axis=1,
    index=cy,
    output_path=OUTPUT_DIR / "coronal_center.png",
)

make_slice_figure(
    axis=2,
    index=cz,
    output_path=OUTPUT_DIR / "axial_center.png",
)


# ---------------------------------------------------------
# Generate axial views through the whole component
# ---------------------------------------------------------

axial_indices = list(range(80, 127, 5))

for z in axial_indices:
    make_slice_figure(
        axis=2,
        index=z,
        output_path=OUTPUT_DIR / f"axial_z{z:03d}.png",
    )


print()
print("Done.")
print(f"Output directory: {OUTPUT_DIR}")
