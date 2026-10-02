from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MRI_PATH = PROJECT_ROOT / "data/raw/IBSR_15/IBSR_15.nii.gz"
PRED_PATH = PROJECT_ROOT / "outputs/predictions/test/IBSR_15/IBSR_15_pred.nii.gz"

OUTPUT_DIR = PROJECT_ROOT / "outputs/qc/ibsr15_csf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

mri_img = nib.load(MRI_PATH)
pred_img = nib.load(PRED_PATH)

mri = np.squeeze(np.asarray(mri_img.dataobj))
pred = np.squeeze(np.asarray(pred_img.dataobj))

if mri.shape != pred.shape:
    raise ValueError(f"MRI and prediction shapes differ: {mri.shape} vs {pred.shape}")


# ---------------------------------------------------------
# Find CSF components
# ---------------------------------------------------------

csf_mask = pred == 1

structure = np.ones((3, 3, 3), dtype=np.uint8)

components, num_components = ndimage.label(
    csf_mask,
    structure=structure,
)

sizes = np.bincount(components.ravel())

# Ignore background
sizes[0] = 0

component_ids = np.argsort(sizes)[::-1]

print("=" * 70)
print("IBSR_15 CSF COMPONENT ANALYSIS")
print("=" * 70)

print(f"Total CSF voxels: {csf_mask.sum()}")
print(f"Number of CSF components: {num_components}")
print()

print(f"{'Rank':<6}{'ID':<8}{'Voxels':<10}{'Percent':<10}{'Centroid':<30}")

print("-" * 70)


component_info = []

for rank, component_id in enumerate(component_ids, start=1):
    size = sizes[component_id]

    if size == 0:
        continue

    coords = np.argwhere(components == component_id)

    centroid = coords.mean(axis=0)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)

    percent = 100.0 * size / csf_mask.sum()

    component_info.append(
        {
            "rank": rank,
            "id": component_id,
            "size": size,
            "percent": percent,
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }
    )

    print(
        f"{rank:<6}{component_id:<8}{size:<10}{percent:<10.2f}{np.round(centroid, 1)}"
    )

print("=" * 70)
print()


# ---------------------------------------------------------
# MRI intensity range
# ---------------------------------------------------------

nonzero = mri[mri > 0]

vmin = np.percentile(nonzero, 1)
vmax = np.percentile(nonzero, 99)


# ---------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------


def save_component_views(component_info, name):

    component_id = component_info["id"]
    centroid = component_info["centroid"]

    cx, cy, cz = np.round(centroid).astype(int)

    mask = components == component_id

    views = [
        ("sagittal", 0, cx),
        ("coronal", 1, cy),
        ("axial", 2, cz),
    ]

    for orientation, axis, index in views:
        if axis == 0:
            mri_slice = mri[index, :, :]
            pred_slice = pred[index, :, :]
            mask_slice = mask[index, :, :]

        elif axis == 1:
            mri_slice = mri[:, index, :]
            pred_slice = pred[:, index, :]
            mask_slice = mask[:, index, :]

        else:
            mri_slice = mri[:, :, index]
            pred_slice = pred[:, :, index]
            mask_slice = mask[:, :, index]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        # MRI
        axes[0].imshow(
            np.rot90(mri_slice),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        axes[0].set_title(f"{orientation.capitalize()} MRI\nslice {index}")

        # Full prediction
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

        axes[1].set_title(f"{orientation.capitalize()} prediction\nslice {index}")

        # Individual CSF component
        axes[2].imshow(
            np.rot90(mri_slice),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        component_display = np.ma.masked_where(
            ~mask_slice,
            mask_slice,
        )

        axes[2].imshow(
            np.rot90(component_display),
            alpha=0.9,
            vmin=0,
            vmax=1,
        )

        axes[2].set_title(f"{orientation.capitalize()} CSF component\n{name}")

        for ax in axes:
            ax.set_xlabel("voxel")
            ax.set_ylabel("voxel")

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{name}_{orientation}_slice{index:03d}.png"

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Visualize largest CSF components
# ---------------------------------------------------------

for info in component_info[:3]:
    rank = info["rank"]

    save_component_views(
        info,
        f"csf_component_{rank}",
    )


# ---------------------------------------------------------
# Also create total-CSF views at selected anatomical levels
# ---------------------------------------------------------


def save_total_csf_view(axis, index, name):

    if axis == 0:
        mri_slice = mri[index, :, :]
        csf_slice = csf_mask[index, :, :]

    elif axis == 1:
        mri_slice = mri[:, index, :]
        csf_slice = csf_mask[:, index, :]

    else:
        mri_slice = mri[:, :, index]
        csf_slice = csf_mask[:, :, index]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(
        np.rot90(mri_slice),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(f"{name.capitalize()} MRI\nslice {index}")

    axes[1].imshow(
        np.rot90(mri_slice),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    csf_display = np.ma.masked_where(
        ~csf_slice,
        csf_slice,
    )

    axes[1].imshow(
        np.rot90(csf_display),
        alpha=0.85,
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(f"{name.capitalize()} — all predicted CSF\nslice {index}")

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"total_csf_{name}_slice{index:03d}.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# Anatomically useful levels
save_total_csf_view(
    axis=0,
    index=114,
    name="sagittal",
)

save_total_csf_view(
    axis=1,
    index=64,
    name="coronal",
)

for z in [80, 90, 100, 110, 120, 130]:
    save_total_csf_view(
        axis=2,
        index=z,
        name="axial",
    )


print()
print("CSF inspection figures written to:")
print(OUTPUT_DIR)
