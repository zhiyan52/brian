# Data

This directory contains the dataset configuration and split definitions used by the IBSR18 3D U-Net project.

The IBSR18 medical imaging data are **not included in this repository**. Users must obtain the dataset separately and comply with the dataset's original terms of use.

## Directory structure

After preparing the dataset locally, the `data/` directory is expected to have the following structure:

```text
data/
├── README.md
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── raw/
│   └── ...
└── processed/
    └── ...
```

### `splits/`

Contains the fixed subject-level dataset splits used throughout the project.

```text
train.txt
val.txt
test.txt
```

Each file contains one IBSR subject identifier per line.

### `raw/`

Contains the original IBSR18 NIfTI volumes and corresponding segmentation labels.

This directory is intentionally excluded from version control because the medical imaging data are not redistributed through this repository.

Example:

```text
data/raw/
├── IBSR_01/
│   ├── IBSR_01.nii.gz
│   └── IBSR_01_seg.nii.gz
├── IBSR_03/
│   ├── IBSR_03.nii.gz
│   └── IBSR_03_seg.nii.gz
└── ...
```

### `processed/`

Contains generated preprocessing outputs, including N4 bias-field-corrected volumes and other derived data.

These files are generated locally by the preprocessing pipeline and are not tracked by Git.

For example:

```text
data/processed/
└── n4/
    └── ...
```

## Dataset splits

The project uses fixed subject-level splits to ensure reproducible experiments.

| Split      | Subjects                                                                                 |
| ---------- | ---------------------------------------------------------------------------------------- |
| Training   | IBSR_01, IBSR_03, IBSR_04, IBSR_05, IBSR_06, IBSR_07, IBSR_08, IBSR_09, IBSR_16, IBSR_18 |
| Validation | IBSR_11, IBSR_12, IBSR_13, IBSR_14, IBSR_17                                              |
| Test       | IBSR_02, IBSR_10, IBSR_15                                                                |

The test subjects are kept separate from model development and are used only for final evaluation.

> **Note:** The exact split definitions used by the project are stored in `data/splits/`.

## Data format

The pipeline expects volumetric MRI data in NIfTI format:

```text
.nii
.nii.gz
```

Input images and segmentation labels should have compatible spatial dimensions and affine information.

The segmentation labels represent the tissue classes provided by the dataset.

## Preprocessing

The preprocessing pipeline performs dataset validation and prepares the images for 3D U-Net training.

The current workflow includes:

1. NIfTI metadata and spatial consistency checks
2. Subject-level split assignment
3. N4 bias-field correction
4. Intensity preprocessing
5. Conversion to the format expected by the training pipeline

N4 bias-field correction is precomputed rather than performed during every training iteration. This avoids repeating the computationally expensive correction step during training and makes experiments more reproducible.

The preprocessing scripts are located in:

```text
scripts/
```

## Reproducing the processed data

After obtaining the IBSR18 dataset and placing it under:

```text
data/raw/
```

run the data preparation pipeline from the project root.

For example:

```powershell
python scripts/prepare_data.py
```

The N4 preprocessing can then be generated with:

```powershell
python scripts/precompute_n4.py
```

Refer to the corresponding script help or project documentation for the available configuration options.

## Data availability

This repository contains **code and dataset split definitions only**. It does not redistribute the IBSR18 medical imaging dataset.

Users are responsible for obtaining the dataset through its appropriate source and following all applicable terms, conditions, and usage restrictions.

## Reproducibility

To reproduce the experiments:

1. Obtain the IBSR18 dataset separately.
2. Place the required files under `data/raw/`.
3. Use the provided subject splits in `data/splits/`.
4. Run the preprocessing pipeline.
5. Generate the N4-corrected volumes.
6. Use the provided configuration files for training and evaluation.

The dataset itself remains outside version control, while the preprocessing code, configurations, and split definitions are tracked in the repository.