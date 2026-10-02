# Architecture

## Overview

`ibsr18-3d-unet` is a modular 3D medical image segmentation pipeline for brain tissue segmentation on the IBSR-18 T1-weighted MRI dataset.

The final baseline uses a **residual 3D U-Net** trained at native image spacing, without N4 bias-field correction, followed by Gaussian-blended sliding-window inference.

The segmentation task contains four classes:

| Label | Tissue |
|---:|---|
| 0 | Background |
| 1 | CSF |
| 2 | Gray matter (GM) |
| 3 | White matter (WM) |

---

## High-Level Pipeline

```text
IBSR-18 T1 MRI
      │
      ▼
Data preparation / fixed subject splits
      │
      ▼
Load NIfTI → channel-first → RAS → native spacing
      │
      ▼
Intensity normalization → foreground crop
      │
      ▼
96³ training patches
      │
      ▼
Residual 3D U-Net
      │
      ▼
4-class logits
      │
      ├── Training: Dice + Cross-Entropy
      │
      └── Inference: Gaussian-blended sliding windows
                         │
                         ▼
                  4-class segmentation
                         │
                         ▼
                 Restore native space
                         │
                         ▼
                  uint8 NIfTI output
```

# 1. Repository Structure

```text
ibsr18-3d-unet/
├── .github/workflows/
│   ├── ci.yml
│   └── lint.yml
├── configs/
│   ├── train.yaml
│   └── finetune.yaml
├── data/README.md
├── docs/
│   ├── architecture.md
│   └── experiments.md
├── scripts/
│   ├── analyze_labels.py
│   ├── check_test_predictions.py
│   ├── evaluate.py
│   ├── precompute_n4.py
│   ├── predict.py
│   ├── predict_test.py
│   ├── prepare_data.py
│   ├── qc_n4.py
│   ├── smoke_test_experiment4.py
│   ├── smoke_test_n4.py
│   ├── train.py
│   ├── visualize_predictions.py
│   └── visualize_test_predictions.py
├── src/ibsr_unet/
│   ├── config/schema.py
│   ├── data/
│   │   ├── datasets.py
│   │   ├── datamodule.py
│   │   ├── splits.py
│   │   └── transforms.py
│   ├── models/unet.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── scheduler.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── predictor.py
│   │   └── tta.py
│   ├── visualization/plots.py
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── reproducibility.py
├── tests/
│   ├── test_dataset.py
│   ├── test_inference.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_transforms.py
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
└── pre-commit-config.yaml
```

# 2. Dataset and Splits

The project uses skull-stripped IBSR-18 T1-weighted MRI volumes stored as NIfTI files. Native volumes are typically `256 × 128 × 256`, with subject-dependent voxel spacing.

### Training

```text
IBSR_01, IBSR_03, IBSR_04, IBSR_05, IBSR_06,
IBSR_07, IBSR_08, IBSR_09, IBSR_16, IBSR_18
```

### Validation

```text
IBSR_11, IBSR_12, IBSR_13, IBSR_14, IBSR_17
```

### Test

```text
IBSR_02, IBSR_10, IBSR_15
```

The test subjects do not have ground-truth labels available to the pipeline, so quantitative test Dice scores are not reported.

# 3. Preprocessing

Implemented in `src/ibsr_unet/data/transforms.py`.

The final deterministic preprocessing sequence is:

```text
NIfTI
  → Load
  → Ensure single channel-first
  → RAS orientation
  → Native spacing
  → Nonzero intensity normalization
  → Foreground crop
  → Tensor conversion
```

The model does **not** use isotropic 1 mm resampling in the final baseline.

The input channel conversion is:

```text
[D, H, W]    → [1, D, H, W]
[D, H, W, 1] → [1, D, H, W]
```

# 4. Optional N4 Bias-Field Correction

N4 bias-field correction is implemented as a separate controlled experiment. Precomputed outputs are stored under:

```text
data/processed/n4/
```

The final baseline uses:

```yaml
use_precomputed_n4: false
use_n4_bias_correction: false
```

Therefore, N4 is **not part of the selected final model**.

# 5. Model Architecture

The model is implemented in `src/ibsr_unet/models/unet.py`.

The final model is a **residual 3D U-Net** using:

```yaml
model:
  name: "3d_unet"
  in_channels: 1
  out_channels: 4
  channels: [16, 32, 64, 128, 256]
  strides: [2, 2, 2, 2]
  num_res_units: 2
```

Input:

```text
[B, 1, D, H, W]
```

Output:

```text
[B, 4, D, H, W]
```

Output channels:

```text
0 → Background
1 → CSF
2 → GM
3 → WM
```

The network produces raw logits.

Conceptually:

```text
Input
  │
  ├── Encoder: 16
  ├── Encoder: 32
  ├── Encoder: 64
  ├── Encoder: 128
  ├── Bottleneck: 256
  │
  ├── Decoder + skip: 128
  ├── Decoder + skip: 64
  ├── Decoder + skip: 32
  └── Decoder + skip: 16
          │
          ▼
     4-class logits
```

Each level uses two residual units.

# 6. Patch-Based Training

Training uses `96 × 96 × 96` 3D patches:

```yaml
patch_size: [96, 96, 96]
```

Final training settings include:

```yaml
batch_size: 1
num_samples: 1
```

Patch-based training allows the native-resolution volumes to be processed within available GPU memory.

# 7. Loss Function

The final baseline uses equal-weight Dice and cross-entropy losses:

```yaml
loss:
  name: "dice_ce"
  dice_weight: 1.0
  ce_weight: 1.0
  include_background: true
  class_weights: null
```

Conceptually:

```text
Loss = Dice Loss + Cross-Entropy Loss
```

Class weighting is supported but disabled in the final baseline.

A controlled CSF-weighted experiment was evaluated separately and was not selected.

# 8. Training

Final training configuration:

```yaml
epochs: 400
batch_size: 1
learning_rate: 0.0001
weight_decay: 0.00001
seed: 42
```

The selected checkpoint is the best validation checkpoint rather than simply the final epoch.

```text
Best checkpoint epoch: 391
Checkpoint:
outputs/checkpoints/best_model.pt
```

# 9. Inference

Implemented in `src/ibsr_unet/inference/predictor.py`.

Final inference configuration:

```yaml
inference:
  roi_size: [96, 96, 96]
  sw_batch_size: 1
  overlap: 0.50
```

Inference flow:

```text
Full MRI
  │
  ▼
Foreground crop
  │
  ▼
96³ sliding windows
  │
  ▼
Residual 3D U-Net
  │
  ▼
Window logits
  │
  ▼
Gaussian-weighted blending
  │
  ▼
Full-volume logits
  │
  ▼
Argmax
  │
  ▼
4-class segmentation
```

Gaussian blending is used to reduce potential discontinuities between sliding windows.

# 10. Test Prediction Restoration

Implemented in `scripts/predict_test.py`.

Predictions are generated in transformed/cropped space and restored to native spatial extent.

Final test subjects:

```text
IBSR_02
IBSR_10
IBSR_15
```

Native output dimensions:

```text
256 × 128 × 256
```

Predictions preserve the MRI affine and are saved as `uint8` NIfTI files:

```text
outputs/predictions/test/<SUBJECT>/<SUBJECT>_pred.nii.gz
```

# 11. Evaluation

Implemented in:

```text
src/ibsr_unet/evaluation/metrics.py
src/ibsr_unet/evaluation/evaluator.py
scripts/evaluate.py
```

The primary quantitative metric is Dice similarity coefficient:

```text
Dice_c = 2 |P_c ∩ G_c| / (|P_c| + |G_c|)
```

The main model-comparison metric is mean foreground Dice across:

```text
CSF
GM
WM
```

Background Dice is reported separately.

# 12. Final Validation Result

The selected model and inference protocol are:

```text
Residual 3D U-Net
Native spacing
No N4
400 training epochs
Best checkpoint: epoch 391
ROI: 96³
Sliding-window overlap: 0.50
Gaussian blending
No TTA
```

Validation results:

| Class | Dice |
|---|---:|
| Background | 0.9805 |
| CSF | 0.8942 |
| GM | 0.9344 |
| WM | 0.9280 |
| **Mean foreground** | **0.9188** |

The reported `0.9188` is the verified validation mean foreground Dice using the final selected checkpoint with 0.50 sliding-window overlap.

# 13. Test-Time Augmentation

A conservative left-right flip TTA implementation is available in:

```text
src/ibsr_unet/inference/tta.py
```

TTA averages predictions from the original and left-right flipped inputs after inverse transformation.

The controlled TTA experiment produced:

```text
Mean foreground Dice: 0.9174
```

versus:

```text
Standard inference: 0.9188
```

Therefore, TTA is retained as a reproducible experiment but is **not enabled in the final baseline**.

# 14. Test Prediction Quality Control

Because the test subjects do not have ground-truth labels, test QC focuses on spatial, numerical, and structural integrity rather than segmentation accuracy.

The main script is:

```text
scripts/check_test_predictions.py
```

It checks:

### Spatial integrity

- prediction file exists
- prediction shape matches the MRI
- prediction affine matches the MRI

### Numerical integrity

- finite prediction values
- labels restricted to `{0, 1, 2, 3}`
- non-empty foreground
- `uint8` output datatype

### Structural diagnostics

- MRI foreground containment
- native voxel spacing
- CSF/GM/WM volumes
- 26-connected components
- slice continuity along x, y, and z

Results are written to:

```text
outputs/qc/test_prediction_qc.csv
```

These checks are diagnostic and do not establish anatomical correctness.

# 15. Anatomical QC Philosophy

Connected-component statistics and MRI-foreground containment are treated as **diagnostic signals**, not automatic anatomical judgments.

A disconnected tissue component is not automatically an error. For example, anatomically meaningful structures can appear as separate connected components in a voxel-wise segmentation.

Likewise, voxels outside the nonzero MRI foreground are not automatically false-positive brain predictions because the nonzero MRI region is only a conservative image-foreground proxy.

Potentially unusual components should therefore be inspected against the underlying MRI before post-processing or removal is considered.

No connected-component filtering is automatically applied in the final inference pipeline.

# 16. Qualitative Visualization

Visualization utilities are provided through:

```text
scripts/visualize_predictions.py
scripts/visualize_test_predictions.py
src/ibsr_unet/visualization/plots.py
```

Inspection can include:

- axial views
- coronal views
- sagittal views
- MRI/prediction overlays
- individual tissue classes
- connected components

For unlabeled test subjects, qualitative inspection focuses on:

- whole-brain coverage
- cortical coverage
- ventricular and CSF structures
- GM/WM organization
- small disconnected components
- holes and islands
- possible sliding-window artifacts
- anatomical plausibility across multiple planes

# 17. Reproducibility

The main configuration is:

```text
configs/train.yaml
```

Final baseline:

```yaml
seed: 42

data:
  root_dir: "data/raw"
  splits_dir: "data/splits"
  spacing: null
  n4_dir: "data/processed/n4"
  use_precomputed_n4: false
  use_n4_bias_correction: false
  patch_size: [96, 96, 96]

training:
  epochs: 400
  batch_size: 1
  num_samples: 1
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 0
  pin_memory: true

model:
  name: "3d_unet"
  in_channels: 1
  out_channels: 4
  channels: [16, 32, 64, 128, 256]
  strides: [2, 2, 2, 2]
  num_res_units: 2

loss:
  name: "dice_ce"
  dice_weight: 1.0
  ce_weight: 1.0
  include_background: true
  class_weights: null

inference:
  roi_size: [96, 96, 96]
  sw_batch_size: 1
  overlap: 0.50

output:
  checkpoint_dir: "outputs/checkpoints"
```

Reproducibility helpers are implemented in:

```text
src/ibsr_unet/utils/reproducibility.py
```

# 18. Software Quality

The repository includes:

```text
pyproject.toml
pre-commit-config.yaml
Dockerfile
Makefile
.github/workflows/ci.yml
.github/workflows/lint.yml
```

The automated test suite covers:

- data transforms
- dataset handling
- model construction
- loss functions
- metrics
- inference

Current test status:

```text
14 passed
```

The tests provide regression coverage for the core research pipeline while experiment-specific scripts remain separate.

# 19. Design Principles

### Configuration over hard-coded experiments

Training and inference parameters are defined through configuration files wherever practical.

### Modular components

Data, model, training, evaluation, inference, and visualization are separated into dedicated modules.

### Reproducible experiments

Subject splits, seeds, preprocessing choices, model parameters, and inference settings are explicitly recorded.

### Controlled experimentation

Alternative approaches such as isotropic resampling, N4 correction, CSF-weighted loss, and TTA were evaluated separately rather than silently incorporated into the final model.

### No unjustified post-processing

Connected-component filtering or anatomical cleanup is not automatically applied without evidence that it improves segmentation quality.

### Separate quantitative and qualitative evaluation

Validation subjects with ground truth are evaluated quantitatively. Unlabeled test predictions are assessed using structural QC and qualitative anatomical inspection.

# 20. Final Inference Configuration

```text
Task
└── 3D brain tissue segmentation

Dataset
└── IBSR-18

Input
└── Skull-stripped T1 MRI

Classes
├── 0 Background
├── 1 CSF
├── 2 GM
└── 3 WM

Preprocessing
├── RAS orientation
├── Native spacing
├── Nonzero intensity normalization
├── Foreground cropping
└── No N4

Model
└── Residual 3D U-Net
    ├── Channels: 16, 32, 64, 128, 256
    ├── Strides: 2, 2, 2, 2
    └── Residual units: 2

Training
├── Patch size: 96³
├── Batch size: 1
├── Epochs: 400
├── Learning rate: 1e-4
├── Weight decay: 1e-5
└── Seed: 42

Loss
└── Dice + Cross Entropy

Checkpoint
└── Best validation checkpoint
    └── Epoch 391

Inference
├── ROI: 96³
├── Sliding-window inference
├── Gaussian blending
├── Overlap: 0.50
└── No TTA

Output
└── Native-space uint8 NIfTI segmentation

Validation
└── Mean foreground Dice: 0.9188
```

## Summary

The final pipeline is intentionally simple and reproducible:

**native-resolution MRI → deterministic preprocessing → residual 3D U-Net → patch-based training → Gaussian-blended sliding-window inference → native-space segmentation → quantitative and qualitative QC.**

Alternative preprocessing, loss weighting, and inference strategies remain documented as controlled experiments, while the final baseline remains fixed and reproducible.
