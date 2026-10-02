# Experiments

This document records the controlled experiments used to evaluate the IBSR-18 3D brain tissue segmentation pipeline.

The experiments investigate the following questions:

1. Does a residual 3D U-Net outperform a conventional 3D U-Net?
2. Does resampling MRI volumes to 1 mm isotropic spacing improve segmentation?
3. Does N4 MRI bias-field correction improve segmentation performance?
4. Does extending training beyond 100 epochs continue to improve validation performance?
5. Does increasing the loss weight assigned to CSF improve segmentation of this more challenging tissue class?
6. Does increasing sliding-window overlap improve inference performance without retraining the model?
7. Does conservative left-right flip test-time augmentation (TTA) improve inference performance without retraining the model?

All experiments use the same train/validation split and evaluation protocol unless explicitly stated otherwise.

---

# Dataset

The project uses the IBSR-18 T1-weighted brain MRI dataset.

## Splits

### Training

IBSR_01, IBSR_03, IBSR_04, IBSR_05, IBSR_06, IBSR_07, IBSR_08, IBSR_09, IBSR_16, IBSR_18

### Validation

IBSR_11, IBSR_12, IBSR_13, IBSR_14, IBSR_17

### Test

IBSR_02, IBSR_10, IBSR_15

The test split used by this project does not contain ground-truth segmentation labels. Therefore, quantitative model selection is performed exclusively on the validation set.

The test subjects are subsequently used for unseen-subject inference and qualitative generalization assessment.

## Segmentation labels

| Label | Tissue |
| ----: | ------ |
| 0 | Background |
| 1 | CSF |
| 2 | Gray matter (GM) |
| 3 | White matter (WM) |

The primary metric is the **Mean Foreground Dice**, defined as the average Dice score across CSF, GM, and WM:

```math
(Dice_CSF + Dice_GM + Dice_WM) / 3
```

Background Dice is reported for completeness but is excluded from the primary model-selection metric.

---

# Common Training Configuration

Unless otherwise specified, the experiments use:

- 3D U-Net architecture
- residual or conventional convolutional blocks as specified by each experiment
- native or explicitly configured target spacing
- `96 × 96 × 96` training patches
- batch size of 1
- Dice + cross-entropy loss
- PyTorch
- MONAI
- RAS reorientation
- intensity normalization
- foreground cropping
- randomized spatial/intensity augmentation
- validation-based checkpoint selection

The validation and inference pipelines do not use training-time augmentation.

The final training configuration uses native voxel spacing and does not apply N4 bias-field correction.

---

# Experiment 1 — Residual 3D U-Net, Native Spacing

## Objective

Establish the primary segmentation configuration using a residual 3D U-Net while preserving each subject's native voxel spacing.

## Configuration

| Parameter | Value |
| ------------------ | -------- |
| Architecture | 3D U-Net |
| Residual units | 2 |
| Voxel spacing | Native |
| Target spacing | None |
| N4 bias correction | Disabled |
| Runtime N4 | Disabled |
| Precomputed N4 | Disabled |

The preprocessing pipeline includes RAS reorientation, intensity normalization, foreground cropping, random 3D patch sampling, and spatial/intensity augmentation.

## Initial 100-epoch result

The initial training run used 100 epochs.

| Parameter | Value |
| -------------------- | ---------- |
| Training epochs | 100 |
| Best checkpoint | Epoch 99 |
| Mean foreground Dice | **0.8840** |

The validation results were:

| Class | Dice |
| ------------------- | ---------: |
| Background | 0.9718 |
| CSF | 0.8437 |
| GM | 0.9055 |
| WM | 0.9028 |
| **Mean foreground** | **0.8840** |

The result indicated that the model was still capable of improving with additional optimization, motivating longer training runs using the same architecture and preprocessing configuration.

---

# Training-Duration Study

Because the native-spacing residual U-Net continued to improve after 100 epochs, training was progressively extended to determine whether additional optimization improved validation performance.

No architecture or preprocessing change was introduced between these runs.

| Training duration | Best mean foreground Dice | Best epoch |
| ----------------: | ------------------------: | ---------: |
| 100 epochs | 0.8840 | 99 |
| 200 epochs | 0.9054 | 199 |
| 300 epochs | 0.9114 | 290 |
| **400 epochs** | **0.9185** | **391** |

## 100 → 200 epochs

Extending training from 100 to 200 epochs improved mean foreground Dice from:

```text
0.8840 → 0.9054
```

Absolute improvement:

```text
0.9054 - 0.8840 = 0.0214
```

## 200 → 300 epochs

Further extending training to 300 epochs improved the best validation score to:

```text
0.9114
```

Absolute improvement over the 200-epoch run:

```text
0.9114 - 0.9054 = 0.0060
```

## 300 → 400 epochs

The 400-epoch run reached a best validation mean foreground Dice of:

```text
0.9185
```

at epoch 391.

Absolute improvement over the 300-epoch run:

```text
0.9185 - 0.9114 = 0.0071
```

## Overall improvement

From the initial 100-epoch run to the final 400-epoch run:

```text
0.9185 - 0.8840 = 0.0345
```

Thus, extending training produced a **0.0345 absolute improvement in mean foreground Dice** without changing the architecture, voxel spacing, or preprocessing configuration.

---

# Final 400-Epoch Run

The final native-spacing residual U-Net was trained for 400 epochs.

The best validation checkpoint occurred at **epoch 391**:

```text
Epoch 391/400

train_loss = 0.2335
val_loss   = 0.2027

CSF Dice = 0.8938
GM Dice  = 0.9340
WM Dice  = 0.9276

Mean foreground Dice = 0.9185
```

The subsequent epochs did not exceed the epoch-391 result:

| Epoch | Mean foreground Dice |
| ----: | -------------------: |
| 392 | 0.9128 |
| 393 | 0.9134 |
| 394 | 0.9131 |
| 395 | 0.9131 |
| 396 | 0.9093 |
| 397 | 0.9085 |
| 398 | 0.9140 |
| 399 | 0.9057 |
| 400 | 0.9142 |

Therefore, **epoch 391 was selected as the final training checkpoint** rather than the final training epoch.

The checkpoint is stored at:

```text
outputs/checkpoints/best_model.pt
```

---

# Final Validation Results

The final epoch-391 checkpoint was independently evaluated on the five held-out validation subjects using the original 0.25 sliding-window overlap:

| Subject | CSF Dice | GM Dice | WM Dice |
| ------------- | ---------: | ---------: | ---------: |
| IBSR_11 | 0.8819 | 0.9338 | 0.9443 |
| IBSR_12 | 0.8886 | 0.9187 | 0.9267 |
| IBSR_13 | 0.8655 | 0.9326 | 0.9051 |
| IBSR_14 | 0.9113 | 0.9446 | 0.9374 |
| IBSR_17 | 0.9217 | 0.9404 | 0.9245 |
| **Aggregate** | **0.8938** | **0.9340** | **0.9276** |

Overall validation performance:

| Metric | Dice |
| ------------------- | ---------: |
| Background | 0.9803 |
| CSF | 0.8938 |
| GM | 0.9340 |
| WM | 0.9276 |
| **Mean foreground** | **0.9185** |

This represents an absolute improvement of:

| Metric | Initial 100 epochs | Final 400-epoch run | Improvement |
| ------------------- | -----------------: | ------------------: | ----------: |
| CSF | 0.8437 | 0.8938 | +0.0501 |
| GM | 0.9055 | 0.9340 | +0.0285 |
| WM | 0.9028 | 0.9276 | +0.0248 |
| **Mean foreground** | **0.8840** | **0.9185** | **+0.0345** |

The final model performs particularly strongly on GM and WM, while CSF remains the most challenging foreground class.

---

# Experiment 2 — Residual 3D U-Net, 1 mm Isotropic Spacing

## Objective

Evaluate whether resampling all subjects to a common 1 mm isotropic voxel spacing improves segmentation performance.

## Configuration

The model and training procedure were kept consistent with the native-spacing residual U-Net.

| Parameter | Value |
| ------------------- | ------------ |
| Architecture | 3D U-Net |
| Residual units | 2 |
| Voxel spacing | 1 × 1 × 1 mm |
| N4 bias correction | Disabled |
| Training epochs | 100 |
| Selected checkpoint | Epoch 99 |

## Validation results

| Class | Dice |
| ------------------- | ---------: |
| Background | 0.9813 |
| CSF | 0.8259 |
| GM | 0.9073 |
| WM | 0.8911 |
| **Mean foreground** | **0.8748** |

## Comparison with the initial native-spacing run

The 1 mm isotropic configuration achieved:

```text
0.8748
```

compared with:

```text
0.8840
```

for the 100-epoch native-spacing residual U-Net.

The difference was:

```text
0.8748 - 0.8840 = -0.0092
```

Thus, 1 mm isotropic resampling reduced mean foreground Dice by **0.0092** in this controlled comparison.

The comparison with the final 400-epoch native model is:

```text
0.9185 - 0.8748 = 0.0437
```

This latter comparison is descriptive rather than a controlled comparison because the training duration differs.

## Result

For this dataset and model configuration, 1 mm isotropic resampling did not improve segmentation performance.

The final pipeline therefore retains native voxel spacing.

---

# Experiment 3 — Conventional 3D U-Net, Native Spacing

## Objective

Measure the contribution of residual units by comparing the residual U-Net against an otherwise comparable conventional 3D U-Net.

## Configuration

| Parameter | Value |
| ------------------- | -------- |
| Architecture | 3D U-Net |
| Residual units | 0 |
| Voxel spacing | Native |
| N4 bias correction | Disabled |
| Training epochs | 100 |
| Selected checkpoint | Epoch 99 |

## Validation results

| Class | Dice |
| ------------------- | ---------: |
| Background | 0.9810 |
| CSF | 0.7730 |
| GM | 0.8985 |
| WM | 0.8764 |
| **Mean foreground** | **0.8493** |

## Comparison with the initial residual model

| Metric | Conventional U-Net | Residual U-Net | Difference |
| ------------------- | -----------------: | -------------: | ----------: |
| CSF | 0.7730 | 0.8437 | +0.0707 |
| GM | 0.8985 | 0.9055 | +0.0070 |
| WM | 0.8764 | 0.9028 | +0.0264 |
| **Mean foreground** | **0.8493** | **0.8840** | **+0.0347** |

Residual connections produced a substantial improvement in the 100-epoch comparison.

The largest class-specific difference occurred for CSF, while GM and WM also improved.

## Result

The comparison supports retaining residual units in the selected architecture.

The final model subsequently benefited further from extended training, reaching **0.9185 mean foreground Dice** at its best checkpoint after 400 epochs.

---

# Experiment 4 — Residual 3D U-Net with Precomputed N4 Correction

## Objective

Evaluate whether MRI bias-field correction improves tissue segmentation when applied as deterministic preprocessing.

## Motivation

T1-weighted MRI can contain low-frequency intensity inhomogeneity, commonly referred to as bias field.

N4 bias-field correction was therefore evaluated as a preprocessing intervention while keeping the model architecture and native voxel spacing unchanged.

## N4 preprocessing

N4 correction was performed once per subject and stored as precomputed volumes to avoid repeating the computationally expensive correction during every training epoch.

Final N4 configuration:

| Parameter | Value |
| ----------------------- | ------------------------------------------ |
| Implementation | SimpleITK N4BiasFieldCorrectionImageFilter |
| Shrink factor | 4 |
| Maximum iterations | `[50, 50, 50, 50]` |
| Convergence threshold | 0.001 |
| B-spline control points | `[4, 4, 4]` |
| Foreground mask | `image > 0` |
| Runtime N4 | Disabled |
| Precomputed N4 | Enabled |
| Target spacing | Native |

The corrected image was generated from the estimated multiplicative bias field:

```text
I_corrected = exp(log(max(I, ε)) - log(B))
```

with the original zero-valued background restored.

---

## N4 Preprocessing Quality Control

All **18 IBSR-18 subjects passed automated N4 preprocessing QC** before the segmentation experiment was started.

The QC process checked:

- image geometry preservation
- foreground voxel preservation
- finite image and bias-field values
- positive bias-field values
- reconstruction consistency
- spatial smoothness of the estimated bias field

The reconstruction check verified:

```text
I_N4(x) × B(x) ≈ I_raw(x)
```

using a P99 relative-error acceptance threshold of:

```text
1 × 10^-4
```

All subjects passed this criterion.

The estimated bias fields also remained below the predefined spatial-gradient review threshold.

This confirms that the finalized N4 preprocessing pipeline itself was technically valid before evaluating its effect on segmentation.

---

## Segmentation Configuration

| Parameter | Value |
| ------------------- | ----------------- |
| Architecture | Residual 3D U-Net |
| Residual units | 2 |
| Voxel spacing | Native |
| Precomputed N4 | Enabled |
| Runtime N4 | Disabled |
| Training epochs | 100 |
| Selected checkpoint | Best validation checkpoint |

## Validation Results

| Class | Dice |
| ------------------- | ---------: |
| Background | 0.9708 |
| CSF | 0.8408 |
| GM | 0.8984 |
| WM | 0.8939 |
| **Mean foreground** | **0.8777** |

## Comparison with the 100-epoch native/no-N4 model

| Metric | Native baseline | N4 corrected | Δ |
| ------------------- | --------------: | -----------: | ----------: |
| CSF | 0.8437 | 0.8408 | −0.0029 |
| GM | 0.9055 | 0.8984 | −0.0071 |
| WM | 0.9028 | 0.8939 | −0.0089 |
| **Mean foreground** | **0.8840** | **0.8777** | **−0.0063** |

N4 correction therefore changed mean foreground Dice by:

```text
0.8777 - 0.8840 = -0.0063
```

## Comparison with the final 400-epoch model

The final native/no-N4 model achieved:

```text
0.9185
```

while the N4 experiment achieved:

```text
0.8777
```

The difference is:

```text
0.9185 - 0.8777 = 0.0408
```

This comparison should be interpreted carefully because the N4 experiment used the shorter 100-epoch training schedule, whereas the final native model was trained for 400 epochs.

The controlled N4 comparison against the 100-epoch native baseline nevertheless shows that N4 did not provide a measurable benefit under the evaluated training configuration.

## Result

N4 correction did not improve segmentation performance.

Importantly, this result is **not attributed to a failure of the N4 preprocessing implementation**: all 18 subjects passed the independent preprocessing QC procedure.

For this dataset and model configuration, the simpler native/no-N4 preprocessing pipeline is therefore retained.

---

# Experiment 5 — CSF-Weighted Dice + Cross-Entropy Loss

## Objective

Evaluate whether explicitly increasing the loss contribution of CSF improves segmentation of the CSF class and consequently improves mean foreground Dice.

CSF was selected for this experiment because it remained the lowest-scoring foreground class in the final native-spacing model.

## Configuration

This was designed as a controlled training experiment. The final native-spacing residual U-Net configuration was retained, with only the class weighting changed.

| Parameter | Value |
| --------------- | ----------------------- |
| Architecture | Residual 3D U-Net |
| Residual units | 2 |
| Voxel spacing | Native |
| N4 | Disabled |
| Patch size | `96 × 96 × 96` |
| Batch size | 1 |
| Training epochs | 400 |
| Learning rate | `0.0001` |
| Weight decay | `0.00001` |
| Dice weight | 1.0 |
| CE weight | 1.0 |
| Class weights | `[1.0, 1.5, 1.0, 1.0]` |
| Class order | Background, CSF, GM, WM |
| Seed | 42 |

The same class weights were applied to both the Dice and cross-entropy components.

The experiment used a separate checkpoint directory:

```text
outputs/experiments/csf_weighted/checkpoints/best_model.pt
```

The original baseline checkpoint was therefore preserved.

## Validation results

The best validation mean foreground Dice for the CSF-weighted experiment was **0.9167**, reached at approximately epoch 371.

At that checkpoint:

| Class | Dice |
| ------------------- | ---------: |
| CSF | 0.8953 |
| GM | 0.9315 |
| WM | 0.9233 |
| **Mean foreground** | **0.9167** |

## Comparison with the unweighted final model

| Metric | Unweighted baseline | CSF weight 1.5 | Difference |
| ------------------- | ------------------: | -------------: | ----------: |
| CSF | 0.8938 | 0.8953 | +0.0015 |
| GM | 0.9340 | 0.9315 | −0.0025 |
| WM | 0.9276 | 0.9233 | −0.0043 |
| **Mean foreground** | **0.9185** | **0.9167** | **−0.0018** |

The CSF-weighted objective produced a small improvement in CSF Dice at its best checkpoint, but this was accompanied by lower GM and WM Dice.

The net effect was a reduction in mean foreground Dice from 0.9185 to 0.9167.

## Result

The evaluated CSF weighting of `[1.0, 1.5, 1.0, 1.0]` did not improve the overall segmentation performance.

The experiment suggests that simply increasing the optimization weight assigned to CSF was insufficient to produce a useful overall improvement under the tested training configuration.

The unweighted loss is therefore retained for the selected model.

This result is specific to the tested class-weight configuration and does not establish that all CSF-aware or boundary-aware loss formulations would be ineffective.

---

# Experiment 6 — Sliding-Window Inference Overlap

## Objective

Evaluate whether increasing sliding-window overlap improves validation segmentation without changing the trained model.

This is an inference-only experiment. No model parameters were retrained.

## Configuration

The original epoch-391 final checkpoint was evaluated using the same validation set, preprocessing, ROI size, and Gaussian blending. Only the sliding-window overlap was changed.

| Parameter | Overlap 0.25 | Overlap 0.50 |
| ------------------------- | -----------: | -----------: |
| Checkpoint | Epoch 391 | Epoch 391 |
| ROI size | `96³` | `96³` |
| Sliding-window batch size | 1 | 1 |
| Gaussian blending | Enabled | Enabled |
| Overlap | 0.25 | **0.50** |
| Training changes | None | None |

The checkpoint used for both evaluations was:

```text
outputs/checkpoints/best_model.pt
```

## Validation results

| Metric | Overlap 0.25 | Overlap 0.50 | Difference |
| ------------------- | -----------: | -----------: | ----------: |
| Background | 0.9803 | 0.9805 | +0.0002 |
| CSF | 0.8938 | 0.8942 | +0.0004 |
| GM | 0.9340 | 0.9344 | +0.0004 |
| WM | 0.9276 | 0.9280 | +0.0004 |
| **Mean foreground** | **0.9185** | **0.9188** | **+0.0003** |

The subject-level results at 0.50 overlap were:

| Subject | CSF Dice | GM Dice | WM Dice |
| ------------- | ---------: | ---------: | ---------: |
| IBSR_11 | 0.8819 | 0.9338 | 0.9443 |
| IBSR_12 | 0.8886 | 0.9187 | 0.9267 |
| IBSR_13 | 0.8655 | 0.9326 | 0.9051 |
| IBSR_14 | 0.9113 | 0.9446 | 0.9374 |
| IBSR_17 | 0.9235 | 0.9424 | 0.9263 |
| **Aggregate** | **0.8942** | **0.9344** | **0.9280** |

## Result

Increasing sliding-window overlap from 0.25 to 0.50 produced a small improvement in validation mean foreground Dice:

```text
0.9185 → 0.9188
```

The absolute improvement was:

```text
0.9188 - 0.9185 = 0.0003
```

Because the model checkpoint and all training-related settings remained unchanged, this difference represents an inference-level improvement rather than a change in learned model performance.

The 0.50 overlap setting is therefore retained as the current inference configuration.

The improvement is small and should not be interpreted as a substantial change in segmentation capability.

---

# Experiment 7 — Conservative Left-Right Flip TTA

## Objective

Evaluate whether a simple left-right flip test-time augmentation strategy improves validation performance without retraining the model.

This is an inference-only experiment.

## Configuration

The selected epoch-391 checkpoint was evaluated using the final native-space preprocessing and sliding-window configuration. TTA consisted of:

1. standard inference on the original volume;
2. left-right flip of the input;
3. inference on the flipped volume;
4. inverse flip of the resulting logits;
5. averaging of the original and restored flipped logits before `argmax`.

The evaluation used the same:

| Parameter | Value |
| ------------------------- | ---------------- |
| Checkpoint | Epoch 391 |
| Architecture | Residual 3D U-Net |
| Voxel spacing | Native |
| N4 | Disabled |
| ROI size | `96³` |
| Sliding-window batch size | 1 |
| Overlap | 0.50 |
| Gaussian blending | Enabled |
| TTA transform | Left-right flip |
| Model retraining | None |

## Validation results

| Metric | Baseline, no TTA | LR-flip TTA | Difference |
| ------------------- | ----------------: | -----------: | ----------: |
| Background | 0.9805 | 0.9810 | +0.0005 |
| CSF | 0.8942 | 0.8918 | −0.0024 |
| GM | 0.9344 | 0.9342 | −0.0002 |
| WM | 0.9280 | 0.9263 | −0.0017 |
| **Mean foreground** | **0.9188** | **0.9174** | **−0.0014** |

The TTA results by subject were:

| Subject | CSF Dice | GM Dice | WM Dice |
| ------------- | ---------: | ---------: | ---------: |
| IBSR_11 | 0.8770 | 0.9322 | 0.9416 |
| IBSR_12 | 0.8859 | 0.9184 | 0.9252 |
| IBSR_13 | 0.8618 | 0.9333 | 0.9046 |
| IBSR_14 | 0.9105 | 0.9445 | 0.9356 |
| IBSR_17 | 0.9237 | 0.9425 | 0.9243 |
| **Aggregate** | **0.8918** | **0.9342** | **0.9263** |

## Result

The left-right flip TTA reduced mean foreground Dice from:

```text
0.9188 → 0.9174
```

with an absolute change of:

```text
0.9174 - 0.9188 = -0.0014
```

The TTA experiment therefore did not improve validation performance under the evaluated configuration.

TTA is **not adopted** in the final inference pipeline.

This is a controlled negative inference result: it does not imply that other TTA strategies would necessarily be ineffective.

---

# Overall Comparison

The main controlled training experiments are:

| Experiment | Architecture | Spacing | N4 | Epochs | Mean FG Dice |
| ------------------------ | ------------------ | -------------- | ------ | ------: | -----------: |
| Native baseline | Residual U-Net | Native | No | 100 | 0.8840 |
| Native extended | Residual U-Net | Native | No | 200 | 0.9054 |
| Native extended | Residual U-Net | Native | No | 300 | 0.9114 |
| **Final training model** | **Residual U-Net** | **Native** | **No** | **400** | **0.9185** |
| 1 mm isotropic | Residual U-Net | 1 mm isotropic | No | 100 | 0.8748 |
| Conventional U-Net | Conventional U-Net | Native | No | 100 | 0.8493 |
| N4 preprocessing | Residual U-Net | Native | Yes | 100 | 0.8777 |
| CSF-weighted loss | Residual U-Net | Native | No | 400 | 0.9167 |

The inference-only experiments using the selected epoch-391 checkpoint were:

| Inference configuration | Mean FG Dice |
| ------------------------------- | -----------: |
| Sliding-window overlap 0.25 | 0.9185 |
| **Sliding-window overlap 0.50** | **0.9188** |
| LR-flip TTA, overlap 0.50 | 0.9174 |

These results distinguish **training/model-selection performance** from **inference-protocol effects**. The 0.9188 value is the current validation result under the selected inference configuration, while 0.9185 was the score used to select the epoch-391 checkpoint under the original 0.25-overlap evaluation protocol.

---

# Main Findings

## 1. Extended training substantially improved the native residual model

The same architecture and preprocessing configuration improved consistently as training was extended:

```text
0.8840 → 0.9054 → 0.9114 → 0.9185
```

for 100, 200, 300, and 400 epochs respectively.

The overall improvement from the initial 100-epoch run to the best 400-epoch checkpoint was:

```text
0.9185 - 0.8840 = 0.0345
```

This demonstrates that the initial 100-epoch result should be regarded as a baseline rather than the final performance of the native residual architecture.

## 2. Residual architecture improved performance

The conventional U-Net achieved:

```text
0.8493
```

while the comparable 100-epoch residual U-Net achieved:

```text
0.8840
```

The difference was:

```text
0.8840 - 0.8493 = 0.0347
```

The largest class-specific difference occurred for CSF, while GM and WM also improved.

## 3. Native spacing performed better than 1 mm isotropic resampling

The 100-epoch native residual U-Net achieved:

```text
0.8840
```

compared with:

```text
0.8748
```

after resampling to 1 mm isotropic spacing.

The native configuration therefore achieved:

```text
0.8840 - 0.8748 = 0.0092
```

higher mean foreground Dice in this controlled comparison.

## 4. N4 correction did not improve segmentation

The N4 configuration achieved:

```text
0.8777
```

compared with:

```text
0.8840
```

for the comparable 100-epoch native/no-N4 configuration.

N4 therefore changed mean foreground Dice by:

```text
0.8777 - 0.8840 = -0.0063
```

Although N4 did not improve segmentation, the preprocessing implementation itself was independently validated on all 18 subjects.

## 5. Increasing the CSF loss weight did not improve the overall result

The CSF-weighted experiment increased CSF Dice slightly at its best checkpoint:

```text
0.8938 → 0.8953
```

However, GM and WM Dice decreased:

```text
GM: 0.9340 → 0.9315
WM: 0.9276 → 0.9233
```

and mean foreground Dice decreased:

```text
0.9185 → 0.9167
```

Therefore, the tested CSF weighting did not provide a useful overall improvement.

## 6. Higher sliding-window overlap produced a small inference improvement

Increasing overlap from 0.25 to 0.50 while keeping the trained checkpoint unchanged resulted in:

```text
0.9185 → 0.9188
```

The absolute improvement was:

```text
0.0003
```

This is a small inference-level improvement rather than a change in learned model capability.

## 7. Left-right flip TTA did not improve the final inference result

The LR-flip TTA experiment changed mean foreground Dice from:

```text
0.9188 → 0.9174
```

an absolute decrease of:

```text
0.0014
```

Therefore, the final pipeline does not use this TTA configuration.

---

# Final Model and Inference Selection

The final trained model is:

- **Residual 3D U-Net**
- **2 residual units**
- **Native voxel spacing**
- **No N4 preprocessing**
- **400 training epochs**
- **Best checkpoint: epoch 391**
- **Mean foreground validation Dice: 0.9185 under the original 0.25-overlap protocol**

The selected checkpoint is:

```text
outputs/checkpoints/best_model.pt
```

For current inference, the configuration is:

- ROI size: `96 × 96 × 96`
- sliding-window batch size: 1
- overlap: **0.50**
- Gaussian blending
- native-space reconstruction
- **no TTA**

Under this inference configuration, the epoch-391 checkpoint achieved:

- **CSF Dice: 0.8942**
- **GM Dice: 0.9344**
- **WM Dice: 0.9280**
- **Mean foreground Dice: 0.9188**

The distinction between the two results is intentional:

- **0.9185** — best validation score used to select the trained epoch-391 checkpoint under the original 0.25-overlap evaluation protocol.
- **0.9188** — validation score obtained from the same checkpoint after increasing inference overlap from 0.25 to 0.50.

No model retraining was performed for the 0.50-overlap result.

The LR-flip TTA experiment was evaluated at the 0.50-overlap configuration but was not adopted because it reduced mean foreground Dice to 0.9174.

---

# Test-Set Inference

After selecting the final checkpoint, inference was performed on the three unseen test subjects:

- IBSR_02
- IBSR_10
- IBSR_15

The current final inference protocol is:

- ROI size: `96 × 96 × 96`
- sliding-window batch size: 1
- overlap: 0.50
- Gaussian blending
- native-space reconstruction
- no TTA

The three test subjects were successfully processed with this final inference configuration.

The test predictions were restored from the cropped inference representation to the original native NIfTI geometry.

## Native-space sanity checks

All three subjects passed the core prediction integrity checks:

- spatial-shape matching
- affine matching
- finite-value validation
- label-range validation
- non-empty foreground validation
- `uint8` output validation

The automated QC additionally records:

- native voxel spacing and voxel volume
- CSF/GM/WM voxel counts and derived volumes
- connected-component statistics
- slice continuity
- the number and fraction of predicted voxels outside the nonzero MRI foreground proxy

The MRI foreground containment measurement is treated as a **diagnostic proxy**, not as a true brain-mask test. Nonzero MRI voxels do not necessarily define the anatomical brain boundary.

Because ground-truth segmentations are unavailable for the test subjects, no quantitative test Dice is reported.

---

# Qualitative Test Assessment

Qualitative inspection was performed in axial, coronal, and sagittal planes.

Across the three test subjects, the predictions showed:

- plausible whole-brain coverage
- recognizable GM/WM organization
- major ventricular and CSF structures
- substantial cortical coverage
- no obvious large sliding-window seams
- no obvious large outside-brain false-positive regions

The recurring limitation was **coarse or irregular GM/WM boundary delineation**, particularly around the cortical ribbon and smaller sulcal CSF spaces.

Small disconnected or secondary components were also observed in some cases. These were treated as candidates for anatomical/QC inspection rather than automatically removed, because connected-component size alone does not establish whether a component is erroneous.

## IBSR_02

The overall segmentation is anatomically plausible, with recognizable cortical coverage and GM/WM organization.

The major ventricular spaces and large CSF regions are represented. The main limitation is relatively coarse and irregular GM/WM boundary delineation, particularly around cortical regions.

Small secondary components were observed in the automated topology analysis and were treated as diagnostic findings rather than automatically deleted.

## IBSR_10

IBSR_10 showed globally plausible tissue organization, including:

- continuous cortical coverage
- recognizable GM/WM organization
- well-formed ventricular CSF
- coherent central white matter
- reasonable whole-brain extent

The main qualitative limitation remained fine GM/WM boundary fragmentation at some cortical locations.

## IBSR_15

IBSR_15 was the most challenging qualitative case.

The prediction retained plausible large-scale organization, including cerebral tissue, central white matter, ventricular/CSF spaces, and inferior brain structures.

Automated component analysis identified a relatively large secondary WM component and smaller WM components. These were inspected in relation to neighboring tissue and MRI appearance. The large secondary component was spatially coherent and surrounded predominantly by GM, making automatic removal inappropriate without stronger anatomical evidence.

The smaller components remain candidates for further anatomical QC.

The recurring limitations in IBSR_15 were:

- coarser GM/WM boundaries
- simplified small CSF spaces
- localized cortical irregularities
- small disconnected components

No clear rectangular or grid-aligned structure characteristic of a sliding-window artifact was identified during the qualitative inspection.

Qualitative inspection alone cannot establish the exact anatomical correctness of every small component.

---

# Interpretation

The experiments support several conclusions within the evaluated IBSR-18 setup.

First, **training duration had a substantial effect** on the native-spacing residual U-Net. The model improved from 0.8840 at 100 epochs to 0.9185 at the best checkpoint of the 400-epoch run.

Second, **residual connections improved performance** compared with the conventional U-Net, providing a 0.0347 absolute improvement in mean foreground Dice in the 100-epoch comparison.

Third, **native voxel spacing was preferable to 1 mm isotropic resampling** in the evaluated controlled comparison.

Fourth, **N4 bias-field correction was technically successful but did not improve segmentation performance**. This distinction is important: a preprocessing method can be implemented correctly and pass independent QC without improving the downstream segmentation task.

Fifth, **the tested CSF-weighted loss did not improve overall segmentation performance**. Although CSF Dice increased slightly at the best weighted checkpoint, the accompanying reductions in GM and WM resulted in a lower mean foreground Dice.

Sixth, **increasing sliding-window overlap from 0.25 to 0.50 produced a small validation improvement** without changing the trained model. This supports using 0.50 overlap as the current inference configuration, while recognizing that the magnitude of the improvement is small.

Seventh, **the evaluated left-right flip TTA did not improve performance** and is therefore not included in the final inference protocol.

The final pipeline therefore uses the native/no-N4 residual U-Net trained for 400 epochs, checkpoint selection at epoch 391, 0.50 sliding-window overlap, and no TTA.

---

# Limitations

The experimental conclusions should be interpreted within the limitations of the dataset and evaluation setup:

1. IBSR-18 contains a relatively small number of subjects.
2. The validation set contains only five subjects.
3. The test split used here has no available ground-truth labels in this project configuration.
4. The validation set is therefore the primary quantitative evaluation set.
5. Qualitative test inspection revealed subject-dependent errors.
6. CSF remains more difficult to segment than GM and WM.
7. Some isolated or secondary predicted components occur in difficult cases.
8. The tested CSF-weighted loss used one specific class-weight configuration and does not rule out other boundary-aware or class-aware loss formulations.
9. The improvement from increased sliding-window overlap was small and was observed on the same five-subject validation set.
10. The LR-flip TTA experiment evaluated one conservative TTA strategy and does not rule out other augmentation strategies.
11. The current test predictions were regenerated using the final 0.50-overlap inference configuration, but the test set remains unlabeled and therefore cannot provide quantitative Dice.
12. The MRI foreground containment measurement used by automated QC is a nonzero-image proxy and should not be interpreted as a true anatomical brain mask.
13. The results should not be assumed to generalize directly to other MRI scanners, acquisition protocols, datasets, or clinical populations.
14. The final checkpoint was selected using the validation set; an external dataset would provide a stronger assessment of generalization.

---

# Final Conclusion

The final selected trained model is a **native-resolution residual 3D U-Net without N4 preprocessing**, trained for 400 epochs with checkpoint selection based on validation performance.

The progression of the native residual model was:

```text
100 epochs → 0.8840
200 epochs → 0.9054
300 epochs → 0.9114
400 epochs → 0.9185
```

The best checkpoint occurred at **epoch 391**, achieving under the original 0.25-overlap evaluation:

- **CSF Dice: 0.8938**
- **GM Dice: 0.9340**
- **WM Dice: 0.9276**
- **Mean foreground Dice: 0.9185**

The controlled experiments show that:

- residual architecture improved segmentation substantially;
- extended training produced a further improvement without changing the architecture;
- 1 mm isotropic resampling did not improve performance;
- N4 bias-field correction did not improve performance despite passing independent preprocessing QC;
- the tested CSF class weighting did not improve the overall validation result;
- increasing sliding-window overlap from 0.25 to 0.50 produced a small inference-level improvement from 0.9185 to 0.9188;
- the evaluated left-right flip TTA reduced the validation result from 0.9188 to 0.9174 and was therefore not adopted.

The current final inference configuration therefore uses:

```text
Residual 3D U-Net
Native spacing
No N4
Epoch-391 checkpoint
ROI: 96 × 96 × 96
Sliding-window overlap: 0.50
Gaussian blending
No TTA
```

Under this final validation inference protocol, the model achieved:

- **CSF Dice: 0.8942**
- **GM Dice: 0.9344**
- **WM Dice: 0.9280**
- **Mean foreground Dice: 0.9188**

The completed workflow additionally validates unseen test predictions through native-space reconstruction, automated structural sanity checks, connected-component and slice-continuity diagnostics, and qualitative inspection.

The resulting project provides both a reproducible **medical image segmentation experiment** and a modular **research software engineering implementation**.
