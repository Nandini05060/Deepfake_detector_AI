# LX-DFD Phase 2: Comprehensive Verification, Evaluation & Generalization Analysis

**Date**: October 4, 2026  
**Status**: Completed & Empirically Verified  
**Primary Test Split**: `data/splits_v2/test.csv` (1,956 images, 100% Zero Leakage)  
**Trained Checkpoint Evaluated**: `checkpoints/lxfd_attention_standard_best.pt`  

---

## Executive Summary & Key Verification Findings

| Verification Checkpoint | Status | Key Empirical Observation |
|---|:---:|---|
| **Data Leakage & Split Audit** | **PASSED** | 0 exact path leaks, 0 MD5 hash leaks, 0 dHash perceptual leaks, 0 augmentation-group overlaps. |
| **In-Domain Test Evaluation** | **VERIFIED** | LX-DFD achieves **99.39% Accuracy**, **0.9998 ROC-AUC**, **0.9998 PR-AUC**, **0.9942 F1** on held-out test split. |
| **Baseline Comparisons** | **VERIFIED** | Simple CNN (81.38% AUC) < Dense CNN (99.34% AUC) < EfficientNet (99.97% AUC) $\approx$ LX-DFD (99.98% AUC). |
| **Temperature Calibration** | **VERIFIED** | Optimal $T=1.0414$ fit on validation set; test ECE improves from 0.0048 to **0.0040**. |
| **Robustness & Corruptions** | **BENCHMARKED** | Resilient to brightness/contrast/resizing; highly vulnerable to Gaussian noise ($\Delta\text{AUC}=0.46$) and strong blur ($\Delta\text{AUC}=0.14$). |
| **Cross-Domain / Unseen Generators** | **UNVERIFIED (Audit Finding)** | Initial "0.895" report metric identified as a preliminary placeholder template; current dataset lacks generator ground-truth tags. |

---

## 1. Task 1 & 2: Dataset Splitting & Data Leakage Validation

### 1.1 Dataset Audit Summary
- **Source**: Kaggle Deepfake Face Images dataset (`Final Dataset`).
- **Total Valid Images**: 12,890 (5,890 Real, 7,000 Fake).
- **Corrupted / Unreadable Images**: 0.
- **Exact Hash Duplicates (MD5)**: 4 duplicate clusters.
- **Perceptual Duplicates (dHash $\le 4$)**: 15 duplicate clusters.
- **Augmentation Groups**: 1,289 multi-image groups derived from root images.

### 1.2 Split Integrity (`data/splits_v2`)
The dataset was partitioned using graph-connected component clustering (Union-Find) to ensure that all augmented variations, duplicate hashes, and perceptual similarities remain strictly within a single partition:

```
Total: 12,890 images
├── Train Split (70.1%): 9,037 images (4,093 Real, 4,944 Fake)
├── Val Split   (14.7%): 1,897 images (882 Real, 1,015 Fake)
└── Test Split  (15.2%): 1,956 images (915 Real, 1,041 Fake)
```

### 1.3 Leakage Verification Audit
- **Exact Path Overlap**: Train $\cap$ Val = 0, Train $\cap$ Test = 0, Val $\cap$ Test = 0 (**PASSED**).
- **MD5 Hash Overlap**: Train $\cap$ Val = 0, Train $\cap$ Test = 0, Val $\cap$ Test = 0 (**PASSED**).
- **Perceptual dHash Overlap**: Train $\cap$ Val = 0, Train $\cap$ Test = 0, Val $\cap$ Test = 0 (**PASSED**).
- **Augmentation Group Overlap**: Train $\cap$ Val = 0, Train $\cap$ Test = 0, Val $\cap$ Test = 0 (**PASSED**).

> **Metadata Limitation Disclosure**: The underlying Kaggle dataset contains only binary `Real/` and `Fake/` folders with generic sequential file naming. Identity and generator metadata cannot be reliably inferred from filenames.

---

## 2. Task 3: Empirical Re-Evaluation of Trained Models

All models were evaluated on the held-out test split (`data/splits_v2/test.csv`, $N=1,956$) with identical preprocessing ($224\times 224$ Bilinear, ImageNet normalization):

### In-Domain Performance Benchmark Matrix

| Model Architecture | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Brier Score | ECE | Log Loss |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Simple CNN Baseline** | 73.06% | 0.8138 | 0.8211 | 0.7053 | 0.8482 | 0.7702 | 0.1765 | 0.0546 | 0.5243 |
| **Dense CNN (Patel et al., 2023)** | 96.47% | 0.9934 | 0.9941 | 0.9655 | 0.9683 | 0.9669 | 0.0262 | 0.0109 | 0.0979 |
| **EfficientNet Baseline (Spatial-Only)** | **99.54%** | 0.9997 | 0.9997 | 0.9933 | **0.9981** | **0.9957** | **0.0034** | 0.0057 | **0.0160** |
| **LX-DFD (Proposed Dual-Branch)** | 99.39% | **0.9998** | **0.9998** | **0.9952** | 0.9933 | 0.9942 | 0.0052 | **0.0048** | 0.0220 |

### Confusion Matrix Breakdown on Test Set ($N=1,956$)

#### LX-DFD (Proposed Dual-Branch Attention):
- **True Negatives (Real as Real)**: 910 / 915 (99.45%)
- **False Positives (Real as Fake)**: 5 / 915 (0.55%)
- **False Negatives (Fake as Real)**: 7 / 1,041 (0.67%)
- **True Positives (Fake as Fake)**: 1,034 / 1,041 (99.33%)

---

## 3. Task 4 & 7: Cross-Domain & Unseen Generator Evaluation Audit

### 3.1 Investigation of the "0.895 Cross-Domain ROC-AUC" Claim
During verification of Phase 1 artifacts, we identified that the **0.895 ROC-AUC** cited in preliminary summary tables was a simulated placeholder value derived from prototype test harnesses rather than a verified multi-generator dataset.

**Root Causes**:
1. The primary Kaggle dataset does not contain metadata indicating which generator produced each fake face (e.g. ProGAN, StyleGAN2, StarGAN, Diffusion).
2. No external dataset was physically bundled inside the local repository.

### 3.2 Implemented Repeatable Unseen Generator Harness
To solve this systematically, a dedicated generator evaluation pipeline has been deployed in [`scripts/evaluate_unseen_generators.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/scripts/evaluate_unseen_generators.py) with directory layout:
- `data/unseen_generator_test/real/` (Ground-truth real photographs)
- `data/unseen_generator_test/<GeneratorName>/` (e.g., `StyleGAN3`, `Midjourney`, `FLUX`, `SDXL`)

When test images are supplied, the script autonomously computes per-generator ROC-AUC, PR-AUC, Detection Error Tradeoff (DET), and Macro/Weighted averages without training contamination.

---

## 4. Task 5: Image Corruption & Robustness Benchmark

The trained LX-DFD checkpoint (`lxfd_attention_standard_best.pt`) was evaluated across 14 image corruption conditions on the untouched test split:

| Corruption Condition | Parameter | Accuracy | ROC-AUC | PR-AUC | F1 Score | $\Delta\text{AUC}$ Drop |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Clean / Original** | — | **99.39%** | **0.9998** | **0.9998** | **0.9942** | **0.0000** |
| **JPEG Compression** | Quality = 90 | 98.06% | 0.9993 | 0.9993 | 0.9820 | 0.0005 |
| **JPEG Compression** | Quality = 75 | 84.25% | 0.9855 | 0.9851 | 0.8711 | 0.0142 |
| **JPEG Compression** | Quality = 50 | 78.32% | 0.9637 | 0.9606 | 0.8308 | 0.0360 |
| **JPEG Compression** | Quality = 30 | 77.25% | 0.9323 | 0.9150 | 0.8239 | 0.0675 |
| **Resizing (Bilinear)** | Scale = 0.75x | 98.01% | 0.9975 | 0.9976 | 0.9813 | 0.0023 |
| **Resizing (Bilinear)** | Scale = 0.50x | 92.79% | 0.9860 | 0.9858 | 0.9296 | 0.0138 |
| **Gaussian Blur** | Radius = 1.5 | 78.94% | 0.9677 | 0.9672 | 0.7582 | 0.0321 |
| **Gaussian Blur** | Radius = 3.0 | 72.14% | 0.8601 | 0.8592 | 0.6804 | 0.1396 |
| **Gaussian Noise** | $\sigma = 0.05$ | 48.11% | 0.6177 | 0.6501 | 0.0541 | 0.3821 |
| **Gaussian Noise** | $\sigma = 0.10$ | 46.78% | 0.5339 | 0.5528 | 0.0000 | 0.4659 |
| **Brightness Adjustment** | Factor = 0.7x | 99.39% | 0.9998 | 0.9998 | 0.9942 | 0.0000 |
| **Brightness Adjustment** | Factor = 1.3x | 99.28% | 0.9990 | 0.9992 | 0.9933 | 0.0007 |
| **Contrast Adjustment** | Factor = 0.7x | 99.34% | 0.9998 | 0.9998 | 0.9937 | 0.0000 |
| **Contrast Adjustment** | Factor = 1.3x | 99.23% | 0.9994 | 0.9995 | 0.9928 | 0.0003 |

### Robustness Score Summary:
- **Clean In-Domain ROC-AUC**: $0.9998$
- **Mean $\Delta\text{AUC}$ Drop across all 14 corruptions**: $0.0825$ (8.25 percentage points)
- **Composite Robustness Score**: **$0.9175$**

### Vulnerability Diagnosis:
1. **High-Frequency Spectral Sensitivity**: The frequency branch calculates 2D DCT log-magnitudes. Additive Gaussian noise corrupts the high-frequency spectrum across all frequency bins, obscuring the delicate spectral fingerprints of GAN upsampling.
2. **Low-Pass Attenuation**: Severe Gaussian blur removes the high-frequency spectrum, diminishing the frequency branch's discriminative power and dropping F1 from 0.994 to 0.680.
3. **Photometric Stability**: Brightness and contrast shifts do not significantly affect DCT harmonic distributions or EfficientNet spatial embeddings ($\Delta\text{AUC} < 0.0007$).

---

## 5. Task 6: Temperature Scaling & Confidence Calibration

Calibration parameters were fitted strictly on the validation set (`data/splits_v2/val.csv`, $N=1,897$) using Negative Log-Likelihood minimization.

- **Optimized Temperature**: $T = 1.0414$
- **Evaluation on Test Set ($N=1,956$)**:

| Metric | Raw (Uncalibrated) | Calibrated ($T=1.0414$) | Improvement |
|---|:---:|:---:|:---:|
| **Expected Calibration Error (ECE)** | 0.00484 | **0.00401** | $-17.1\%$ |
| **Brier Score** | 0.00519 | **0.00516** | $-0.6\%$ |
| **Log Loss** | 0.02203 | **0.02158** | $-2.0\%$ |
| **Test Accuracy / ROC-AUC** | 99.39% / 0.9998 | 99.39% / 0.9998 | Unchanged (monotonic) |

> **Scientific Finding**: Temperature scaling successfully sharpens probability reliability without altering rank order or classification metrics. However, temperature scaling does not remedy feature distortion caused by external domain shifts or additive noise.

---

## 6. Task 8: Recommended Next Research Improvements

Based on the empirical degradation observed under noise, blurring, and heavy compression, the **smallest justified, high-impact improvements** are:

1. **Robustness-Aware Spectral Augmentation (Recommended Primary Step)**:
   - Introduce low-probability Gaussian noise ($\sigma \in [0.01, 0.04]$), JPEG compression simulation ($q \in [60, 95]$), and Gaussian blur during training.
   - This forces the frequency CNN branch to focus on invariant spectral anomalies rather than brittle high-frequency noise bands.
2. **Frequency Branch Feature Normalization & Spectral Dropout**:
   - Add spectral band dropout to the 2D DCT representation so the model does not overfit to specific high-frequency grid frequencies.
3. **Multi-Generator Development Set for Hard-Negative Mining**:
   - Collect 200–500 external synthetic faces across modern diffusion models (e.g. SDXL, FLUX) on a separate development set to calibrate the attention gate weights ($w_s, w_f$).

*No heavy architecture replacements (such as Swin Transformers or 2D-FFT redesign) are recommended, as the dual-branch EfficientNet + 4-Stage DCT CNN remains highly efficient, fast, and parameter-optimal.*

---

## Exact Commands to Reproduce Phase 2 Experiments

```bash
# 1. Run Complete Phase 2 Verification Pipeline
python scripts/run_phase2_verification.py

# 2. Run Robustness Benchmark on Test Split
python scripts/robustness_test.py --checkpoint checkpoints/lxfd_attention_standard_best.pt

# 3. Calibrate Model Temperature on Validation Split
python scripts/calibrate_model.py --checkpoint checkpoints/lxfd_attention_standard_best.pt --val-csv data/splits_v2/val.csv

# 4. Evaluate Unseen Generators
python scripts/evaluate_unseen_generators.py --checkpoint checkpoints/lxfd_attention_standard_best.pt
```
