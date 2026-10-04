# Phase 3 Baseline Verification & Inspection Report

**Date**: October 4, 2026  
**Primary Split Evaluated**: `data/splits_v2/test.csv` ($N=1,956$ test images)  
**Baseline Checkpoints**:
- `checkpoints/efficientnet_best.pt` (EfficientNet-B0 Baseline)
- `checkpoints/lxfd_attention_standard_best.pt` (LX-DFD Proposed Baseline)

---

## 1. Inspection & Verification Summary

Prior to executing Phase 3 experiments, an exhaustive code and artifact audit was conducted:
1. **Dataset Splits**: Verified `data/splits_v2` contains 9,037 train, 1,897 validation, and 1,956 test samples with zero hash, path, perceptual, or augmentation-group leakage.
2. **Preprocessing Consistency**: All training and inference routines utilize exact $224\times 224$ resizing and standard ImageNet normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
3. **Model Weights**: Checkpoint state dictionaries strictly map to model parameters without layer mismatch or random re-initialization.
4. **Calibration**: Temperature scaling parameter $T=1.0414$ was fit strictly on the validation set, ensuring zero test data contamination.

---

## 2. Baseline Performance Summary on Held-Out Test Split

| Metric | EfficientNet Baseline | LX-DFD Proposed Baseline |
|---|:---:|:---:|
| **Accuracy** | 99.54% | 99.39% |
| **ROC-AUC** | 0.9997 | 0.9998 |
| **PR-AUC** | 0.9997 | 0.9998 |
| **Precision** | 0.9933 | 0.9952 |
| **Recall** | 0.9981 | 0.9933 |
| **F1 Score** | 0.9957 | 0.9942 |
| **Brier Score** | 0.0034 | 0.0052 |
| **Expected Calibration Error (ECE)** | 0.0057 | 0.0048 |
| **Log Loss** | 0.0160 | 0.0220 |

---

## 3. Baseline Robustness Profile (Failure Diagnosis)

| Corruption Category | Condition | Baseline LX-DFD ROC-AUC | $\Delta\text{AUC}$ Drop | Vulnerability Level |
|---|---|:---:|:---:|:---:|
| **Clean** | Clean Test Set | 0.9998 | 0.0000 | None |
| **Compression** | JPEG Q=90 | 0.9993 | 0.0005 | Negligible |
| | JPEG Q=75 | 0.9855 | 0.0142 | Moderate |
| | JPEG Q=50 | 0.9637 | 0.0360 | High |
| | JPEG Q=30 | 0.9323 | 0.0675 | Severe |
| **Blur** | Gaussian Blur ($r=1.5$) | 0.9677 | 0.0321 | High |
| | Gaussian Blur ($r=3.0$) | 0.8601 | 0.1396 | Critical |
| **Noise** | Gaussian Noise ($\sigma=0.05$) | 0.6177 | 0.3821 | Catastrophic |
| | Gaussian Noise ($\sigma=0.10$) | 0.5339 | 0.4659 | Catastrophic |
| **Photometric** | Brightness (0.7x, 1.3x) | >0.9990 | <0.0007 | Negligible |
| | Contrast (0.7x, 1.3x) | >0.9994 | <0.0003 | Negligible |

The baseline models are preserved intact in `checkpoints/` and serve as the exact control reference for Phase 3 experiments.
