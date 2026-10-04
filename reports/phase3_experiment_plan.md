# LX-DFD Phase 3: Research & Experiment Plan

**Objective**: Address the high-frequency spectral vulnerabilities discovered during Phase 2 robustness benchmarking and establish a rigorous, reproducible evaluation protocol for cross-generator generalization.

---

## 1. Research Motivation & Failure Diagnosis

During Phase 2, baseline models achieved near-perfect in-domain metrics on clean test images ($>99.3\%$ accuracy, $>0.9997$ ROC-AUC). However, comprehensive corruption testing exposed critical failure modes:
1. **High-Frequency Noise Breakdown**: Additive Gaussian noise ($\sigma=0.05, 0.10$) caused severe performance degradation ($\Delta\text{AUC} = 0.3821$ and $0.4659$). The 2D DCT log-magnitude spectrum is overwhelmed by broadband high-frequency noise.
2. **High-Frequency Attenuation via Blur**: Gaussian blur ($r=3.0$) dropped ROC-AUC by $0.1396$, attenuating the high-frequency grid artifacts upon which the DCT branch relies.
3. **JPEG Compression Sensitivity**: Quality 30 compression dropped ROC-AUC by $0.0675$ due to $8\times 8$ block DCT quantization artifacts interfering with GAN fingerprint recognition.

---

## 2. Experimental Formulations

### Experiment A: Robustness-Aware Augmentation (`robustness_v2`)
- **Hypothesis**: Exposing the model to mild stochastic noise, compression, and blur during training will force both the Spatial Branch (EfficientNet-B0) and Frequency Branch (4-Stage DCT CNN) to learn noise-invariant representations and rely on structural semantics and robust mid-frequency harmonics.
- **Transform Pipeline** (Applied strictly to `train` split only):
  - Stochastic JPEG compression ($q \in [60, 95]$, $p=0.35$)
  - Stochastic Gaussian blur ($r \in [0.5, 1.5]$, $p=0.25$)
  - Stochastic additive Gaussian noise ($\sigma \in [0.01, 0.04]$, $p=0.30$)
  - Spatial transforms: RandomResizedCrop (scale $0.85–1.0$), RandomHorizontalFlip ($p=0.5$), RandomRotation ($\pm 15^\circ$), ColorJitter ($0.2$)
- **Validation/Test**: Untouched clean images ($224\times 224$ Bilinear, ImageNet normalization).
- **Target Checkpoint**: `checkpoints/lxfd_attention_robustness_v2_best.pt`.

### Experiment B: Spectral Dropout for 2D DCT Frequency Branch
- **Hypothesis**: Randomly masking rectangular blocks ($16\times 16$ to $48\times 48$) of the 2D DCT log-magnitude map during training ($p=0.20$) prevents the 4-layer Frequency CNN from co-adapting or overfitting to brittle single-band frequencies.
- **Implementation**:
  - `SpectralDropout2D` active only when `self.training == True`.
  - In evaluation mode (`self.eval()`), the 2D DCT map is completely unmasked.
- **Target Checkpoint**: `checkpoints/lxfd_attention_robustness_v2_spectral_dropout_best.pt`.

---

## 3. Evaluation & Validation Protocol

- **Dataset**: `data/splits_v2` (Zero leakage verified).
- **Benchmarked Models**:
  1. EfficientNet-B0 Baseline (`checkpoints/efficientnet_best.pt`)
  2. LX-DFD Baseline (Standard Aug) (`checkpoints/lxfd_attention_standard_best.pt`)
  3. LX-DFD Robustness v2 (`checkpoints/lxfd_attention_robustness_v2_best.pt`)
  4. LX-DFD Robustness v2 + Spectral Dropout (`checkpoints/lxfd_attention_robustness_v2_spectral_dropout_best.pt`)
- **Evaluation Conditions**:
  - Clean Test Split ($N=1,956$)
  - 14 Corruptions: JPEG (90, 75, 50, 30), Resize (0.75x, 0.50x), Gaussian Blur ($r=1.5, 3.0$), Gaussian Noise ($\sigma=0.05, 0.10$), Brightness ($0.7x, 1.3x$), Contrast ($0.7x, 1.3x$).
- **Success Criteria**:
  - Significant reduction in $\Delta\text{AUC}$ under noise ($\sigma=0.05, 0.10$), blur, and JPEG compression.
  - Retention of clean in-domain ROC-AUC $\ge 0.990$.
  - Improvement in composite Robustness Score.
