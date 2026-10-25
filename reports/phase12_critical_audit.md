# Phase 12A Critical Forensic Audit of Phase 11 & Baseline Pipeline
## Comprehensive Architecture Trace, Checkpoint Integrity, Preprocessing Audits, and Baseline Reproducibility

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Machine Learning Engineer & Deepfake Forensics Auditor  
**Document Identifier:** `reports/phase12_critical_audit.md`  
**Associated Files:**
- `reports/phase12_baseline_reproduction.json`
- `reports/phase12_metric_discrepancies.md`

---

## 1. Executive Summary & Verification Protocol

In accordance with Phase 12A instructions, an exhaustive architectural inspection, checkpoint authentication, and independent baseline reproduction were conducted on the LX-DFD codebase.

### Primary Audit Verdicts:
1. **Checkpoint Authentication Verified:**
   - Active Checkpoint: `checkpoints/lxfd_phase10_modern_generators_best.pt`
   - Checkpoint SHA-256: `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05`
   - Calibration Temperature: $T = 1.634375$ (documented as $T = 1.6344$)
   - Class-Label Mapping: `0 = REAL`, `1 = FAKE`
2. **Exact Baseline Metric Reproduction Confirmed:**
   - Across the 457 locked test samples, the active model reproduces Phase 11 metrics with zero numerical discrepancy:
     - **Spatial-Only Model:** ROC-AUC = 0.8526, Accuracy = 0.5536, F1 = 0.5870, Fake Recall = 0.4303
     - **Frequency-Only Model:** ROC-AUC = 0.6431, Accuracy = 0.5864, F1 = 0.6595, Fake Recall = 0.5430
     - **Attention Fusion (Active Baseline):** ROC-AUC = 0.8544, Accuracy = 0.5624, F1 = 0.6000, Fake Recall = 0.4451
     - **Equal-Weight Average Fusion:** ROC-AUC = 0.8386, Accuracy = 0.6346, F1 = 0.6890, Fake Recall = 0.5490
3. **Core Failure Modes Identified:**
   - **Gemini / Imagen 3 Blind Spot:** Baseline recall is only **11.76%** (2/17 detected), with mean $P(\text{Fake}) = 0.2134$. Modern diffusion synthesizes naturalistic microtextures that deceive the spatial branch ($\bar{P}_s = 0.2231$) and lack the high-frequency grid harmonics exploited by the frequency branch ($\bar{P}_f = 0.0771$).
   - **Attention Modality Starvation:** On StyleGAN2, the spatial branch achieves 95.0% accuracy, causing the attention module to allocate $\sim 94\%$ weight to spatial features ($w_s \approx 0.938$) and clamp the frequency branch to its lower floor ($w_f \approx 0.062$). Consequently, on inpainting and face swaps where the frequency branch excels (68.5% and 76.7% recall), the attention head suppresses its alarm, degrading overall detection to 28.0% and 53.3%.
   - **Crop Geometry Sensitivity:** A discrepancy between `YuNetFacePreprocessor` ($1.5\times$ crop) and `main.py` ($2.0\times$ crop) alters Gemini fake predictions from $P=0.9489$ down to $P=0.0341$ on identical raw imagery.

---

## 2. Model Architecture & Pipeline Verification

### 2.1 Model Class: `LXDFDModel` (`src/models/lxfd_model.py`)
- **Spatial Branch:** `SpatialBranch` wraps torchvision `EfficientNet-B0` with `AdaptiveAvgPool2d` and a linear projection to $F_s \in \mathbb{R}^{256}$.
- **Frequency Branch:** `FrequencyBranch` computes 2D DCT Log-Magnitude spectra via `DiscreteCosineTransform2D(size=224)` followed by an optional `SpectralDropout2D` and a 4-stage convolutional feature extractor (Conv2d-BN-ReLU blocks with channels 32, 64, 128, 256) projected to $F_f \in \mathbb{R}^{256}$.
- **Fusion Module:** `SpatialFrequencyFusion` concatenates $[F_s, F_f] \in \mathbb{R}^{512}$, passes through a two-layer MLP (`Linear(512, 128) -> ReLU -> Linear(128, 2)`), applies softmax, and bounds weights with a minimum floor $\tau = 0.05$:
  $$w_s = 0.05 + 0.90 \cdot \alpha_s, \quad w_f = 0.05 + 0.90 \cdot \alpha_f$$
  $$F_{\text{fused}} = w_s F_s + w_f F_f$$
- **Classifier Head:** `Linear(256, 128) -> GELU -> Dropout(0.3) -> Linear(128, 1)` producing logit $z$.
- **Calibration:** $P(\text{Fake}) = \sigma(z / T)$ where $T = 1.6344$.

### 2.2 Numerical & Color-Space Precision
- Input RGB tensors are scaled to $[0.0, 1.0]$ and normalized via standard ImageNet parameters:
  $$\text{Mean} = [0.485, 0.456, 0.406], \quad \text{Std} = [0.229, 0.224, 0.225]$$
- In `DiscreteCosineTransform2D`, luminance is computed directly from the standardized tensor:
  $$Y = 0.299 R + 0.587 G + 0.114 B$$
  The orthogonal matrix multiplication $C Y C^T$ is performed deterministically. Log-magnitude spectra are min-max normalized per sample to $[0.0, 1.0]$.

---

## 3. Preprocessing & Face Localization Audit

### 3.1 OpenCV YuNet Configuration
- Detector model: `yunet.onnx` (file size: 232,589 bytes).
- Standard parameters: input size $(320, 320)$, score threshold $0.50$, NMS threshold $0.30$, top_k $5$.
- Bounding box selection: Primary face is chosen as the largest detected bounding box by area ($fw \cdot fh$).

### 3.2 Crop Geometry Divergence
- **Implementation 1 (`app/backend/main.py`):**
  $$\Delta x = \text{int}(fw \cdot 0.50), \quad \Delta y = \text{int}(fh \cdot 0.50)$$
  $$\text{Crop Width} = fw + 2 \Delta x = 2.0 \cdot fw$$
- **Implementation 2 (`src/data/face_preprocessor.py`):**
  $$\text{side} = \text{int}(\text{round}(\max(fw, fh) \cdot (1.0 + \text{margin})))$$
  With $\text{margin} = 0.50$, $\text{Crop Width} = 1.50 \cdot fw$.
- **Audit Finding:** Implementation 1 includes broader surrounding context (hair, ears, jawline, upper neck), which exhibits subtle diffusion synthesis boundary artifacts. Implementation 2 focuses strictly on facial interior skin. This divergence accounts for why Phase 10 reported higher Gemini detection rates than Phase 11.

---

## 4. Software Regression & API Audit

- **Test Suite Status:** 35/35 pytest unit tests pass with zero errors when `PYTHONPATH="."` is set.
- **FastAPI Endpoints Verified:**
  - `GET /health`: Returns system state, CUDA device (`NVIDIA GeForce RTX 3050`), PyTorch version.
  - `GET /models`: Returns model inventory and active checkpoint.
  - `GET /metrics`: Returns baseline performance table.
  - `GET /generalization`: Returns per-generator breakdown and generalization gap.
  - `POST /predict`: Uploads image, executes YuNet face crop, computes dual-branch inference with calibrated temperature scaling, and outputs empirical uncertainty telemetry.
- **Frontend-Backend Contract:** API responses include exact field names expected by `app/frontend/app.js` (`fake_probability`, `real_probability`, `prediction`, `branch_outputs`, `generator_warning`, `is_uncertain`).

---

## 5. Audit Conclusions and Phase 12 Priorities

1. **Ablation Baseline Validated:** The Phase 11 metrics are 100% reproducible and confirmed as ground truth for all subsequent comparisons.
2. **The 112x112 Downsampling Effect Must Be Audited on Controls:** Before adopting multi-scale or downsampling architectures, Phase 12C must evaluate the impact across the full 120-image authentic real control dataset to guard against false-positive inflation.
3. **Multi-Scale Spatial Candidate Justification:** If downsampling macro-features improves diffusion detection without degrading real photograph specificity, a dual-scale spatial architecture (`224x224` detail + `112x112` global) is theoretically and empirically warranted.
4. **Attention Head Regularization Required:** The attention fusion module must be re-engineered (higher floor or entropy regularization) so that the frequency branch is not silenced on localized manipulations (inpainting and face swaps).
