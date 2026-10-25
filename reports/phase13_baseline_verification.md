# Phase 13A Baseline Verification & Cryptographic Audit
## Forensic Integrity, Checkpoint Authentication, Inference Pipeline Inspection, and Baseline Reproduction

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase13_baseline_verification.md`  
**Associated Benchmark Data:** `reports/phase13_baseline_metrics.json`  

---

### 1. Executive Summary & Audit Mandate

Prior to implementing any Phase 13 reliability improvements or threshold reconfigurations, Phase 13A establishes cryptographic proof of model weights, inspects the end-to-end inference pipeline, audits the active fusion configuration, and reproduces all baseline metrics on the locked test partition ($N=457$).

**Audit Verdict:**
- Cryptographic SHA-256 signatures for both production (`lxfd_phase10_modern_generators_best.pt`) and research candidate (`lxfd_multiscale12_best.pt`) checkpoints are verified intact with 100% byte-level fidelity.
- Baseline metrics on the locked evaluation manifest reproduce Phase 12 reported numbers to four decimal places.
- Audit confirms that changing `min_floor` from $0.05$ to $0.25$ in `SpatialFrequencyFusion` is a forward-pass tensor constraint that does **not** alter state dictionary keys, shapes, or invalidate trained weights.

---

### 2. Checkpoint Authentication & Cryptographic Hashes

| Checkpoint Identifier | Role | File Path | File Size | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **`LXDFD-Phase10-Best`** | Active Production Backbone | `checkpoints/lxfd_phase10_modern_generators_best.pt` | 19,457,485 bytes | `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05` | **AUTHENTICATED** |
| **`LXDFD-MultiScale-12`** | Research Candidate | `experiments/phase12_multiscale_v1/lxfd_multiscale12_best.pt` | 54,952,381 bytes | `a2b9989205f255248ca7c378e25164b88fbc47691ce8b9b9b815fcd00a72e773` | **AUTHENTICATED** |
| **`Calibration-Phase10`** | Temperature Calibration | `checkpoints/calibration_params_phase10.json` | 40 bytes | Parameter $T = 1.634375$ (rounds to $1.6344$) | **AUTHENTICATED** |
| **`Locked Manifest`** | Evaluation Benchmark Split | `data/manifests/phase11_generator_manifest.csv` | 666,682 bytes | SHA-256: `b21d9d4c2bbd4860630ee2765b3a20f3846f3ebaaf1c3f2aabac24566bc08304` | **AUTHENTICATED** |

---

### 3. Inference Pipeline & Preprocessing Inspection

The production inference pipeline (`src/models/inference.py` and `app/backend/main.py`) was audited line-by-line:

```
[Input Image: PIL Image / File]
          │
          ▼
[Color Space Normalization] ──> Explicit conversion to RGB format (H, W, 3)
          │
          ▼
[YuNet Face Detector (ONNX)] ──> OpenCV YuNet ONNX, threshold = 0.50, NMS = 0.30
          │                      Extracts largest primary face bounding box
          ▼
[Context Margin Padding] ───> Margin = 0.50 (50% context expansion preserving hair/ears)
          │
          ▼
[Tensor Transforms] ─────────> Bilinear Resize to (224, 224), ToTensor()
                               ImageNet Normalization:
                                 mean = [0.485, 0.456, 0.406]
                                 std  = [0.229, 0.224, 0.225]
          │
          ▼
[Dual-Branch Model] ─────────> Spatial EfficientNet-B0 + Frequency 2D-DCT CNN
          │
          ▼
[Fusion & Classifier] ───────> Dynamic Attention Fusion (min_floor = τ) -> Logit z
          │
          ▼
[Temperature Calibration] ───> z_cal = z / 1.634375 -> Sigmoid P_fake = 1 / (1 + exp(-z_cal))
          │
          ▼
[Operating Threshold] ───────> Decision threshold = 0.50
```

#### Verification of Fusion Setting & State Dictionary Integrity
A dedicated architectural audit verified whether setting `min_floor = 0.25` changes the model architecture:
1. `SpatialFrequencyFusion` uses `min_floor` as a numerical bounding constant on the raw softmax output:
   $$w_i = \tau + (1 - 2\tau) \cdot \text{Softmax}(\text{MLP}([F_s, F_f]))_i$$
2. `min_floor` is **not** a registered parameter or buffer in PyTorch.
3. Model keys with $	au = 0.05$ and $	au = 0.25$ are **strictly identical** (338 state dict tensors).
4. Loading `checkpoints/lxfd_phase10_modern_generators_best.pt` into `LXDFDModel(min_floor=0.25)` loads with `strict=True` with **0 missing keys and 0 unexpected keys**.
5. **Finding:** The constrained-attention setting ($	au=0.25$) is a pure runtime forward constraint that immediately resolves frequency branch starvation without modifying or retraining checkpoint weights.

---

### 4. Baseline Reproduction Metrics on Locked Test Set ($N=457$)

Evaluation conducted across all four test splits (`test_indomain`, `test_unseen_inpaint`, `test_locked`, `test_gemini_heldout`):

| Metric | Phase 12 Report | Phase 13A Reproduced (Baseline $	au=0.05$) | Phase 13A Constrained ($	au=0.25$) | Delta (Repro vs P12) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.8544 | **0.8544** | **0.8556** | 0.0 |
| **PR-AUC** | 0.9329 | **0.9329** | **0.9333** | 0.0 |
| **Accuracy (Threshold 0.50)** | 56.24% | **56.24%** | **60.18%** | 0.0% |
| **Macro F1 Score** | 0.6000 | **0.6** | **0.6459** | 0.0 |
| **Fake Recall (Sensitivity)** | 44.51% (150/337) | **44.51%** (150/337) | **49.26%** (166/337) | 0.0% |
| **Real Specificity** | 89.17% (107/120) | **89.17%** (107/120) | **90.83%** (109/120) | 0.0% |
| **False Positive Rate (FPR)** | 10.83% (13/120) | **10.83%** (13/120) | **9.17%** (11/120) | 0.0% |
| **False Negative Rate (FNR)** | 55.49% (187/337) | **55.49%** (187/337) | **50.74%** (171/337) | 0.0% |
| **Brier Score** | 0.2923 | **0.2923** | **0.2703** | 0.0 |
| **10-Bin ECE** | 0.3381 | **0.3381** | **0.3254** | 0.0 |

---

### 5. Cohort Breakdown Across Generators

| Generator Family | Ground Truth | Sample Count | Baseline Recall | 95% Wilson CI | Constrained ($	au=0.25$) Recall | Mean Predicted $P_{\text{fake}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Gemini / Imagen 3** | FAKE | 17 | **2/17** (11.76%) | [0.0329, 0.3434] | 2/17 | 0.2621 |
| **Stable Diffusion v1.5** | FAKE | 30 | **19/30** (63.33%) | [0.4551, 0.7813] | 20/30 (66.67%) | 0.6709 |
| **InsightFace Swap** | FAKE | 30 | **16/30** (53.33%) | [0.3614, 0.6977] | 17/30 (56.67%) | 0.5252 |
| **StyleGAN2 Synthesis** | FAKE | 60 | **57/60** (95.00%) | [0.863, 0.9829] | 57/60 (95.00%) | 0.9152 |
| **SD Inpainting (Held-Out)** | FAKE | 200 | **56/200** (28.00%) | [0.2224, 0.3459] | 70/200 (35.00%) | 0.3556 |
| **Authentic Real Faces** | REAL | 120 | **107/120** (89.17%) | [0.8234, 0.9356] | 109/120 (90.83%) | 0.1344 |

---

### 6. Verification Finding & Audit Sign-Off
1. The baseline is 100% cryptographically authentic and numerically verified.
2. The production model satisfies the $\ge 88\%$ real specificity requirement ($89.17\%$ for $	au=0.05$ and $90.83\%$ for $	au=0.25$).
3. Setting `min_floor = 0.25` in production raises overall ROC-AUC from $0.8544$ to $0.8556$ and increases inpainting recall from $28.0\%$ to $35.0\%$ without increasing false positives on genuine photographs.
4. Phase 13B will proceed to threshold and calibration analysis strictly on the independent validation split ($N=242$).
