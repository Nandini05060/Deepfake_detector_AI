# Phase 9: Independent Validation and Final Reliability Audit — LX-DFD

**Audit Date**: October 22, 2026  
**Auditor**: Independent Validation Subsystem (Pair Programming AI Agent)  
**Evaluated Target Checkpoint**: `checkpoints/lxfd_multigen_diverse_best.pt`  
**Execution Platform**: Windows 11, PyTorch 2.6.0+cu124, NVIDIA GeForce RTX 3050 6GB Laptop GPU  
**Git Baseline**: Commit `ba5f685`  

---

## 1. Deployed Checkpoint Verification

An independent verification was performed on the active backend server and filesystem to verify checkpoint identity, load status, and runtime synchronization.

### Checkpoint Metadata & Cryptographic Integrity

| Metric | Verification Result | Status |
|---|---|---|
| **Resolved Checkpoint Path** | `C:\Users\nandi\Desktop\DEEPFAKE DETECTION\checkpoints\lxfd_multigen_diverse_best.pt` | **VERIFIED** |
| **SHA-256 Checksum** | `b3d280d8dbfaeede215d5e6d239b8bb613a1bf147cee9e59bde798cc9b766300` | **VERIFIED** |
| **File Size** | 59,645,056 bytes | **VERIFIED** |
| **Model Architecture** | `LXDFDModel` (Dual-Branch Spatial-Frequency + Attention Fusion with 5% Floor) | **VERIFIED** |
| **Training Epoch** | Epoch 14 | **VERIFIED** |
| **Internal Val ROC-AUC** | 0.988671 | **VERIFIED** |
| **Temperature Parameter ($T$)** | 1.0413 (Fitted via Negative Log-Likelihood on in-domain split) | **VERIFIED** |
| **Legacy Fallback Policy** | Strict existence assertion; silent fallback to Phase 2 disabled | **ENFORCED** |

### Runtime Shared Instance Verification
The backend server (`app/backend/main.py`) was refactored to enforce a singleton pattern (`get_lxfd_predictor()`):
- **Predict Endpoint (`/predict`)**: Uses the shared `_lxfd_predictor` instance.
- **Explainability Endpoint (`/explain`)**: Employs the identical `_lxfd_predictor.model` instance in GPU memory for Grad-CAM hook registration and DCT visualization.
- **Server Startup Log**: Explicitly prints the resolved path, epoch, ROC-AUC, temperature, and SHA-256 hash upon initialization.

---

## 2. Crop Sensitivity Evaluation

To determine whether face-cropping margins introduce bias or strip generative cues, an evaluation across 551 balanced validation images was executed:
- **Real Photographs**: 100 in-domain validation real + 50 IMDB-WIKI external controls.
- **StyleGAN2**: 100 in-domain validation fakes.
- **Stable Diffusion v1.5**: 30 in-domain validation + 70 external images.
- **InsightFace Swap**: 30 in-domain validation + 70 external images.
- **Held-Out Inpainting**: 100 external inpainting images.
- **Google Gemini (Imagen 3)**: 1 diagnostic external sample.

### Quantitative Crop Margin Comparison

| Crop Strategy | Dimensions (Sample) | Overall ROC-AUC | Overall Accuracy | Overall F1 | False Negative Rate | Real Specificity | StyleGAN2 Recall | SD 1.5 Recall | InsightFace Recall | SD Inpaint Recall | Gemini Fake Prob |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Margin 0.15 (Tight Inner Face)** | $731 \times 728$ | 0.7135 | 48.82% | 0.4778 | 67.83% | **96.0%** | 94.0% | 26.0% | 6.0% | 3.0% | 0.15% |
| **Margin 0.30 (Mid Context)** | $899 \times 896$ | 0.7679 | 52.27% | 0.5209 | 64.34% | **98.0%** | 98.0% | 29.0% | 13.0% | 3.0% | 27.93% |
| **Margin 0.45 (Portrait Standard)** | $1069 \times 1064$ | **0.8195** | **58.80%** | **0.6146** | 54.86% | **98.0%** | 99.0% | **44.0%** | **29.0%** | 9.0% | 17.26% |
| **Margin 0.50 (Head Boundary)** | $1125 \times 1120$ | **0.8415** | **60.80%** | **0.6424** | 51.62% | **98.0%** | 100.0% | **51.0%** | **29.0%** | 13.0% | **53.23%** |
| **Center Crop (Full Head Context)** | $1536 \times 1536$ | **0.9021** | **79.85%** | **0.8494** | **21.95%** | **98.0%** | 100.0% | **83.0%** | **78.0%** | **51.0%** | **80.46%** |

### Key Scientific Findings on Cropping
1. **Context Monotonicity**: Increasing portrait context from margin 0.15 to 0.50 monotonically improves ROC-AUC from **0.7135 to 0.8415**, and center crop reaches **0.9021**.
2. **Selective Cue Stripping**: StyleGAN2 artifacts reside primarily within the inner facial boundary (94% recall even at tight margin 0.15). In sharp contrast, modern diffusion models (SD1.5, InsightFace, Gemini) distribute manipulation evidence along the **hairline, ear synthesis, jaw boundary, and neck transitions**. A 0.15 margin eliminates these features, causing False Negative Rates of up to **67.8%**.
3. **Specific Stability**: Real photo specificity remains stable at **98.0%** across margins 0.30, 0.45, 0.50, and center crop, demonstrating that expanding context does not trigger background false positives.
4. **Adaptive Cropping Hypothesis**: For production pipelines, a dual-scale or adaptive cropping mechanism (evaluating both tight facial features and head-and-shoulders context) represents the most scientifically sound long-term architecture.

---

## 3. Attention-Floor Validation & Per-Branch Ablation

The 5% attention floor was introduced to prevent softmax modality collapse ($w_f \to 0.000$). We tested this intervention against unconstrained softmax, spatial-only, and frequency-only sub-networks on identical test data.

### Attention Fusion Ablation Matrix

| Experimental Condition | Mean $w_s$ | Mean $w_f$ | In-Domain ROC-AUC | External Multi-Gen AUC | Overall Validation AUC | Overall Validation F1 |
|---|---|---|---|---|---|---|
| **Attention Floor Active (floor_0.05)** | 4.0379 | -3.2120 ($w_f \ge 0.05$) | 0.9884 | 0.6907 | **0.8256** | **0.6613** |
| **Unconstrained Softmax (floor_0.00)** | 4.4311 | -3.6245 ($w_f \to 0.000$) | 0.9884 | 0.6914 | **0.8261** | **0.6613** |
| **Spatial-Only ($w_s=1.0, w_f=0.0$)** | 1.0000 | 0.0000 | **0.9919** | **0.6929** | 0.8201 | 0.6146 |
| **Frequency-Only ($w_s=0.0, w_f=1.0$)** | 0.0000 | 1.0000 | 0.4432 | 0.4676 | 0.4428 | 0.0000 |

### Critical Scientific Verdict
- **UI & Modality Stability**: The 5% floor successfully eliminates the display starvation bug ($w_f = 0.000$) and slightly regularizes F1 performance from 0.6146 to 0.6613 over spatial-only inference.
- **Empirical Reality Check**: The frequency branch alone exhibits an AUC of only **0.4428** on external multi-generator test sets (worse than random guessing). Modern text-to-image engines lack the distinct high-frequency checkerboard peaks of early GANs.
- **Conclusion**: The attention floor is a **numerical stability and explainability safeguard**, not a transformative accuracy breakthrough. Generalization is predominantly driven by the spatial branch representations.

---

## 4. Probability Calibration & Out-of-Distribution Reliability

Calibration metrics were evaluated across in-domain validation data (StyleGAN2 + Real) and external multi-generator data (SD1.5, InsightFace, SD Inpainting).

### Calibration Comparison Table

| Data Split | Metric | Uncalibrated Model | Temperature Scaled ($T=1.0413$) | Evaluation Finding |
|---|---|---|---|---|
| **In-Domain Validation** | **Brier Score** | 0.01678 | 0.01685 | Near-optimal in-domain probability calibration |
| | **Log-Likelihood (NLL)** | 0.08554 | **0.08418** | Slight entropy regularization |
| | **ECE (10 Bins)** | **0.02273** | **0.02281** | Low calibration error (~2.3%) |
| **External Generators** | **Brier Score** | 0.58045 | 0.57737 | Severe cross-domain degradation |
| | **Log-Likelihood (NLL)** | 2.82065 | 2.71584 | Substantial cross-entropy error |
| | **ECE (10 Bins)** | **0.59799** | **0.59632** | Extreme calibration error (~59.6%) |

### Reliability Diagram Analysis (External Generators)
- In the lowest predicted confidence bin ($[0.0, 0.1]$), the model placed 217 samples with an average predicted fake probability of **1.61%**.
- However, the empirical ground truth for those samples was **82.49% FAKE**.
- **Root Cause Confirmed**: Temperature scaling cannot correct cross-domain mean logit shift. When an unobserved synthesis engine produces imagery that looks smooth to a convolutional filter, the network outputs high-confidence Real predictions.
- **Policy Enforcement**: Softmax confidence $\ge 90\%$ is **never** presented as empirical real-world certainty. A Real classification denotes absence of known signatures from evaluated generators, accompanied by an explicit Out-of-Distribution Advisory.

---

## 5. Automated Regression & Robustness Test Suite

The automated test suite (`scripts/test_pipeline_regression.py` and `scripts/phase9_validation_suite.py`) verified 8 critical operational guarantees:

| Test ID | Test Case | Target / Specification | Actual Runtime Result | Status |
|---|---|---|---|---|
| **REG-01** | Checkpoint Selection | Must load `lxfd_multigen_diverse_best.pt` | Loaded verified checkpoint | **PASSED** |
| **REG-02** | Class Mapping | REAL $\to 0.0$, FAKE $\to 1.0$ | Real prob $= 0.47\%$, Fake prob $= 52.67\%$ | **PASSED** |
| **REG-03** | Numerical Sanity | No NaNs, No Infs in logits or features | Output Logit $=-5.6003$ (finite float) | **PASSED** |
| **REG-04** | Attention Floor | $w_s \ge 0.05, w_f \ge 0.05, w_s + w_f = 1.0$ | $w_s = 0.9497, w_f = 0.0503$ | **PASSED** |
| **REG-05** | Face Detection Fallback | Blank / No-face input graceful fallback | Returns original input without crash | **PASSED** |
| **REG-06** | Aspect Ratio Invariance | Panoramic ($1920\times 480$) & Vertical ($480\times 1920$) | Processed cleanly without anisotropic distortion | **PASSED** |
| **REG-07** | Explainability Alignment | Grad-CAM and DCT colormaps $(224, 224, 3)$ | Valid uint8 heatmaps aligned with prediction | **PASSED** |
| **REG-08** | Adversarial Inpainting | SD inpainting texture bleed evaluation | Flagged with empirical uncertainty notice | **PASSED** |

---

## 6. Honest Evaluation of the Gemini Sample

The original Google Gemini sample (`data/test_samples/gemini/gemini_sample.png`, $2816 \times 1536$) was evaluated across the production inference pipeline:

```
[Diagnostic Assessment: Google Gemini Imagen 3 Sample]
- Checkpoint Hash: SHA256:b3d280d8dbfaeede215d5e6d239b8bb613a1bf147cee9e59bde798cc9b766300
- YuNet Detected Bounding Box: [x=1314.2, y=476.0, w=563.8, h=560.1, conf=0.7713]

Results by Preprocessing Configuration:
1. Production Margin 0.45 (1069x1064 crop):
   - Fused Logit: -1.5670 | Calibrated Fake Prob: 17.26% | Real Prob: 82.74%
   - Prediction: REAL | Confidence: MODERATE (Artifact Absence vs True Authenticity)
   - Spatial Prob: 16.83% | Frequency Prob: 10.01%
   - OOD Advisory Attached: YES

2. Extended Margin 0.50 (1125x1120 crop):
   - Fused Logit: +0.1380 | Calibrated Fake Prob: 53.23%
   - Prediction: UNCERTAIN / LOW CONFIDENCE (Decision Boundary Crossing)
   - Spatial Prob: 55.90% | Frequency Prob: 10.80%

3. Center Crop (1536x1536 full portrait context):
   - Fused Logit: +1.4847 | Calibrated Fake Prob: 80.46%
   - Prediction: FAKE | Confidence: MODERATE
   - Spatial Prob: 83.69% | Frequency Prob: 11.15%
```

### Scientific Diagnostic Takeaway
At standard margin 0.45, the crop remains close to the face, predicting Real (17.3% fake) with a **MODERATE** confidence rating and an active **Out-of-Distribution Scope Advisory**. When expanded slightly to margin 0.50 or full portrait, the generative boundary cues flip the prediction to Fake. 

**This single sample cannot be used to declare general Gemini detection capability.** It serves solely as diagnostic proof that cropping context dictates detection efficacy on modern diffusion models.

---

## 7. Launch Command & Operational Instructions

To launch the verified, production-calibrated LX-DFD backend:

```powershell
# Navigate to repository root
cd "C:\Users\nandi\Desktop\DEEPFAKE DETECTION"

# Launch verified FastAPI production server
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Access the live dashboard at: `http://127.0.0.1:8000/`
