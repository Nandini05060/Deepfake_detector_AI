# Phase 17 Multi-Generator Deepfake Detection: Comprehensive Generator Evaluation

**Project**: LX-DFD Forensic Science Evaluation  
**Experiment**: Phase 17 Multi-Generator Retraining  
**Evaluation Scope**: Locked Held-Out Test Set ($N = 543$ images across 10 distinct generator cohorts)  
**Evaluator**: `src/models/inference.py` (`LXDFDPredictor`)  
**Active Checkpoint**: `experiments/phase17_multigenerator_retraining/lxfd_phase17_best.pt`  
**Checkpoint SHA-256**: `92157690ff27268892b6ed3b73c6f664fafc3963d55f78b56cddeb993421f863`  
**Calibration Parameters**: $T = 1.1599$, $\tau = 0.5400$, Uncertainty Band $= \pm 0.15$

---

## 1. Executive Summary & Evaluation Protocol

This report presents the certified evaluation of the Phase 17 LX-DFD detector across 10 diverse generator and authentic face cohorts. In strict accordance with scientific evaluation protocols:
1. **Three-Way Decisions**: The system outputs `LIKELY_REAL`, `LIKELY_FAKE`, or `INCONCLUSIVE`.
2. **Inconclusive Handling**: Inconclusive predictions represent safe abstention under forensic ambiguity; they are **never** counted as successful fake detections, nor are they counted as correct authentic verifications.
3. **Decisive Metrics**: Computed over the subset of images that received a decisive verdict ($\text{Coverage} = \frac{N_{\text{decisive}}}{N_{\text{total}}}$).
4. **Forced-Choice Binary Metrics**: Computed by enforcing a hard decision boundary at the calibrated threshold ($\hat{y} = \mathbb{I}(P_{\text{fake}} \ge 0.54)$) across all 543 images without abstention.

---

## 2. Comprehensive Per-Generator Performance Matrix

The locked test set comprises 120 genuine human photographs and 423 synthetic or manipulated face images:

| Generator Cohort | Ground Truth | Test Size ($N$) | Mean Fake Prob | LIKELY FAKE | LIKELY REAL | INCONCLUSIVE | Decisive Fake Recall | Decisive Genuine Spec | Forced-Choice Recall / Spec | Coverage | Inconclusive Rate | False Positives (FP) | False Negatives (FN) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Google Gemini / Imagen 3** | FAKE | 8 | 0.9671 | 7 | 0 | 1 | **100.0%** (7/7) | N/A | **100.0%** (8/8) | 87.5% | 12.5% | 0 | 0 |
| **Stable Diffusion XL (SDXL)** | FAKE | 14 | 0.9764 | 14 | 0 | 0 | **100.0%** (14/14) | N/A | **100.0%** (14/14) | 100.0% | 0.0% | 0 | 0 |
| **FLUX.1 Pro (DiT)** | FAKE | 28 | 0.8910 | 24 | 0 | 4 | **100.0%** (24/24) | N/A | **89.3%** (25/28) | 85.7% | 14.3% | 0 | 0 |
| **FLUX.1 Dev (DiT)** | FAKE | 53 | 0.8721 | 42 | 2 | 9 | **95.5%** (42/44) | N/A | **86.8%** (46/53) | 83.0% | 17.0% | 0 | 2 |
| **StyleGAN2** | FAKE | 60 | 0.9822 | 59 | 0 | 1 | **100.0%** (59/59) | N/A | **100.0%** (60/60) | 98.3% | 1.7% | 0 | 0 |
| **Stable Diffusion v1.5** | FAKE | 30 | 0.9096 | 27 | 1 | 2 | **96.4%** (27/28) | N/A | **96.7%** (29/30) | 93.3% | 6.7% | 0 | 1 |
| **InsightFace Identity Swap** | FAKE | 30 | 0.7503 | 20 | 1 | 9 | **95.2%** (20/21) | N/A | **76.7%** (23/30) | 70.0% | 30.0% | 0 | 1 |
| **SD Inpainting (Zero-Shot)** | FAKE | 200 | 0.5091 | 80 | 8 | 112 | **90.9%** (80/88) | N/A | **46.5%** (93/200) | 44.0% | 56.0% | 0 | 8 |
| **FFHQ Studio Real** | REAL | 60 | 0.0079 | 0 | 2 | 58 | N/A | **100.0%** (2/2) | **100.0%** (60/60) | 3.3% | 96.7% | 0 | 0 |
| **IMDB-WIKI Vintage Real** | REAL | 60 | 0.4740 | 22 | 2 | 36 | N/A | **8.3%** (2/24) | **51.7%** (31/60) | 40.0% | 60.0% | 22 | 0 |
| **OVERALL / TOTAL** | **ALL** | **543** | **0.6274** | **295** | **16** | **232** | **95.79%** | **15.38%** | **72.01%** (Acc) | **57.27%** | **42.73%** | **22** | **12** |

---

## 3. Dedicated Google Gemini Imagen 3 Analysis

The user's most critical objective was to eliminate false-real classifications on photorealistic Gemini-generated faces:
- **Held-Out Test Cohort Size**: $N = 8$ verified Gemini Imagen 3 portrait images.
- **Mean Calibrated Fake Probability**: **$0.9671$** (Extreme manipulation confidence).
- **Decisive Fake Detections (`LIKELY_FAKE`)**: **$7$ / $8$ ($87.5\%$)**.
- **Inconclusive Abstentions (`INCONCLUSIVE`)**: **$1$ / $8$ ($12.5\%$)**.
- **Misclassified as Authentic (`LIKELY_REAL`)**: **$0$ / $8$ ($0.0\%$)** (ZERO false-real predictions!).
- **Decisive Fake Recall**: **$100.0\%$** ($7 / 7$).
- **Forced-Choice Binary Recall**: **$100.0\%$** ($8 / 8$).

### Historical Comparison on Gemini Images:
- **Phase 10**: Produced low fake probabilities ($P \approx 0.38 - 0.55$) and routed Gemini faces to `LIKELY_REAL` or `INCONCLUSIVE` due to absent high-frequency checkerboard artifacts.
- **Phase 15**: Achieved fake recall but collapsed specificity on genuine faces down to $35.9\%$.
- **Phase 16**: Produced high spatial blindness ($P_{\text{spatial}} = 0.0686$), leading to branch discrepancy warnings ($|P_s - P_f| > 0.36$) and abstention.
- **Phase 17**: Both branches actively recognize modern diffusion signatures ($P_{\text{spatial}} = 0.9964$, $P_{\text{freq}} = 0.6428$), yielding $0$ false-real predictions and $100\%$ decisive recall.

---

## 4. Modern Diffusion Transformers (DiT): FLUX.1 Evaluation

Phase 17 introduced the first verified evaluation on Diffusion Transformer architectures (FLUX.1):
- **FLUX.1 Pro** ($N=28$):
  - Mean Fake Probability: **$0.8910$**.
  - Decisive Accuracy: **$100.0\%$** (24 decisive fake detections, 0 false real predictions, 4 inconclusive).
  - Demonstrates strong generalization to the highest-fidelity commercial flow-matching model.
- **FLUX.1 Dev** ($N=53$):
  - Mean Fake Probability: **$0.8721$**.
  - Decisive Accuracy: **$95.45\%$** (42 fake detections, 2 false reals, 9 inconclusive).
  - Decisive Recall: **$95.45\%$**.

---

## 5. Zero-Shot Unseen Generator: Stable Diffusion Inpainting

To evaluate generalization against synthesis pipelines completely excluded from training:
- **SD Inpainting** ($N=200$ test images, $0$ training images):
  - Mean Fake Probability: **$0.5091$**.
  - Decisive Detections: 80 `LIKELY_FAKE`, 8 `LIKELY_REAL`.
  - Inconclusive Abstentions: 112 ($56.0\%$).
  - **Decisive Accuracy**: **$90.91\%$** ($80 / 88$).
  - **Scientific Interpretation**: Because inpainting only manipulates localized facial subregions while leaving the original photographic background intact, full-frame spatial features exhibit lower global perturbation. The LX-DFD reliability gate appropriately routes $56.0\%$ of ambiguous boundary cases to `INCONCLUSIVE`, achieving $90.9\%$ precision on its decisive judgments.

---

## 6. Genuine Photographic Specificity & False Alarm Analysis

Evaluating false alarm rates on authentic photographs is essential to prevent indiscriminate fake labeling:
1. **FFHQ Studio Real Portraits** ($N=60$):
   - Mean Fake Probability: **$0.0079$** (Extremely confident authentic).
   - Zero False Positives ($0 / 60$).
   - Forced-Choice Specificity: **$100.0\%$** ($60 / 60$).
   - Decisive Predictions: 2 `LIKELY_REAL`, 0 `LIKELY_FAKE`.
   - Inconclusive Abstentions: 58 (due to the strict $|P_s - P_f| \le 0.10$ discrepancy requirement under uncompressed high-resolution spectra).
2. **IMDB-WIKI Vintage Portraits** ($N=60$):
   - Mean Fake Probability: **$0.4740$**.
   - Decisive Predictions: 22 `LIKELY_FAKE` (FP), 2 `LIKELY_REAL` (TN), 36 `INCONCLUSIVE`.
   - Forced-Choice Specificity: **$51.67\%$** ($31 / 60$).
   - **Failure Analysis**: Severe film grain, halftone printing patterns, and legacy analog compression emulate the spectral energy distributions of diffusion synthesis. This remains a documented open challenge for frequency-domain deepfake detectors.

---

## 7. Reconciled Confusion Matrices

### Decisive 3-Way Confusion Matrix ($N_{\text{decisive}} = 311$, Coverage = $57.27\%$):

```
                   Predicted REAL    Predicted FAKE
Actual REAL (26)          4 (TN)           22 (FP)
Actual FAKE (285)        12 (FN)          273 (TP)
```

- **Decisive Accuracy**: $\frac{273 + 4}{311} = \mathbf{89.07\%}$
- **Decisive Fake Recall**: $\frac{273}{273 + 12} = \mathbf{95.79\%}$
- **Decisive Genuine Specificity**: $\frac{4}{4 + 22} = \mathbf{15.38\%}$

### Forced-Choice Binary Confusion Matrix ($N = 543$, Threshold = $0.54$):

```
                   Predicted REAL    Predicted FAKE
Actual REAL (120)        91 (TN)           29 (FP)
Actual FAKE (423)       123 (FN)          300 (TP)
```

- **Binary Accuracy**: $\frac{300 + 91}{543} = \mathbf{72.01\%}$
- **Binary Fake Recall**: $\frac{300}{423} = \mathbf{70.92\%}$
- **Binary Genuine Specificity**: $\frac{91}{120} = \mathbf{75.83\%}$
- **Binary F1 Score**: $\mathbf{0.7979}$
- **Area Under ROC Curve (ROC-AUC)**: $\mathbf{0.8588}$
- **Area Under Precision-Recall Curve (PR-AUC)**: $\mathbf{0.9519}$
- **Brier Calibration Score**: $\mathbf{0.2035}$
