# Phase 17 Final Summary & Scientific Completion Report

**Project**: LX-DFD: Spatial-Frequency Deepfake Detection and Forensic Explainability for Cross-Generator Generalization  
**Document**: Phase 17 Final Comprehensive Summary  
**Author**: Antigravity AI Forensic Research System  
**Date**: October 26, 2026  
**Repository**: `https://github.com/Nandini05060/Deepfake_detector_AI`  
**Active Production Checkpoint**: `experiments/phase17_multigenerator_retraining/lxfd_phase17_best.pt`  
**Active Checkpoint SHA-256**: `92157690ff27268892b6ed3b73c6f664fafc3963d55f78b56cddeb993421f863`

---

## 1. What Was Wrong With the Previous Model or Inference Pipeline?

Prior to Phase 17, evaluation of the LX-DFD detector revealed two fundamental vulnerabilities:
1. **Spatial Backbone Blindness to Diffusion Textures**: The spatial ResNet-34 backbone was trained overwhelmingly on legacy GAN models (StyleGAN2) and early latent diffusion (SDv1.5). Modern generative models (Google Gemini Imagen 3, FLUX.1, SDXL) synthesize organic skin pores, iris refractions, and melanin gradients that lack GAN-like high-frequency checkerboard artifacts. Consequently, the spatial branch produced near-zero manipulation logits ($P_{\text{spatial}} = 0.04 - 0.06$) on central face crops ($m=0.50$).
2. **Failure Modes on Gemini Faces**:
   - In **Phase 5 and Phase 8**, this blindness resulted in catastrophic false-real predictions ($P_{\text{fake}} = 0.0309$, predicting authentic human photography).
   - In **Phase 10**, the model output low fake probabilities ($P \approx 0.38 - 0.55$) and routed Gemini images to `INCONCLUSIVE` or `LIKELY_REAL`.
   - In **Phase 15**, fine-tuning solely on a few Gemini images created a **specificity collapse** ($35.90\%$ genuine specificity), falsely classifying almost two-thirds of authentic human photographs as fake.
   - In **Phase 16**, the attention floor restored specificity ($62.79\%$), but the spatial branch still diverged sharply from the frequency branch ($|P_s - P_f| > 0.36$), triggering modality divergence warnings and abstention.

---

## 2. What Was Changed in Phase 17?

1. **Corpus Expansion**: Acquired and integrated 265 verified, curated $1024 \times 1024$ modern synthetic face portraits across Diffusion Transformers (FLUX.1 Dev, FLUX.1 Pro) and modern latent diffusion (SDXL).
2. **Generator-Balanced Loss Weighting**: Implemented fine-grained sample-level loss weighting during dual-branch training:
   - High supervision on modern diffusion: Google Gemini ($2.50\times$), SDXL ($1.50\times$), FLUX.1 Pro ($1.50\times$), FLUX.1 Dev ($1.40\times$).
   - Downweighted legacy GANs: StyleGAN2 ($0.70\times$) to prevent shortcut learning.
   - Boosted vintage genuine supervision: IMDB-WIKI ($1.25\times$) to suppress false alarms on grainy authentic photos.
3. **Dual-Branch Supervision**: Added auxiliary loss terms to directly supervise the spatial logit ($+0.30 \mathcal{L}_{\text{spatial}}$) and frequency logit ($+0.20 \mathcal{L}_{\text{freq}}$) alongside the fused cross-attention logit.
4. **Post-Hoc Probability Calibration**: Fitted a temperature parameter ($T = 1.1599$) and decision threshold ($\tau = 0.5400$) on the validation split alone, strictly isolating the locked test set.

---

## 3. Which Additional Datasets Were Actually Obtained and Used?

To ensure legal compliance, verified provenance, and immediate reproducibility without unapproved gated downloads, data was obtained from:
- **BitMind SyntheticFacesHQ (CC-BY/MIT Open License)** via Hugging Face Hub (`bitmind/SyntheticFacesHighQuality-T2I`):
  - **FLUX.1 Dev** (Diffusion Transformer): 133 high-resolution $1024 \times 1024$ face portraits.
  - **FLUX.1 Pro** (Commercial Diffusion Transformer): 63 high-resolution $1024 \times 1024$ face portraits.
  - **Stable Diffusion XL** (Modern Latent Diffusion): 69 high-resolution $1024 \times 1024$ face portraits.
- **Local Curated Repository Cohorts**:
  - FFHQ Studio Real (634 images), IMDB-WIKI Vintage Real (200 images), StyleGAN2 (420 images), Stable Diffusion v1.5 (200 images), InsightFace Swap (200 images), Google Gemini Imagen 3 (25 images), and SD Inpainting (200 images).

---

## 4. How Many Images Were Actually Used for Training, Validation, and Testing?

The complete audited, deduplicated manifest contains **$2,145$ images**:

| Split | Genuine Real Images | Synthetic / Deepfake Images | Total Split Size | Verification Status |
|:---|:---:|:---:|:---:|:---|
| **Training** | 590 | 718 | **1,308** | Verified (Zero test leakage) |
| **Validation** | 124 | 169 | **293** | Dedicated calibration split |
| **Locked Test** | 120 | 423 | **543** | Locked evaluation cohort |
| **Diagnostic Isolated**| 0 | 1 | **1** | `gemini_failure.png` held-out case |
| **TOTAL** | **834** | **1,311** | **2,145** | **100% Cryptographically Verified** |

---

## 5. Were Gemini Images Represented During Training?

**Yes.** To allow the model to learn the specific micro-texture and spectral signatures of Google Gemini / Imagen 3:
- **13 Gemini images** were included in the training split ($1.8\%$ of fake training data).
- **4 Gemini images** were placed in the validation split for checkpoint selection and temperature fitting.
- **8 Gemini images** were strictly reserved for the locked final test set.
- **1 Gemini diagnostic image** (`gemini_failure.png`) was strictly isolated from all splits.

---

## 6. How Many Gemini Images Were Held Out for Final Testing?

**8 independent, verified Gemini Imagen 3 face images** were held out in the locked test set. None of these 8 images were seen during training, optimization, threshold sweeping, or calibration.

---

## 7. Was the Diagnostic Gemini Image Kept Out of Training?

**Yes.** `data/diagnostics/gemini_failure.png` (SHA-256 `944cb50e9187a22a46dc6232387aa6769d4513c4654b141832b1eec04cd377fa`) was strictly excluded from all training, validation, and calibration splits. It was used solely as an independent diagnostic regression sample.

---

## 8. Real Results on Gemini and Other Generators

Across the locked test set ($N=543$):
- **Google Gemini / Imagen 3** ($N=8$):
  - Mean Fake Probability: **$0.9671$**
  - Decisive Fake Detections: **$7$ / $8$**
  - Inconclusive Abstentions: **$1$ / $8$**
  - Misclassified as Real: **$0$ / $8$ ($0.0\%$ false reals!)**
  - Decisive Fake Recall: **$100.0\%$**
- **Modern Diffusion Models**:
  - **SDXL** ($N=14$): **$100.0\%$ Decisive Recall** (14/14 detected as fake; Mean P = 0.9764).
  - **FLUX.1 Pro** ($N=28$): **$100.0\%$ Decisive Recall** (24/24 decisive detected as fake, 4 inconclusive; Mean P = 0.8910).
  - **FLUX.1 Dev** ($N=53$): **$95.45\%$ Decisive Recall** (42 fake, 2 real, 9 inconclusive; Mean P = 0.8721).
- **Classic Generators & Swaps**:
  - **StyleGAN2** ($N=60$): **$100.0\%$ Decisive Recall** (59/59 decisive detected as fake; Mean P = 0.9822).
  - **SD v1.5** ($N=30$): **$96.43\%$ Decisive Recall** (27/28 decisive detected as fake; Mean P = 0.9096).
  - **InsightFace Swap** ($N=30$): **$95.24\%$ Decisive Recall** (20/21 decisive detected as fake; Mean P = 0.7503).
- **Zero-Shot Unseen Generator (SD Inpainting)** ($N=200$):
  - Decisive Accuracy: **$90.91\%$** (80 fake, 8 real).
  - Inconclusive Rate: **$56.0\%$** (112 inconclusive due to partial facial editing).

---

## 9. Genuine-Image False-Positive Performance

- **FFHQ Studio Real** ($N=60$):
  - Mean Fake Probability: **$0.0079$** (Extremely confident authentic).
  - **Zero False Positives ($0 / 60$)**.
  - Decisive Specificity: **$100.0\%$** ($2 / 2$).
  - Forced-Choice Specificity: **$100.0\%$** ($60 / 60$).
- **IMDB-WIKI Vintage Real** ($N=60$):
  - Mean Fake Probability: **$0.4740$**.
  - Decisive Predictions: 22 False Positives, 2 True Negatives, 36 Inconclusive.
  - Forced-Choice Specificity: **$51.67\%$** ($31 / 60$).
  - **Analysis**: Legacy analog film grain, halftone scan noise, and historical contrast simulate high-frequency diffusion artifacts. This remains an active challenge across the deepfake detection literature.

---

## 10. Active Checkpoint and Cryptographic Verification

- **Active Model Key**: `phase17`
- **Active Model Name**: `LX-DFD Phase 17 Multi-Generator Retrained Champion`
- **Checkpoint Location**: `experiments/phase17_multigenerator_retraining/lxfd_phase17_best.pt`
- **Checkpoint SHA-256**: `92157690ff27268892b6ed3b73c6f664fafc3963d55f78b56cddeb993421f863`
- **Calibration File**: `experiments/phase17_multigenerator_retraining/calibration_params_phase17.json`

---

## 11. Tests Actually Run and Outcomes

1. **`scripts/reproduce_phase17_baseline.py`**: Reproduced historical baseline predictions across Phases 5, 8, 10, 15, and 16 on `gemini_failure.png` ($\checkmark$ PASSED).
2. **`scripts/build_phase17_manifest.py`**: Created audited 2,145-image manifest and certified 0% split leakage ($\checkmark$ PASSED).
3. **`experiments/phase17_multigenerator_retraining/train_phase17.py`**: Completed 8 epochs on RTX 3050 GPU; best epoch 2 checkpoint saved ($\checkmark$ PASSED).
4. **`scripts/validate_phase17_specificity.py`**: Evaluated full test set ($N=543$) and diagnostic Gemini image across margins $0.15, 0.50, 0.75$ using production `LXDFDPredictor` ($\checkmark$ PASSED).
5. **Backend Model Switching Integration**: Added Phase 17 into `app/backend/main.py` as default champion with rollback support to Phase 16, 15, and 10.
6. **Automated Regression Suite (`tests/test_phase17_verification.py`)**: Validates checkpoint integrity, SHA-256 matching, temperature calibration, prediction consistency, and zero-leakage certification.

---

## 12. Remaining Limitations

1. **Vintage & Analog Film Granularity**: Heavy film grain and legacy compression in historical portraits (IMDB-WIKI) can simulate high-frequency diffusion signatures, leading to higher inconclusive rates ($60\%$) and false alarms ($22 / 60$).
2. **Inner Central Face Skin Realism**: On central crops ($m=0.50$) without ear or hair boundaries, photorealistic diffusion skin textures can partially mask spatial artifacts, necessitating the frequency branch or context margin ($m=0.75$) to achieve decisive detection.
3. **Localized Inpainting Abstention**: Facial inpainting modifying only small regions (e.g., eyes or mouth) triggers high inconclusive rates ($56\%$), correctly reflecting the model's reliability gate when full-frame spatial context remains predominantly authentic.
