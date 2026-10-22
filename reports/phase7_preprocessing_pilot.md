# Phase 7: Preprocessing Correction & Pilot Evaluation Report

**Author**: Senior Deepfake Detection Researcher  
**Project**: LX-DFD (Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning)  
**Date**: October 2026  
**Status**: Experimental Validation & Verification Report  

---

## 1. Executive Summary & Problem Formulation

In Phase 4 and Phase 5, external evaluation across unseen generator families (Stable Diffusion v1.5, InsightFace Swap, Stable Diffusion Inpainting) yielded poor generalization metrics ($\text{AUC}_{\text{macro}} \approx 0.529 - 0.577$). 

The Phase 6 evidence audit ([`reports/phase6_evidence_audit.md`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/reports/phase6_evidence_audit.md)) uncovered a fundamental **methodological confounding factor**:
1. **In-Domain Training Data**: Trained on $256 \times 256$ square facial crops from Kaggle 140k Faces (FFHQ authentic portraits and StyleGAN2 synthetic faces).
2. **External Evaluation Pipeline**: In [`scripts/evaluate_unseen_generators.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/scripts/evaluate_unseen_generators.py), images were loaded and naively resized using `torchvision.transforms.Resize((224, 224))` directly on raw full-frame photographs.
3. **Severe Aspect-Ratio Distortion**: Authentic real images from IMDB-WIKI were non-square vertical portraits ($400 \times 557$, $440 \times 653$, $550 \times 825$; aspect ratios $1.39 - 1.50$), whereas synthetic images from Latent Diffusion and InsightFace were predominantly native $512 \times 512$ squares ($1.00$).
4. **Frequency Shortcut Confounding**: Bilinearly squashing vertical rectangles into squares compressed horizontal spatial wavelengths by $28\% - 33\%$, injecting artificial high-frequency energy spikes ($0.009420$ in real vs $0.008250$ in fake) that severely distorted the 2D DCT spectral representation.

Phase 7 resolves this flaw by engineering and validating a standardized, aspect-ratio-preserving face preprocessing pipeline.

---

## 2. Standardized Face Preprocessing Architecture (`src/data/face_preprocessor.py`)

The corrected preprocessing pipeline ([`src/data/face_preprocessor.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/src/data/face_preprocessor.py)) uses the official OpenCV YuNet face detection engine ([`yunet.onnx`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/yunet.onnx)):

```
Raw Image ──> [YuNet Face Detection] 
                     │
      ┌──────────────┴──────────────┐
  [Face Detected]            [No Face Detected]
      │                             │
Square Bounding Box         Check Aspect Ratio
(1.25x Margin Padding)              │
      │                     ┌───────┴───────┐
Reflection Padding      [Non-Square]     [Square]
(cv2.BORDER_REFLECT)        │               │
      │                Center Square     Direct Resize
      └───────────────> Crop (1:1)       (Zero Stretch)
                             │               │
                             └───────┬───────┘
                                     │
                             Bilinear Resize
                               (224 x 224)
                                     │
                             ImageNet Normalize
```

### Preprocessing Specifications:
1. **Face Detection**: `cv2.FaceDetectorYN` with confidence threshold $\tau = 0.60$, NMS threshold $0.30$.
2. **Aspect-Ratio-Preserving Square Bounding Box**:
   - Takes the maximum facial dimension $S = \max(W_{\text{box}}, H_{\text{box}})$.
   - Expands by a 25% margin padding factor: $S_{\text{pad}} = 1.25 \times S$.
   - Centers the square crop at facial center $[C_x, C_y]$.
3. **Boundary Padding**:
   - If the bounding square extends beyond image margins, boundaries are padded using mirror reflection (`cv2.BORDER_REFLECT_101`).
   - Prevents anisotropic squashing, edge distortion, or black-border artifacts.
4. **Deterministic Fallbacks**:
   - If no face is detected with confidence $\ge 0.60$:
     - Non-square image ($\text{AR} \ne 1.0$): Crops the maximal centered square region (`center_square_crop`).
     - Native square image ($\text{AR} = 1.0$): Directly resizes without crop (`square_direct`).
   - **Zero anisotropic stretching is guaranteed** across all images.
5. **Telemetry Recording**:
   - Every processed sample records: `face_detection_status` (`yunet_detected` vs `no_face_detected`), `fallback_used` (`none`, `center_square_crop`, `square_direct`), and `confidence_score`.

---

## 3. Empirical Pilot Experiment Design & Sample Counts

To validate the corrected pipeline before full ingestion, a balanced pilot experiment ([`scripts/run_phase7_pilot.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/scripts/run_phase7_pilot.py)) was executed on **100 representative images** across five distinct cohorts:

| Cohort Name | Label | Generative Family | Source Dataset | Raw Resolution Range | Aspect Ratio Range |
|---|:---:|:---:|:---:|:---:|:---:|
| **IMDB-WIKI_Real** | REAL | Authentic Human | OpenRL / IMDB-WIKI | $400 \times 557$ to $550 \times 825$ | $1.392 - 1.502$ (Vertical) |
| **FFHQ_Real** | REAL | Authentic Human | Kaggle / FFHQ | $256 \times 256$ to $1024 \times 1024$ | $1.000$ (Square) |
| **StyleGAN2_Fake** | FAKE | GAN Upsampling | Kaggle / StyleGAN2 | $256 \times 256$ | $1.000$ (Square) |
| **StableDiffusion_v1.5_Fake** | FAKE | Latent Diffusion | OpenRL / DeepFakeFace | $512 \times 512$ | $1.000$ (Square) |
| **InsightFace_Swap_Fake** | FAKE | Identity Swap Autoenc | OpenRL / DeepFakeFace | $512 \times 512$ | $1.000$ (Square) |

---

## 4. Empirical Pilot Results & Telemetry Findings

### 4.1 Face Detection & Fallback Distribution

The pilot processed all 100 images without dropped samples or runtime errors:

| Metric | Real Cohorts ($N=40$) | Fake Cohorts ($N=60$) | Overall ($N=100$) |
|---|:---:|:---:|:---:|
| **YuNet Face Detected** | **$39$ / $40$ ($97.5\%$)** | **$59$ / $60$ ($98.3\%$)** | **$98$ / $100$ ($98.0\%$)** |
| **Deterministic Fallback Used** | $1$ / $40$ ($2.5\%$) | $1$ / $60$ ($1.7\%$) | $2$ / $100$ ($2.0\%$) |
| - `center_square_crop` | $1$ (IMDB-WIKI non-square) | $0$ | $1$ |
| - `square_direct` | $0$ | $1$ (InsightFace square) | $1$ |
| **Mean Face Confidence** | $0.8872$ | $0.8984$ | $0.8939$ |
| **Valid Output Shape $(224, 224, 3)$** | **$100\%$ ($40/40$)** | **$100\%$ ($60/60$)** | **$100\%$ ($100/100$)** |
| **Anisotropic Stretching Rate** | **$0.0\%$ ($0/40$)** | **$0.0\%$ ($0/60$)** | **$0.0\%$ ($0/100$)** |

### 4.2 Edge Case & Failure Mode Analysis

Two edge cases triggered fallback handling:
1. **Sample `IMDB_WIKI_real_0013.jpg`** ($400 \times 601$, $\text{AR}=1.5017$): Extreme lateral profile orientation and partial shadow. YuNet facial confidence fell below $0.60$. The pipeline triggered `center_square_crop`, extracting the central $400 \times 400$ square with zero stretching, maintaining proper face proportions.
2. **Sample `InsightFace_Swap_0001.jpg`** ($512 \times 512$, $\text{AR}=1.0000$): Strong motion blur from source video. YuNet confidence fell below $0.60$. Because the image was natively square, the pipeline triggered `square_direct`, scaling directly to $224 \times 224$ without cropping or distortion.

### 4.3 2D DCT Log-Magnitude Energy Analysis (Elimination of Frequency Confounder)

To verify whether the corrected square face cropping eliminated the high-frequency distortion shortcut, we measured the high-frequency normalized DCT energy across all cohorts:

| Cohort | Phase 4/5 Anisotropic Resize Energy | Phase 7 Corrected Square Crop Energy | Frequency Impact |
|---|:---:|:---:|---|
| **IMDB-WIKI Real** | **$0.009420$** | **$0.003980$** | **$-57.7\%$ reduction** (Eliminated artificial squashing energy) |
| **Stable Diffusion v1.5 Fake** | $0.008250$ | $0.003903$ | Aligned with natural facial photography |
| **InsightFace Swap Fake** | $0.008310$ | $0.003975$ | Aligned with natural facial photography |
| **FFHQ Real** | $0.004495$ | $0.004495$ | Native square baseline maintained |
| **StyleGAN2 Fake** | $0.004456$ | $0.004456$ | Native square baseline maintained |

### Key Scientific Finding:
Under anisotropic squashing in Phase 4/5, real IMDB-WIKI images exhibited an energy gap of $+14.2\%$ over diffusion fakes. With corrected aspect-ratio-preserving square cropping, the high-frequency energy of real images ($0.003980$) and diffusion fakes ($0.003903$) **now align naturally**, confirming that the previously observed high-frequency gap was an artifact of rectangular squashing.

---

## 5. Automated Verification Tests

The preprocessing pipeline is covered by unit tests in [`tests/test_face_preprocessor.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/tests/test_face_preprocessor.py):
1. `test_square_image_passthrough`: Verifies that native square images preserve dimensions and channels.
2. `test_blank_image_fallback`: Verifies deterministic fallback on synthetic featureless images without exceptions.
3. `test_face_detection_synthetic`: Verifies face detection, square padding, and metadata consistency on real facial inputs.

All 3 unit tests pass cleanly.

---

## 6. Pilot Verdict & Recommendations

- **Pilot Verdict**: **PASSED**.
- **Recommendation**: Deploy `src/data/face_preprocessor.py` as the official, unified preprocessing pipeline across all multi-generator training and evaluation scripts.
