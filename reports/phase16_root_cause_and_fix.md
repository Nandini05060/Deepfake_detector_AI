# Phase 16 Root Cause Analysis & Forensic Remediation Report

## 1. Executive Summary

When a user uploaded a new Gemini Imagen 3-generated portrait (`gemini_user_uploaded_portrait.jpg`) into the Explainability Studio, the system produced an unexpected output:
- **Classification:** `INCONCLUSIVE`
- **Fake Probability:** `0.0%` (calibrated `0.52%`)
- **Real Probability:** `100.0%` (calibrated `99.48%`)
- **Spatial Fake Probability:** `0.1%` (raw `0.77%`)
- **Frequency Fake Probability:** `41.5%` (raw `43.06%`)
- **Confidence:** `LOW`

This investigation confirms:
1. **No Software Corruption:** There is no head inversion, sign-flip bug, or NaN value corruption.
2. **Mathematical Coherence:** The $0.0\%$ fake probability is the direct mathematical result of an attention mechanism assigning $75\%$ weight to a strongly negative spatial logit ($-4.86$), which overpowered the frequency branch ($-0.28$), followed by temperature calibration sharpening ($T = 0.7169 < 1.0$).
3. **Safety Policy Working as Designed:** The Vigilant Forensic Policy successfully intercepted the modality contradiction (`spatial < 0.10` and `freq >= 0.28`), preventing a catastrophic false-real classification and correctly marking the image `INCONCLUSIVE`.
4. **Physical Root Cause Identified:** Gemini Imagen 3 synthesizes ultra-photorealistic skin micro-texture (pores, freckles, micro-wrinkles) on the central face, which completely fools spatial CNNs trained on legacy GANs. However, diffusion manipulation artifacts remain prominent in **peripheral hair strands, collar edges, and background bokeh**. On wide-aspect images ($1024 \times 558$), standard face cropping clamps vertically to $[0, 558]$, isolating only the deceiving inner facial skin. Expanding context (margin 0.75 or full frame) exposes these boundary artifacts, jumping the spatial fake probability from **$0.77\%$ to $90.48\%$** and the calibrated fake probability to **$90.61\%$ (`LIKELY_FAKE`)**.
5. **The Real-Face Specificity Constraint:** Blindly forcing a 0.75 margin globally is statistically unviable because textured backgrounds in historical photographs (IMDB-WIKI) trigger false fakes ($0\% \to 22.5\%$).

---

## 2. Root Cause Deep Dive: Working Gemini vs Problematic Gemini

To understand why the benchmark Gemini sample succeeded while the uploaded Gemini failed, we compare their geometric, architectural, and spectral traces:

### Comparative Table: Benchmark vs Uploaded Gemini Sample

| Diagnostic Metric | Known Benchmark Sample (`gemini_imagen3_sample.png`) | Problematic Uploaded Sample (`gemini_user_uploaded_portrait.jpg`) |
| :--- | :--- | :--- |
| **Canvas Dimensions** | $2816 \times 1536$ (1.83:1) | $1024 \times 558$ (1.84:1) |
| **YuNet Face Bounding Box** | $[1314, 476, 563, 560]$ | $[391, 94, 246, 337]$ |
| **Face-to-Canvas Height Ratio** | **$36.46\%$** (Centrally framed) | **$60.39\%$** (Fills vertical frame) |
| **Margin 0.50 Crop Coords** | $[1033, 196, 2158, 1316]$ | $[268, 0, 760, 558]$ |
| **Vertical Boundary Clamping** | **No** ($196 > 0$ and $1316 < 1536$) | **Yes: Clamped to $y_1=0$ and $y_2=558$** |
| **Crop Dimensions (W x H)** | $1125 \times 1120$ px (Aspect: $1.0045$, Square) | $492 \times 558$ px (Aspect: $0.8817$, Distorted) |
| **Visible Context Included** | Full head, curly hair perimeter, background | Internal face only; hair perimeter clipped |
| **Spatial Logit / Probability** | **$+7.2727$ / $99.93\%$ Fake** | **$-4.8565$ / $0.77\%$ Fake** |
| **Frequency Logit / Probability**| $-0.7647$ / $31.76\%$ Fake | $-0.2796$ / $43.06\%$ Fake |
| **Attention Weights ($w_s, w_f$)** | $w_s = 0.7494, w_f = 0.2506$ | $w_s = 0.7485, w_f = 0.2515$ |
| **Fused Logit / Raw Fake Prob** | $+5.3432$ / $99.52\%$ Fake | $-3.7624$ / $2.27\%$ Fake |
| **Calibrated Fake Prob ($T=0.7169$)**| **$99.94\%$ Fake** | **$0.52\%$ Fake** |
| **Final Forensic Verdict** | **`LIKELY_FAKE`** (Confident) | **`INCONCLUSIVE`** (Divergence Caught) |

---

## 3. Spatial Sensitivity to Context Padding

To prove the localization of diffusion artifacts, we evaluated `gemini_user_uploaded_portrait.jpg` across multiple crop windows centered on the face:

```
[Margin 0.15] (Face-only, 350x350):  Spatial= 0.03% | Freq= 54.7% | Calibrated Fake=  0.02%
[Margin 0.30] (Tight oval, 450x450): Spatial= 0.50% | Freq= 49.4% | Calibrated Fake=  0.40%
[Margin 0.50] (Standard, 492x558):   Spatial= 0.77% | Freq= 43.1% | Calibrated Fake=  0.52% (INCONCLUSIVE)
[Margin 0.75] (Extended, 614x558):   Spatial= 90.48%| Freq= 45.1% | Calibrated Fake= 90.61% (LIKELY_FAKE)
[Full Canvas] (Uncropped, 1024x558): Spatial= 99.17%| Freq= 55.3% | Calibrated Fake= 99.39% (LIKELY_FAKE)
```

### Forensic Takeaway
1. **The Internal Face Skin is an Effective Adversary:** Diffusion generators produce skin texture that mimics real camera optics, lacking high-frequency checkerboard artifacts.
2. **The Manipulation Footprint Resides at the Boundary:** Latent diffusion models struggle with fine hair boundary blending against out-of-focus background bokeh. When those regions enter the spatial CNN's receptive field, the model detects the manipulation with $>90\%$ confidence.

---

## 4. The Real-Face Specificity Constraint & The IMDB-WIKI Trade-off

A naive engineering response would be to simply increase the face crop margin to 0.75 across the entire backend. However, rigorous benchmark testing across genuine human portraits reveals why this violates forensic safety:

### Benchmark Evaluation (N=40 per set)

| Dataset & Margin Condition | Fake Decisions (FP) | Real Decisions (TN) | Inconclusive | Mean Fake Probability |
| :--- | :--- | :--- | :--- | :--- |
| **FFHQ Real (Margin 0.50)** | **0** (0.0% FP) | 9 | 31 | 0.07% |
| **FFHQ Real (Margin 0.75)** | **0** (0.0% FP) | 8 | 32 | 0.07% |
| **IMDB-WIKI Real (Margin 0.50)** | **0** (0.0% FP) | 1 | 39 | 10.66% |
| **IMDB-WIKI Real (Margin 0.75)** | **9 (22.5% False Fakes!)** | 6 | 25 | **29.33%** |

### Critical Finding
In historical photographs from IMDB-WIKI (celebrity portraits with textured backgrounds, film grain, and complex lighting), expanding the context margin to 0.75 causes background noise to be misclassified as generative diffusion artifacts, causing **22.5% false positive fake classifications**.

Therefore, the system policy strictly forbids globally forcing wider margins to mask generalization shortcomings.

---

## 5. Software Improvements Implemented

While the underlying model behavior is a genuine generalization limitation, the following software bugs and inconsistencies were identified and remediated:

### 5.1 Inconclusive Epistemic Confidence Calculation Bug
In `src/models/inference.py`:
- **Bug:** When `is_uncertain = True`, line 148 executed:
  `confidence = float(max(fake_prob, real_prob))`
  When `fake_prob = 0.0052`, `real_prob = 0.9948`, so `confidence` returned `0.9948` (99.5%) despite `confidence_level` being `"LOW"`.
- **Fix:** When `is_uncertain` is True, numerical confidence must represent epistemic certainty:
  $$\text{epistemic\_confidence} = 1.0 - 2 \cdot |P - 0.50|$$
  or be capped at a low baseline ($\le 0.50$) to reflect total diagnostic ambiguity.

### 5.2 Clamped Crop Aspect Ratio Normalization
In `app/backend/main.py`:
- **Improvement:** When an image's vertical dimension is clamped by canvas boundaries ($y_1 = 0$ or $y_2 = h$), the horizontal margin $dx$ was previously restricted to $fw \times margin$. By calculating a square-normalized bounding box centered on the face, the crop preserves the natural $1:1$ face aspect ratio without introducing artificial horizontal squeezing.

### 5.3 Multi-Crop Context Advisory in Explainability Studio
- When the backend encounters modality divergence (`spatial_prob < 0.10` and `freq_prob >= 0.28`), the API now surfaces a `boundary_clamping_detected` flag in `detection_metadata`, recommending the user evaluate the `/crop_stability` tab to observe how peripheral context influences the forensic verdict.

---

## 6. Recommendations for Controlled Training (Phase 17)

To solve this generalization challenge at the model level without degrading real-face specificity:

1. **Inner Facial Skin Patch Training:** In Phase 17, augment the training distribution by extracting isolated inner facial skin crops ($128 \times 128$ px, excluding hair and background) from both genuine (FFHQ) and Gemini Imagen 3 faces.
2. **Frequency-Branch Re-weighting:** Increase the attention floor from `min_floor = 0.25` to `min_floor = 0.40` during fine-tuning on modern diffusion models, preventing the spatial branch from completely silencing the spectral branch.
3. **Dual-Scale Contrastive Feature Learning:** Enforce feature alignment between tight face crops and context-rich portrait crops during training.
