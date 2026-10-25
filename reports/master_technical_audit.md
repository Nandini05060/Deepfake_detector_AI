# LX-DFD Master Technical Audit & Scientific Forensic Review
## From a College Prototype to a Research-Grade AI Forensics Platform

**Author:** Senior Machine Learning Forensics Specialist & Computer Vision Evaluator  
**Platform:** LX-DFD (AI Deepfake Forensics Laboratory)  
**Date of Technical Audit:** October 2026  
**Document Identifier:** `REP-AUDIT-2026-V4`  
**Active Production Checkpoint:** `checkpoints/lxfd_phase10_modern_generators_best.pt`  
**Cryptographic Hash (SHA-256):** `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05`  
**Active Calibration Vector:** `T = 1.6344` (`checkpoints/calibration_params_phase10.json`)

---

## Executive Summary & Audit Mandate

LX-DFD was initially conceived as a dual-branch neural architecture combining spatial features (via an ImageNet-pretrained **EfficientNet-B0**) and frequency domain representations (via a **2D Discrete Cosine Transform + 4-layer CNN**) coupled by a learned attention fusion layer. Early experimental prototypes reported approximately **99.1% in-domain accuracy** on synthetic GAN faces.

This technical audit was commissioned to eliminate academic complacency, challenge unverified assumptions, and upgrade LX-DFD into a research-grade, forensically grounded platform. The objective is **not** to build cosmetic dashboards that mask underlying classifier vulnerabilities. The objective is to make LX-DFD **measurable, reproducible, explainable, and brutally honest about the real-world limits of deepfake detection.**

---

## 1. The Three Highest-Impact Technical Weaknesses

Before modifying code or deploying models, this audit isolated the three most critical points of forensic failure in the original system:

### Weakness 1: Frequency Branch Attention Floor vs. Diffusion Generalization Collapse
* **The Vulnerability:** The 2D-DCT frequency branch was trained almost exclusively on legacy GAN artifacts (StyleGAN upsampling grid lines and checkerboard spectral spikes). In controlled ablations, when evaluated against unseen modern latent diffusion architectures (Stable Diffusion v1.5, SDXL, FLUX, Gemini Imagen 3), the frequency branch alone achieved an **ROC-AUC of only 0.4676**—worse than random guessing, exhibiting an inverted spectral bias.
* **Why It Matters Forensically:** While the architecture introduced a minimum attention constraint floor ($\tau \ge 0.05$) to prevent the frequency weight $w_f$ from mathematically extinguishing, forcing the model to retain a 5% frequency contribution when encountering diffusion noise actively injected error. Spatial features had to overcompensate for frequency misdirection. Claiming that "dual-branch frequency fusion inherently improves generalization" was scientifically untrue without modern generator-balanced fine-tuning.

### Weakness 2: Boundary/Crop Margin Sensitivity (The Gemini False Negative Flaw)
* **The Vulnerability:** Under tight facial cropping (margin 0.15), internal facial landmarks (pupils, iris geometry, nose bridge, oral cavity) are isolated while discarding forehead hairlines, ear lobes, and jawline-to-background blending seams. When evaluating hyper-realistic diffusion models (e.g., Google Gemini Imagen 3 portraits), the interior facial texture is photorealistic and smooth, causing the classifier to output a **0.0015 Fake Probability (99.85% Real)**. When the crop margin was expanded to standard portrait context (0.50), the subtle diffusion blending artifacts and hair boundary upsampling irregularities shifted the prediction to **0.7210 Fake Probability**—an absolute prediction swing ($\Delta$) of **+0.72**!
* **Why It Matters Forensically:** A deepfake detection system whose verdict flips between Authentic Real and Synthetic Fake merely due to an arbitrary 15-pixel crop bounding box difference cannot be presented in a legal, journalistic, or intelligence context. Preprocessing sensitivity must be explicitly quantified, surfaced to investigators, and bounded.

### Weakness 3: Background Shortcut Leakage vs. Facial Biometric Forensic Grounding
* **The Vulnerability:** In legacy training benchmarks (FFHQ authentic faces vs. StyleGAN synthetic faces), authentic portraits feature organic real-world backgrounds with natural depth-of-field lens bokeh, natural lighting gradients, and realistic environmental clutter. In contrast, early GAN faces feature swirling painterly abstractions, watermarked borders, or synthetic noise patches.
* **Why It Matters Forensically:** When the face was digitally occluded with a neutral gray mask, the model still predicted FAKE on legacy samples with **over 64% confidence** based purely on background perimeter pixels! The classifier had partially learned to classify photographic environments rather than facial manipulation. A forensic tool that flags an authentic person as manipulated because their webcam background resembles synthetic bokeh is dangerous and unreliable.

---

## 2. Active Model Architecture & Cryptographic Audit

### Baseline vs Active Production Checkpoint

| Attribute | Legacy Prototype (Phase 1–4) | Hardened Production (Phase 10) |
| :--- | :--- | :--- |
| **Active Checkpoint File** | `lxfd_best.pt` (Phase 1) | `checkpoints/lxfd_phase10_modern_generators_best.pt` |
| **SHA-256 Checksum** | Stale / Unversioned | `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05` |
| **Spatial Backbone** | EfficientNet-B0 (1280 features) | EfficientNet-B0 (1280 features) with frozen early stages |
| **Frequency Backbone** | 2D DCT + 4-Conv CNN (512 features) | 2D DCT + Multi-Scale Spectral Conv + Spectral Regularization |
| **Fusion Mechanism** | Unconstrained Softmax Attention | Attention Fusion with Constrained Entropy Floor ($\tau = 0.05$) |
| **Calibration Status** | Uncalibrated Softmax ($T=1.0$) | Temperature Scaled ($T=1.6344$, in-domain ECE = 0.0228) |
| **Face Preprocessing** | Ad-hoc OpenCV Haar Cascade | OpenCV YuNet ONNX Face Detector (margin 0.50 default) |
| **Input Resolution** | $224 \times 224 \times 3$ RGB | $224 \times 224 \times 3$ RGB (Bicubic Anti-Aliased) |

### Formal Mathematical Architecture

Given an RGB facial crop $x \in \mathbb{R}^{3 \times 224 \times 224}$:

1. **Spatial Representation:**
   $$f_s(x) = \text{EfficientNet-B0}(x) \in \mathbb{R}^{1280}$$
2. **Frequency Representation:**
   $$X_{DCT} = \mathcal{D}_{2D}(x) = \log(1 + |\text{DCT}(x)|)$$
   $$f_f(x) = \text{SpectralCNN}(X_{DCT}) \in \mathbb{R}^{512}$$
3. **Attention Fusion Mechanism:**
   $$h = [f_s(x) \,\|\, f_f(x)] \in \mathbb{R}^{1792}$$
   $$\alpha = \text{Softmax}(W_a h + b_a) = [w_s, w_f]$$
   $$\text{Constrained Weights: } \tilde{w}_s = \max(\tau, w_s), \quad \tilde{w}_f = \max(\tau, w_f), \quad \text{normalized so } \tilde{w}_s + \tilde{w}_f = 1$$
4. **Fused Feature Vector & Logit:**
   $$f_{\text{fused}} = [\tilde{w}_s \cdot W_{proj,s} f_s(x) \,\|\, \tilde{w}_f \cdot W_{proj,f} f_f(x)] \in \mathbb{R}^{512}$$
   $$z = W_{\text{cls}} f_{\text{fused}} + b_{\text{cls}} \in \mathbb{R}^1$$
5. **Calibrated Forensic Probability:**
   $$\hat{P}(\text{Fake} \mid x) = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z / 1.6344}}$$

---

## 3. Dataset Composition, Integrity & Leakage Audit

The core dataset corpus was subjected to cryptographic and perceptual hashing to identify cross-split contamination and duplicate leakage:

* **Total Audited Images:** 12,890 images
  * Authentic Real: 5,890 images (FFHQ, CelebA-HQ, real web portraits)
  * Synthetic Fake: 7,000 images (StyleGAN2, StyleGAN3, Stable Diffusion v1.5, InsightFace Swap)
* **Cryptographic Deduplication:**
  * Exact MD5/SHA256 duplicates detected: **4 duplicate pairs** (quarantined into `data/quarantine/`).
  * Near-duplicate perceptual hash collision rate across splits: **0.00%**.
* **Split Partition Protocol:**
  * Train Set: 8,890 images (70% stratified by generator)
  * Validation Set: 2,000 images (15% used strictly for early stopping and temperature calibration)
  * In-Domain Test Set: 2,000 images (15% held-out until final evaluation)
  * External Multi-Generator Benchmark: 551 images completely isolated from training.
  * Held-Out Gemini Imagen 3 Evaluation Benchmark: 4 verified Gemini Imagen 3 images matched with 60 authentic real controls.

---

## 4. Current Preprocessing & YuNet Localization Pipeline

A major finding of the audit was that discrepancies between training preprocessing and deployment preprocessing severely impaired inference reliability. The finalized, frozen preprocessing pipeline is:

```
Raw Input Image (JPEG / PNG / WebP)
          │
          ▼
OpenCV YuNet ONNX Face Detection (score_threshold=0.60, nms_threshold=0.30)
          │
    ┌─────┴───────────────────────────────────┐
    ▼                                         ▼
Face Detected (98.4%)                 No Face Detected (1.6%)
    │                                         │
Bounding Box [fx, fy, fw, fh]          Central Aspect-Preserving 60% Crop
    │                                         │
Context Expansion: Margin = 0.50              │
    dx = fw * 0.50, dy = fh * 0.50            │
    [x1, y1, x2, y2] clamped to image dims    │
    └─────────────────┬───────────────────────┘
                      ▼
            RGB Crop Extraction
                      ▼
       Bicubic Anti-Aliased Resize to 224×224
                      ▼
   ImageNet Normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])
                      ▼
       Dual-Branch PyTorch Inference Engine
```

---

## 5. Summary of Audit Recommendations & Delivered Improvements

1. **Separation of Attention Weights from Probabilities:** In the original UI, the attention weights $w_s$ and $w_f$ were confusingly presented as branch probabilities. The backend and frontend now explicitly separate branch predictions ($P_s(\text{Fake}), P_f(\text{Fake})$) from dynamic fusion attention weights ($w_s, w_f$).
2. **Crop Sensitivity Lab Built & Integrated:** Researchers can now evaluate an image simultaneously at margins 0.15, 0.50, and 0.75 to observe prediction drift before certifying a forensic result.
3. **Dataset Shortcut Detector Implemented:** Evaluates background masking, facial interior zooming, and frequency attenuation to determine whether model decisions are biometrically grounded.
4. **Counterfactual Robustness Stress Testing:** Measures single-image resilience under JPEG compression (Q=95, 75, 40), Gaussian blur, sensor noise, downsampling, and chroma stripping.
5. **Temperature Scaling Integrated:** Deployed calibrated temperature $T = 1.6344$, reducing in-domain calibration error from 14.8% to 2.28%.
6. **Gemini Limitation Formally Documented:** Disclosed that Gemini evaluation is restricted to $N=4$ images and prohibited marketing claims of general diffusion omniscience.

---

*Report certified by LX-DFD Scientific Evaluation Taskforce.*
