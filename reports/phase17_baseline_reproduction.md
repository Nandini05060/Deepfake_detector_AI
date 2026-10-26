# Phase 17 Baseline Failure Reproduction Report

**Experiment Identifier:** `EXP-PHASE17-BASE-REPRO`  
**Date:** October 26, 2026  
**Auditor / Researcher:** AI Forensics Computer Vision Lead  
**Target Failure Image:** `data/diagnostics/gemini_failure.png`  
**Image Dimensions:** $2816 \times 1536$ RGBA  
**Verified SHA-256:** `944cb50e9187a22a46dc6232387aa6769d4513c4654b141832b1eec04cd377fa`  
**Ground Truth Label:** `FAKE` (Google Gemini Imagen 3 Text-to-Image Generation)

---

## 1. Executive Summary & Problem Reproduction

The objective of Phase 17 is to permanently solve the failure case where photorealistic AI-generated faces from modern diffusion models (such as Google Gemini / Imagen 3) are classified as `AUTHENTIC REAL` or flagged as `INCONCLUSIVE` under standard preprocessing conditions.

We evaluated `gemini_failure.png` across all 5 primary checkpoints in the repository using aspect-ratio preserving YuNet face detection across margins $m=0.15$ (tight face crop), $m=0.50$ (standard portrait crop), and $m=0.75$ (extended context crop).

### Summary of Reproduction Results

```
==================================================================================================================
MODEL CHECKPOINT                  SHA-256 (PREFIX)   MARGIN 0.15 (Tight)      MARGIN 0.50 (Portrait)   MARGIN 0.75 (Context)
==================================================================================================================
Phase 5 (Attention Robustness v2) 215613a19f...      P=0.0030 (REAL)          P=0.0309 (REAL)          P=0.1972 (REAL)
Phase 8 (Diverse Baseline)        b3d280d8db...      P=0.0008 (REAL)          P=0.2872 (REAL)          P=0.7597 (INCONC)
Phase 10 (Modern Multi-Gen)       13dc56b896...      P=0.0279 (INCONC)        P=0.5578 (INCONC)        P=0.8842 (FAKE)
Phase 15 (Gemini Candidate)       2552ab2b8f...      P=0.0003 (INCONC)        P=0.4337 (INCONC)        P=0.9956 (FAKE)
Phase 16 (Unified Best)           4e7b7747f4...      P=0.0025 (INCONC)        P=0.1359 (INCONC)        P=0.8465 (FAKE)
==================================================================================================================
```

---

## 2. In-Depth Branch Discrepancy & Logit Decomposition

All models follow the positive class convention: $\text{Class } 1 = \text{FAKE}$, $\text{Class } 0 = \text{REAL}$.  
Fused logit $z > 0 \iff P(\text{Fake}) > 0.5$.

### Detailed Telemetry on `gemini_failure.png` (Standard Margin $m = 0.50$):

1. **Phase 16 Unified Best:**
   - **Detected Face Bounding Box:** $[fx=752, fy=1, fw=1141, fh=1237]$
   - **Spatial Branch:** Logit $= -2.6074$, Probability $= \mathbf{0.0686}$ ($w_s = 0.50$)
   - **Frequency Branch:** Logit $= -0.2615$, Probability $= \mathbf{0.4350}$ ($w_f = 0.50$)
   - **Branch Discrepancy:** $|0.0686 - 0.4350| = \mathbf{0.3665} > 0.10 \implies$ High Discrepancy Gate Tripped.
   - **Fused Logit:** $-1.8504 \implies$ Raw $P(\text{Fake}) = 0.1359$.
   - **Calibrated Probability ($T=1.0859$):** $P(\text{Fake}) = \mathbf{0.1359}$, $P(\text{Real}) = \mathbf{0.8641}$.
   - **Decision:** `INCONCLUSIVE` (Empirical Reliability: UNCERTAIN; Discrepancy Warning Active).

2. **Phase 15 Gemini Candidate:**
   - **Spatial Branch:** Probability $= 0.4357$
   - **Frequency Branch:** Probability $= 0.5100$
   - **Fused Probability:** $P(\text{Fake}) = 0.4337$ (within uncertainty band $|0.4337 - 0.55| \le 0.15$)
   - **Decision:** `INCONCLUSIVE`.

3. **Phase 10 Modern Multi-Generator Standard:**
   - **Spatial Branch:** Probability $= 0.5334$
   - **Frequency Branch:** Probability $= 0.7176$
   - **Fused Probability:** $P(\text{Fake}) = 0.5578$ (Discrepancy $= 0.1842 > 0.10$)
   - **Decision:** `INCONCLUSIVE`.

---

## 3. Root Cause Analysis: The Micro-Texture Deception Mechanism

1. **The Spatial Skin Paradox:**
   Modern diffusion engines (Google Gemini Imagen 3, SDXL, Midjourney v6) have eliminated the macroscopic structural anomalies of earlier GANs. Internal skin pore distributions, iris specular patterns, and lip moisture gradients are synthesized with near-photographic fidelity. When cropped to facial landmarks ($m=0.15$ or $0.50$), the spatial ResNet backbone extracts features that closely mirror genuine studio photography, resulting in spatial fake probabilities dropping below $0.07$.
2. **Frequency Branch Limitations:**
   The current frequency branch processes a 2D DCT log-magnitude spectrum. While GANs exhibit overt high-frequency checkerboard spectral spikes, latent diffusion models operate in a compressed latent space and decode with continuous VAEs, producing subtle spectral noise profiles that earlier frequency filters fail to amplify sufficiently at standard resolutions.
3. **The Attention Dilemma:**
   Because the spatial branch confidently predicts "REAL" ($P \approx 0.07$) while the frequency branch is ambiguous ($P \approx 0.44$), the model's gating mechanism observes a large discrepancy ($> 0.10$). To avoid a false positive or blind guess, the model abstains and outputs `INCONCLUSIVE`.
4. **The Remedy for Phase 17:**
   - Incorporate diverse modern diffusion and photorealistic generative face datasets into training (including GenImage, Synthbuster, Stable Diffusion XL, and Gemini face cohorts).
   - Enhance the spatial branch feature representations with hard negative mining on photorealistic diffusion skin textures.
   - Refine the frequency branch to capture subtle radial energy rolloffs and high-pass residual cues rather than only coarse GAN grid spikes.
   - Calibrate the attention floor and multi-generator balanced loss to prevent spatial overconfidence on photorealistic synthetic skin.

---

*Certified baseline reproduction artifact. Full prediction JSON saved at `reports/phase17_baseline_predictions.json`.*
