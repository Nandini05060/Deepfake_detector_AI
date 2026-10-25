# Phase 12E Frequency Branch & Fusion Strategy Audit
## Rigorous Inspection of 2D-DCT Mechanics, Attention Weight Starvation, and Constrained Fusion Benchmarking

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Deepfake Detection Specialist & Computer Vision Researcher  
**Document Identifier:** `reports/phase12_frequency_branch_audit.md`  
**Associated Data:** `reports/phase12_fusion_ablation.csv`  

---

## 1. Executive Summary & Verification Objectives

Phase 11 highlighted an apparent paradox:
1. On localized spatial boundary manipulations (e.g., Stable Diffusion Inpainting and InsightFace Swap), the Frequency Branch achieved **68.5%** and **76.7%** detection recall, massively outperforming the Spatial Branch (**25.5%** and **56.7%**).
2. Yet on Google Gemini / Imagen 3 images, the Frequency Branch collapsed to **5.88%** recall (mean $P = 0.0771$).
3. The overall active model achieved only **28.0%** inpainting recall because the dynamic attention module allocated $\approx 94\%$ weight to the spatial branch and only $\approx 6\%$ to the frequency branch.

Phase 12E audited the complete mathematical and computational implementation of the 2D-DCT pipeline, measured attention weight distributions across all cohorts, and evaluated alternative fusion strategies.

---

## 2. Mathematical & Computational Audit of 2D-DCT

### 2.1 Forward Transform Pipeline: `DiscreteCosineTransform2D` (`src/frequency/dct.py`)
- **Basis Construction:** Orthogonal DCT-II matrix $C \in \mathbb{R}^{N \times N}$ ($N=224$):
  $$C_{0, j} = \sqrt{\frac{1}{N}}, \quad C_{i, j} = \sqrt{\frac{2}{N}} \cos\left( \frac{\pi (2j + 1) i}{2N} \right) \quad \text{for } i \ge 1$$
  Orthogonality verified: $\|C C^T - I\|_\infty < 10^{-6}$.
- **Color-Space Conversion:** Input ImageNet-normalized tensors $[B, 3, 224, 224]$ are converted to luminance $Y = 0.299 R + 0.587 G + 0.114 B$.
- **2D Transform Operator:** Deterministic GPU matrix multiplication:
  $$\text{DCT}(Y) = C \, Y \, C^T$$
- **Log-Magnitude Normalization:**
  $$\text{Spectrum} = \log\left(1.0 + |\text{DCT}(Y)| + 10^{-6}\right)$$
  Followed by sample-level min-max normalization to $[0.0, 1.0]$.
- **Audit Finding on Normalization:** Because the DC component (top-left coefficient $i=0, j=0$) contains the total average luminance, its magnitude is orders of magnitude larger than subtle AC high-frequency coefficients. Min-max normalization compresses high-frequency variations into a narrow dynamic range $[0.0, 0.05]$. For boundary manipulations (inpainting masks), high-frequency seams produce strong localized harmonics that survive this scaling. For whole-image modern diffusion (Imagen 3), spectral roll-off is smooth, producing virtually zero anomalous AC energy.

---

## 3. Attention Weight Inspection & Modality Starvation Analysis

Across the $N=457$ locked test images, actual attention weights assigned by `model.fusion` were extracted:

| Cohort | Sample Size ($N$) | Mean $\alpha_s$ (Raw) | Mean $\alpha_f$ (Raw) | Mean $w_s$ (Bounded) | Mean $w_f$ (Bounded) | Modality Clamping Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Google Gemini Imagen 3** | 17 | 0.9864 | 0.0136 | **0.9378** | **0.0622** | Clamped at minimum floor ($\tau = 0.05$) |
| **StyleGAN2 Synthetic** | 60 | 0.9930 | 0.0070 | **0.9437** | **0.0563** | Clamped at minimum floor |
| **Stable Diffusion v1.5** | 30 | 0.9880 | 0.0120 | **0.9392** | **0.0608** | Clamped at minimum floor |
| **InsightFace Swap** | 30 | 0.9872 | 0.0128 | **0.9385** | **0.0615** | Clamped at minimum floor |
| **SD Inpainting (Held-Out)** | 200 | 0.9869 | 0.0131 | **0.9382** | **0.0618** | Clamped at minimum floor |
| **Authentic Real Controls** | 120 | 0.9879 | 0.0121 | **0.9391** | **0.0609** | Clamped at minimum floor |

### Root Cause of Attention Failure:
- During training, StyleGAN2 artifacts were effortlessly detected by EfficientNet-B0 (95.0% accuracy). The attention head learned to heavily favor $F_s$, pushing raw softmax weight $\alpha_s \to 0.99$.
- Because the minimum floor was set to $\tau = 0.05$, $w_f$ was permanently clamped near $0.06$.
- When presented with Inpainting or Face-Swaps—where the spatial branch fails (25.5% recall) but the frequency branch succeeds (68.5% recall)—the attention head gave only 6% weight to the frequency branch, causing the fused feature vector to follow the failing spatial branch.

---

## 4. Quantitative Evaluation of Fusion Strategies

Evaluated on the exact same locked test samples ($N=457$):

| Strategy | ROC-AUC | Accuracy | F1-Score | Real Specificity | Fake Recall | Inpainting Recall | InsightFace Recall | Gemini Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Spatial-Only** | 0.8526 | 0.5536 | 0.5870 | **90.00%** | 43.03% | 25.50% | 56.67% | 11.76% |
| **B. Frequency-Only** | 0.6431 | 0.5864 | 0.6595 | 70.83% | 54.30% | **68.50%** | **76.67%** | 5.88% |
| **C. Active Attention ($\tau=0.05$)** | 0.8544 | 0.5624 | 0.6000 | 89.17% | 44.51% | 28.00% | 53.33% | 11.76% |
| **D. Equal-Weight ($0.5/0.5$)** | 0.8386 | **0.6346** | **0.6890** | 87.50% | **54.90%** | 44.00% | **66.67%** | **17.65%** |
| **E. Constrained Attention ($\tau=0.25$)** | **0.8556** | 0.6018 | 0.6459 | **90.83%** | 49.26% | 35.00% | 53.33% | 11.76% |
| **F. Constrained Attention ($\tau=0.35$)** | 0.8525 | 0.6105 | 0.6590 | 89.17% | 51.04% | 37.50% | 60.00% | 11.76% |

### Key Scientific Conclusions:
1. **Constrained Floor ($\tau = 0.25$) Achieves Best In-Class ROC-AUC:**  
   Enforcing a minimum 25% floor on both branches prevents modality starvation. It boosts overall accuracy from $56.24\%$ to **$60.18\%$**, increases Inpainting recall from $28.0\%$ to **$35.0\%$**, and achieves the highest ROC-AUC (**0.8556**) while raising real specificity to **90.83%** (FPR dropped to $9.17\%$).
2. **Equal-Weight ($0.5/0.5$) Yields Highest Classification Accuracy:**  
   If operating at fixed $0.50$ threshold, Equal-Weight fusion boosts overall accuracy from $56.24\%$ to **$63.46\%$** (+7.22%) and F1-score from $0.6000$ to **$0.6890$** (+0.089), with Inpainting recall surging to **44.0%** and InsightFace to **66.7%**, while real specificity remains strong at **87.5%**.
3. **Frequency Branch Retention is Scientifically Mandatory:**  
   A spatial-only model is blind to inpainting seams (25.5% recall). The frequency branch provides essential orthogonal forensic evidence for local boundary tampering.
