# Phase 12D Multi-Scale Spatial Architecture & Ablation Report
## Design, Tensor Flow, Mathematical Formulation, and Empirical Ablation of LXDFD-MultiScale-12

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Computer Vision Researcher & Deepfake Detection Specialist  
**Document Identifier:** `reports/phase12_multiscale_architecture.md`  
**Associated Data:** `reports/phase12_multiscale_ablation.csv`  

---

## 1. Architectural Motivation & Design Philosophy

The controlled resolution experiment in Phase 12C conclusively demonstrated that:
1. High-resolution facial crops ($224 \times 224$) preserve high-frequency authentic camera sensor noise, achieving **89.17% specificity on real photographs**, but modern latent diffusion models (e.g., Google Imagen 3) synthesize naturalistic pore structures that trick the spatial branch into predicting real ($11.76\%$ recall).
2. Pure low-resolution facial crops ($112 \times 112$) eliminate deceptive microtextures, increasing Gemini recall to **88.24%**, but destroy sensor noise, causing real-image specificity to collapse to **70.83%** ($29.17\%$ false positive rate).

To capture the benefits of macro-geometry without catastrophically discarding fine sensor noise, **LXDFD-MultiScale-12** was engineered as an asymmetric multi-scale spatial-frequency architecture.

---

## 2. Architecture Specifications: `LXDFDMultiScale12`

```
                                 Aligned Face Crop
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
           Path A (Detail)        Path B (Global)        Frequency Branch
           (B, 3, 224, 224)       (B, 3, 112, 112)       (B, 3, 224, 224)
                 │                       │                       │
          EfficientNet-B0         EfficientNet-B0            2D-DCT II
         Local skin detail,      Macro facial geom,       Log-magnitude
          camera shot noise       illumination symmetry    Orthogonal matrix
                 │                       │                       │
           Conv2d / MBConv         Conv2d / MBConv        4-Stage ConvNet
          AdaptiveAvgPool2d       AdaptiveAvgPool2d       AdaptiveAvgPool2d
                 │                       │                       │
                 ▼                       ▼                       ▼
            F_detail ∈ ℝ^256       F_global ∈ ℝ^256        F_freq ∈ ℝ^256
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                         MultiScaleSpatialFusion Module
                         Concat [F_d, F_g, F_f] ∈ ℝ^768
                         Linear(768, 256) → GELU → Linear(256, 3)
                         Softmax → [α_d, α_g, α_f]
                         Constrained Floor (τ = 0.20):
                         w_i = 0.20 + 0.40 · α_i
                                         ▼
                         Fused Feature Vector F_fused ∈ ℝ^256
                         F_fused = w_d · F_d + w_g · F_g + w_f · F_f
                                         ▼
                                  Classifier Head
                         Linear(256, 128) → GELU → Dropout(0.3) → Linear(128, 1)
                                         ▼
                                    Logit z ∈ ℝ^1
```

### 2.1 Parameter Count & Computational Footprint
- **Weight Sharing:** Path B (Global Structure) shares weights with Path A (Facial Detail), as EfficientNet-B0 naturally accommodates variable input resolutions via adaptive average pooling.
- **Backbone Parameters:** $4.01\text{M}$ parameters (identical to single-branch baseline).
- **Frequency ConvNet Parameters:** $0.78\text{M}$ parameters.
- **Multi-Scale Fusion Head Parameters:** $0.20\text{M}$ parameters (`Linear(768, 256)` + `Linear(256, 3)`).
- **Classifier Head Parameters:** $0.03\text{M}$ parameters.
- **Total Model Parameters:** $5.03\text{M}$ parameters.
- **Inference Latency:** $14.2\text{ms}$ on NVIDIA RTX 3050 GPU ($70\text{ FPS}$).

---

## 3. Quantitative Multi-Scale Spatial Ablation Results

Evaluated on the locked final test set ($N=457$):

| Configuration | ROC-AUC | PR-AUC | Accuracy | F1-Score | Real Specificity | Real FPR | Gemini Recall | Gemini (k/N) | Gemini Mean $P$ | SD Inpainting | InsightFace | SD v1.5 | StyleGAN2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Active Baseline (Single-Scale Attention)** | **0.8544** | **0.9329** | 56.24% | 0.6000 | **89.17%** | **10.83%** | 11.76% | 2/17 | 0.2134 | 28.00% | 53.33% | 63.33% | **95.00%** |
| **Candidate: Detail-Only ($224 \times 224$)** | 0.8255 | 0.9240 | 62.14% | 0.6802 | 83.33% | 16.67% | 41.18% | 7/17 | 0.4588 | 40.00% | 66.67% | 70.00% | 93.33% |
| **Candidate: Global-Only ($112 \times 112$)** | 0.6253 | 0.8217 | 64.11% | 0.7380 | 51.67% | **48.33%** | **100.0%** | 17/17 | 0.9749 | **57.50%** | **80.00%** | **93.33%** | 78.33% |
| **Candidate: Equal-Weight Dual-Spatial** | 0.7810 | 0.9053 | **68.49%** | **0.7584** | 72.50% | 27.50% | **94.12%** | 16/17 | 0.8695 | 51.50% | **80.00%** | 90.00% | 93.33% |
| **Candidate: LXDFD-MultiScale-12 (Full)** | 0.8255 | 0.9212 | 63.68% | 0.7004 | 80.83% | 19.17% | **47.06%** | 8/17 | 0.5409 | 41.50% | **80.00%** | 76.67% | 93.33% |

---

## 4. Scientific Findings & Engineering Assessment

1. **The Dangerous Allure of Global-Only ($112 \times 112$):**
   - Candidate Global-Only achieves $100\%$ detection on Gemini ($17/17$) and strong numbers on other synthetic generators.
   - However, its false positive rate on real photographs is **$48.33\%$**—it misclassifies roughly half of all genuine camera pictures as deepfakes. Its ROC-AUC collapses to $0.6253$. This configuration is clinically and forensically unacceptable.
2. **Performance of Candidate LXDFD-MultiScale-12:**
   - Multi-scale spatial-frequency integration quadruples Gemini detection recall from **$11.76\%$ to $47.06\%$** ($8/17$), shifts mean predicted fake probability from $0.2134$ to **$0.5409$**, and boosts overall test accuracy from $56.24\%$ to **$63.68\%$**.
   - It substantially improves cross-generator generalization on unseen SD Inpainting ($41.5\%$ vs $28.0\%$) and InsightFace Swap ($80.0\%$ vs $53.3\%$).
   - However, because the global structural branch introduces lower-resolution representations, real-image specificity decreases from $89.17\%$ to **$80.83\%$** (an $8.34\%$ increase in false alarms).
3. **Model Selection Adjudication (Part 11 Compliance):**
   - Under a strict forensic validation criterion that prioritizes avoiding false accusations against authentic human photographs (requiring specificity $\ge 88\%$), replacing the production backbone entirely with `LXDFD-MultiScale-12` incurs an unacceptable false-alarm penalty on authentic portraits.
   - Conversely, **Constrained Fusion ($\tau = 0.25$)** on the verified checkpoint achieves **ROC-AUC = 0.8556** and **Real Specificity = 90.83%**, representing the optimal production operating mode.
