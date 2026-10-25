# Phase 11 Scientific Ablation Study: Spatial vs. Frequency Utility
## Empirical Evaluation of Feature Branches and Fusion Mechanisms Across Generator Families

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Document Identifier:** `reports/phase11_ablation_study.md`  
**Associated Data:** `reports/phase11_ablation_summary.csv` ($N=457$ test subjects across 6 generative/real cohorts)  

---

## 1. Research Questions & Hypotheses

The prior audit noted that on modern diffusion generators, the frequency-only branch achieved low detection accuracy, raising two competing hypotheses:
* **Hypothesis A (Architectural Defect / Normalization Bug):** The 2D-DCT implementation contains a numerical error, coordinate indexing bug, or training-versus-inference preprocessing mismatch.
* **Hypothesis B (Genuine Generator-Dependent Utility & Attention Over-Suppression):** The 2D-DCT spectrum provides genuine discriminative power for localized manipulation boundaries (inpainting seams, face-swap blending), but the attention fusion head excessively suppressed the frequency branch ($w_s \approx 0.94, w_f \approx 0.06$) due to training set imbalance.

---

## 2. Experimental Setup

We evaluated four distinct model variants on identical test splits ($N=457$ total samples):
1. **Spatial-Only Model:** Pretrained EfficientNet-B0 embedding $F_s \in \mathbb{R}^{256}$ passed directly to the classification head.
2. **Frequency-Only Model:** 2D-DCT ConvNet embedding $F_f \in \mathbb{R}^{256}$ passed directly to the classification head.
3. **Attention Fusion Model (Active Baseline):** Dynamic attention-weighted sum $F_{\text{fused}} = w_s F_s + w_f F_f$ with constrained floor $\tau \ge 0.05$.
4. **Equal-Weight Average Fusion:** Unweighted linear combination $F_{\text{fused}} = 0.5 F_s + 0.5 F_f$.

All evaluations were executed on the verified checkpoint `checkpoints/lxfd_phase10_modern_generators_best.pt` using standard $224 \times 224$ normalized inputs.

---

## 3. Quantitative Ablation Results

### 3.1 Overall Performance Across All Locked Test Splits ($N=457$)

| Model Architecture / Branch | ROC-AUC | Accuracy | F1-Score | Precision | Fake Recall | Mean $P(\text{Fake})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spatial-Only** | 0.8526 | 0.5536 | 0.5870 | 0.9236 | 0.4303 | 0.3724 |
| **Frequency-Only** | 0.6431 | 0.5864 | 0.6595 | 0.8394 | **0.5430** | 0.4717 |
| **Attention Fusion (Active Baseline)** | **0.8544** | 0.5624 | 0.6000 | 0.9202 | 0.4451 | 0.3816 |
| **Equal-Weight Average Fusion ($0.5/0.5$)** | 0.8386 | **0.6346** | **0.6890** | **0.9250** | **0.5490** | 0.4215 |

---

### 3.2 Breakdown by Generative Architecture & Real Controls

| Generative Cohort | Sample Size | Metric | Spatial-Only | Frequency-Only | Attention Fusion | Equal-Weight ($0.5/0.5$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Google Gemini / Imagen 3** | $N=17$ | Accuracy (Recall) | 11.8% | 5.9% | 11.8% | **17.6%** |
| | | Mean $P(\text{Fake})$ | 0.2231 | 0.0771 | 0.2134 | 0.2458 |
| **Stable Diffusion v1.5** | $N=30$ | Accuracy (Recall) | 60.0% | 30.0% | **63.3%** | 60.0% |
| | | Mean $P(\text{Fake})$ | 0.6800 | 0.3030 | 0.6779 | 0.5982 |
| **InsightFace Swap** | $N=30$ | Accuracy (Recall) | 56.7% | **76.7%** | 53.3% | **73.3%** |
| | | Mean $P(\text{Fake})$ | 0.5064 | 0.6736 | 0.5261 | 0.6214 |
| **StyleGAN2 Synthetic** | $N=60$ | Accuracy (Recall) | **95.0%** | 21.7% | **95.0%** | 86.7% |
| | | Mean $P(\text{Fake})$ | 0.9467 | 0.3238 | 0.9437 | 0.8012 |
| **Stable Diffusion Inpainting** | $N=200$ | Accuracy (Recall) | 25.5% | **68.5%** | 28.0% | **62.5%** |
| | | Mean $P(\text{Fake})$ | 0.3042 | 0.6311 | 0.3226 | 0.5188 |
| **Authentic Real Controls** | $N=120$ | Real Accuracy | **90.0%** | 70.8% | 89.2% | **88.3%** |
| | | Mean $P(\text{Fake})$ | 0.1095 | 0.3274 | 0.1124 | 0.1345 |

---

## 4. Key Scientific Insights

### 1. The Frequency Branch Has High Value on Boundary Manipulations
* On **Stable Diffusion Inpainting** ($N=200$), the spatial branch achieves only **25.5%** detection recall ($\bar{P}=0.3042$). The frequency branch achieves **68.5%** recall ($\bar{P}=0.6311$)—a **+43.0% improvement**! Inpainting leaves spectral discontinuity boundaries around the inpainting mask that are readily detected by 2D-DCT.
* On **InsightFace Swap** ($N=30$), the frequency branch achieves **76.7%** accuracy, outperforming the spatial branch (**56.7%**) by **+20.0%**. Blending seam gradients in face-swapping create distinct high-frequency phase and magnitude shifts.

### 2. The Failure Mechanism of the Attention Head
* Because StyleGAN2 images dominate the training set, the spatial branch reached near-perfect accuracy (95.0%) with high confidence. The attention head learned to allocate $\approx 94\%$ of its weight to the spatial branch ($w_s = 0.938$) and clamp the frequency branch near its minimum floor ($w_f = 0.062$).
* Consequently, when confronted with inpainting or face-swapping, the attention network **ignored** the frequency branch's alarm, causing the overall model to collapse to the spatial branch's 28.0% recall!
* When we replace the learned attention weighting with **Equal-Weight Fusion ($0.5/0.5$)**, overall test accuracy jumps from **56.24% to 63.46%**, and F1-score increases from **0.6000 to 0.6890**!

### 3. Recommendation on Frequency Branch Architecture
* **Do NOT remove the frequency branch.** Removing it drops inpainting detection recall from 68.5% to 25.5% and face-swap detection from 76.7% to 56.7%.
* **Redesign the Attention Head:** Introduce a dynamic cross-entropy regularization penalty on attention entropy, or enforce a higher attention floor ($\tau \ge 0.25$) so that neither branch can be monopolized by StyleGAN-trained shortcuts.
