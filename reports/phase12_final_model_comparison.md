# Phase 12I Comprehensive Model Comparison & Selection Audit
## Fair Multi-Model Evaluation on Locked Test Partitions (N=457), Precision-Recall Trade-offs, and Candidate Selection

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase12_final_model_comparison.md`  
**Associated Data:** `reports/phase12_final_metrics.json`  

---

## 1. Executive Summary & Selection Protocol

In accordance with strict forensic research integrity rules, four model architectures were evaluated on identical locked test samples ($N=457$ images across 5 generator families and authentic negative controls):

1. **Model 1: Active Baseline (`LXDFD-Phase10-Attention`)**  
   Checkpoint: `checkpoints/lxfd_phase10_modern_generators_best.pt`  
   SHA-256: `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05`  
   Architecture: EfficientNet-B0 + 2D DCT ConvNet + Dynamic Attention ($\tau = 0.05$) + Temperature Scaler ($T=1.6344$).
2. **Model 2: Constrained Floor Attention (`LXDFD-Constrained-Floor-25`)**  
   Same verified checkpoint with un-starved attention floor ($\tau = 0.25$).
3. **Model 3: Equal-Weight Fusion (`LXDFD-Equal-Weight`)**  
   Same verified checkpoint with $0.5 F_s + 0.5 F_f$.
4. **Model 4: Candidate Architecture (`LXDFD-MultiScale-12`)**  
   Checkpoint: `experiments/phase12_multiscale_v1/lxfd_multiscale12_best.pt`  
   SHA-256: `a2b9989205f255248ca7c378e25164b88fbc47691ce8b9b9b815fcd00a72e773`  
   Architecture: Dual-scale spatial pathways ($224 \times 224$ detail + $112 \times 112$ global) + 2D DCT ConvNet + Constrained Multimodal Fusion ($\tau = 0.20$).

---

## 2. Quantitative Evaluation Matrix (Locked Test Set N=457)

| Metric | Active Baseline | Constrained Attention ($\tau=0.25$) | Equal-Weight ($0.5/0.5$) | Candidate LXDFD-MultiScale-12 |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.8544 | **0.8556** (Best) | 0.8386 | 0.8255 |
| **PR-AUC** | 0.9329 | **0.9333** (Best) | 0.9228 | 0.9212 |
| **Overall Accuracy** | 56.24% | 60.18% | 63.46% | **63.68%** (Best) |
| **F1-Score** | 0.6000 | 0.6459 | 0.6890 | **0.7004** (Best) |
| **Precision** | 0.9202 | 0.9375 | 0.9250 | 0.8940 |
| **Overall Fake Recall** | 44.51% | 49.26% | 54.90% | **57.57%** (Best) |
| **Real Specificity** | 89.17% | **90.83%** (Best) | 87.50% | 80.83% |
| **Real False Positive Rate (FPR)** | 10.83% | **9.17%** (Lowest) | 12.50% | 19.17% |
| **Real Specificity 95% CI** | [82.3%, 93.6%] | [84.3%, 94.8%] | [80.4%, 92.3%] | [72.9%, 86.9%] |
| **Brier Score** | 0.2241 | **0.2185** (Best) | 0.2280 | 0.2312 |
| **Expected Calibration Error (ECE)** | 0.2180 | **0.2015** (Best) | 0.2340 | 0.2450 |
| **GPU Inference Latency** | **11.4 ms** | **11.4 ms** | **11.4 ms** | 14.8 ms |
| **Model Parameter Count** | **4.8M** | **4.8M** | **4.8M** | 5.0M |

---

## 3. Per-Generator Breakdown

| Generator Family | Class | Sample Count ($N$) | Active Baseline | Constrained Attention ($\tau=0.25$) | Equal-Weight ($0.5/0.5$) | Candidate LXDFD-MultiScale-12 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Gemini Imagen 3** | FAKE | $N=17$ | 11.76% (2/17) | 11.76% (2/17) | 17.65% (3/17) | **47.06%** (**8/17**) |
| | | Mean $P(\text{Fake})$ | 0.2134 | 0.1884 | 0.1628 | **0.5409** |
| | | 95% Wilson CI | [3.3%, 34.3%] | [3.3%, 34.3%] | [6.2%, 41.0%] | [26.2%, 69.0%] |
| **Stable Diffusion Inpainting** | FAKE | $N=200$ | 28.00% (56/200) | 35.00% (70/200) | **44.00%** (88/200) | 41.50% (83/200) |
| **InsightFace Swap** | FAKE | $N=30$ | 53.33% (16/30) | 53.33% (16/30) | 66.67% (20/30) | **80.00%** (**24/30**) |
| **Stable Diffusion v1.5** | FAKE | $N=30$ | 63.33% (19/30) | 70.00% (21/30) | 66.67% (20/30) | **76.67%** (**23/30**) |
| **StyleGAN2 Synthetic** | FAKE | $N=60$ | **95.00%** (57/60) | **95.00%** (57/60) | 90.00% (54/60) | 93.33% (56/60) |
| **Authentic Real Controls** | REAL | $N=120$ | 89.17% (107/120) | **90.83%** (**109/120**) | 87.50% (105/120) | 80.83% (97/120) |

---

## 4. Confusion Matrices

### Active Baseline (`LXDFD-Phase10-Attention`):
$$\begin{pmatrix} \text{TN} = 107 & \text{FP} = 13 \\ \text{FN} = 187 & \text{TP} = 150 \end{pmatrix}$$

### Constrained Floor Attention ($\tau=0.25$):
$$\begin{pmatrix} \text{TN} = 109 & \text{FP} = 11 \\ \text{FN} = 171 & \text{TP} = 166 \end{pmatrix}$$

### Candidate LXDFD-MultiScale-12:
$$\begin{pmatrix} \text{TN} = 97 & \text{FP} = 23 \\ \text{FN} = 143 & \text{TP} = 194 \end{pmatrix}$$

---

## 5. Candidate Selection Adjudication (Part 11 & Part 12 Directives)

### Predefined Forensic Criterion:
In high-stakes deepfake forensic intelligence, **falsely accusing authentic human photographs must be strictly controlled**. A model that inflates detection recall by driving up false-positive rates on innocent camera photos is unviable for production certification.
- Minimum Acceptable Real Specificity: $\ge 88.0\%$ ($\text{FPR} \le 12.0\%$).
- Primary Optimization Metric: ROC-AUC and Brier Score.

### Adjudication Decision:
1. **Candidate LXDFD-MultiScale-12:**
   - Successfully quadruples Gemini detection recall ($11.76\% \to 47.06\%$, $8/17$) and boosts Inpainting ($41.5\%$) and InsightFace ($80.0\%$).
   - However, its Real Specificity is **$80.83\%$** (FPR = **$19.17\%$**), which breaches the $88\%$ safety threshold. Nearly 1 in 5 real portraits are falsely flagged as fakes.
   - Therefore, in strict compliance with Part 12 instructions ("If the candidate does not improve the predefined evaluation criteria, do not replace the production model. Explain the result honestly and retain the existing model"), `LXDFD-MultiScale-12` is retained as a specialized experimental candidate in `experiments/phase12_multiscale_v1/`, but is **NOT** promoted to replace the primary production checkpoint.
2. **Recommended Production Enhancement:**
   - **Constrained Floor Attention ($\tau = 0.25$)** on the verified checkpoint (`checkpoints/lxfd_phase10_modern_generators_best.pt`) achieves the optimal operating point:
     - Increases ROC-AUC to **0.8556** (highest across all models).
     - Reduces Real False Positive Rate from $10.83\%$ down to **$9.17\%$** (Specificity = **$90.83\%$**).
     - Reduces Expected Calibration Error from $0.2180$ to **$0.2015$**.
     - Boosts Inpainting detection recall from $28.0\%$ to **$35.0\%$**.
     - Boosts Stable Diffusion v1.5 recall from $63.3\%$ to **$70.0\%$**.
     - Fully backward-compatible with verified checkpoint SHA-256 (`13dc56b8...`).
