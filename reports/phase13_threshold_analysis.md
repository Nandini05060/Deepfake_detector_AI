# Phase 13B Validation-Only Threshold and Calibration Study
## Empirical Operating Point Selection, Probability Calibration Analysis, and Specificity-Constrained Operating Policy

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase13_threshold_analysis.md`  
**Associated Benchmark Data:** `reports/phase13_calibration_analysis.json`, `reports/phase13_threshold_metrics.csv`  

---

### 1. Executive Summary & Experimental Methodology

To prevent data contamination, all threshold calibration and operating point explorations were executed **strictly and exclusively on the independent validation partition** ($N=242$ images: $120$ Authentic Real, $122$ Synthetic Fake). The locked test partition ($N=457$) was strictly quarantined.

**Core Findings:**
1. **Calibration Efficacy:** Temperature scaling ($T = 1.6344$) reduced Expected Calibration Error (ECE) on the validation set from **9.04% down to 7.69%** and reduced Brier score from **0.1256 to 0.1167**, confirming calibration improves posterior probability reliability on in-distribution architectures.
2. **Specificity Operating Window:** The predefined non-negotiable forensic constraint ($\ge 88.0\%$ Real Specificity) is satisfied for all thresholds $\theta \ge 0.49$.
3. **Recommended Validation Operating Point:** $\theta^* = 0.55$ achieves **92.50% Real Specificity** ($108/120$), **74.59% Fake Recall** ($117/122$), and **83.55% Balanced Accuracy**.
4. **Generalization Advisory:** Thresholds satisfying $\ge 88\%$ specificity on validation data do **not** automatically guarantee $88\%$ specificity on unseen distributions. Operating policy must incorporate uncertainty boundaries.

---

### 2. Validation Calibration Analysis: Raw vs. Temperature-Scaled ($T=1.6344$)

| Calibration Metric | Uncalibrated Raw Sigmoid | Temperature Scaled ($T=1.6344$) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Validation ROC-AUC** | 0.933 | 0.933 | 0.0000 (Monotonic Transform) |
| **Validation PR-AUC** | 0.9359 | 0.9359 | 0.0000 (Monotonic Transform) |
| **10-Bin Expected Calibration Error (ECE)** | **9.04%** | **7.69%** | **-1.35% (Improved)** |
| **Brier Score Loss** | **0.1256** | **0.1167** | **-0.0089 (Improved)** |
| **Negative Log-Likelihood (Log Loss)** | **0.423** | **0.3651** | **-0.0579 (Improved)** |

#### 10-Bin Reliability Diagram (Validation Partition)

| Bin | Confidence Range | Sample Count | Mean Confidence | Empirical Accuracy | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | [0.0, 0.1] | 95 | 0.0420 | 0.0737 | 0.0317 |
| 2 | [0.1, 0.2] | 13 | 0.1432 | 0.3846 | 0.2414 |
| 3 | [0.2, 0.3] | 8 | 0.2406 | 0.6250 | 0.3844 |
| 4 | [0.3, 0.4] | 7 | 0.3464 | 0.5714 | 0.2250 |
| 5 | [0.4, 0.5] | 12 | 0.4591 | 0.5833 | 0.1242 |
| 6 | [0.5, 0.6] | 11 | 0.5378 | 0.5455 | 0.0077 |
| 7 | [0.6, 0.7] | 15 | 0.6520 | 0.8000 | 0.1480 |
| 8 | [0.7, 0.8] | 13 | 0.7572 | 0.7692 | 0.0121 |
| 9 | [0.8, 0.9] | 21 | 0.8483 | 0.9048 | 0.0565 |
| 10 | [0.9, 1.0] | 47 | 0.9434 | 1.0000 | 0.0566 |

---

### 3. Threshold Sweep Analysis: Fake Recall vs. Real Specificity

A grid of 91 classification thresholds ($	heta \in [0.05, 0.95]$ in increments of $0.01$) was evaluated on validation data:

```
                      THRESHOLD VS. SPECIFICITY & RECALL (VALIDATION N=242)
100% ──── Real Specificity: ───────► (Increases with θ) ──────────────────────────
 90% ──────────── 88.0% Safety Floor ─────────────────────────────────────────────
 80% ─────────────────────────────────────────────────────────────────────────────
 70% ◄─────── Fake Recall: ───────── (Decreases with θ) ──────────────────────────
 60% ─────────────────────────────────────────────────────────────────────────────
 50% ─────────────────────────────────────────────────────────────────────────────
  0% ─────────────────────────────────────────────────────────────────────────────
    θ = 0.05        0.20        0.35      θ*=0.48      0.65        0.80      0.95
```

#### Key Operating Points Matrix

| Operating Point | Threshold $\theta$ | Accuracy | Balanced Accuracy | Real Specificity | Fake Recall | F1 Score | Confusion Matrix (TN / FP / FN / TP) | Meets $\ge 88\%$ Spec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\theta = 0.35$** | 0.35 | 84.30% | 84.30% | **84.17%** | 84.43% | 0.8443 | 101 / 19 / 19 / 103 | NO |
| **$\theta = 0.40$** | 0.40 | 83.88% | 83.89% | **85.00%** | 82.79% | 0.8382 | 102 / 18 / 21 / 101 | NO |
| **$\theta = 0.45$** | 0.45 | 83.06% | 83.08% | **85.83%** | 80.33% | 0.8270 | 103 / 17 / 24 / 98 | NO |
| **$\theta = 0.50$** | 0.50 | 83.06% | 83.11% | **89.17%** | 77.05% | 0.8210 | 107 / 13 / 28 / 94 | YES |
| **$\theta = 0.55$** | 0.55 | 83.47% | 83.55% | **92.50%** | 74.59% | 0.8198 | 111 / 9 / 31 / 91 | YES |
| **$\theta = 0.60$** | 0.60 | 82.64% | 82.73% | **93.33%** | 72.13% | 0.8073 | 112 / 8 / 34 / 88 | YES |
| **$\theta = 0.65$** | 0.65 | 80.99% | 81.11% | **95.00%** | 67.21% | 0.7809 | 114 / 6 / 40 / 82 | YES |

---

### 4. Forensic Operating Policy Recommendation

1. **Safety Floor:** Operating at any threshold $\theta < 0.49$ is **strictly prohibited**, as it breaches the mandatory $88.0\%$ Real Specificity requirement on genuine photographs.
2. **Recommended Operating Threshold:** $\theta^* = 0.55$ provides optimal balance between high forensic sensitivity (74.59%) and protection of authentic identities (92.50%).
3. **Uncertainty Margin Policy:** Because predictions near $\theta^*$ carry elevated epistemic uncertainty, Phase 13C defines the three-way decision boundary around $[0.40, 0.60]$ to safely route ambiguous cases to `INCONCLUSIVE`.
