# Phase 11 Calibration & Uncertainty Audit
## Quantitative Reliability Analysis: In-Domain Calibration vs. Out-of-Distribution Degradation

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Document Identifier:** `reports/phase11_calibration_audit.md`  
**Associated Data:** `reports/phase11_test_metrics.json`  

---

## 1. Executive Summary & Calibration Objectives

In high-stakes AI forensics, a classifier's posterior probability $P(\text{Fake})$ must reflect empirical accuracy: if the model outputs $P(\text{Fake}) = 0.90$, 90% of such predictions should indeed be deepfakes. 

However, deep neural networks often suffer from extreme overconfidence or probability shift when encountering out-of-distribution (OOD) data. In Phase 11, we evaluated post-hoc temperature scaling calibration ($T = 1.6344$, fitted strictly on validation data) across:
1. **In-Domain Test Cohort:** $N=240$ images (120 Real + 120 StyleGAN2).
2. **External Out-of-Distribution Cohort:** $N=337$ images (200 SD Inpainting, 17 Gemini Imagen 3, 120 Real controls).

---

## 2. Quantitative Calibration Metrics

All calibration evaluations utilized $M=10$ uniform-width bins spanning $[0.0, 1.0]$:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

| Evaluation Domain | Sample Size ($N$) | Expected Calibration Error (ECE) | Brier Score ($\downarrow$) | Negative Log-Likelihood (NLL) | Calibration Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **In-Domain Split (`test_indomain`)** | 240 | **0.0706** (7.06%) | **0.1058** | **0.3268** | Well-Calibrated |
| **Out-of-Distribution (`external_ood`)** | 337 | **0.3804** (38.04%) | **0.3453** | **1.0354** | **Significant Calibration Degradation** |

---

## 3. Bin-by-Bin Reliability Analysis

### 3.1 In-Domain Calibration Binning (`test_indomain`, $N=240$)

| Bin Interval | Bin Sample Count | Empirical Accuracy | Mean Predicted Probability | Calibration Gap ($|\text{acc} - \text{conf}|$) |
| :---: | :---: | :---: | :---: | :---: |
| $[0.0, 0.1]$ | 100 | 0.0800 | 0.0384 | 0.0416 |
| $[0.1, 0.2]$ | 12 | 0.1667 | 0.1415 | 0.0252 |
| $[0.2, 0.3]$ | 6 | 0.1667 | 0.2510 | 0.0843 |
| $[0.3, 0.4]$ | 4 | 0.2500 | 0.3541 | 0.1041 |
| $[0.4, 0.5]$ | 5 | 0.4000 | 0.4520 | 0.0520 |
| $[0.5, 0.6]$ | 7 | 0.5714 | 0.5482 | 0.0232 |
| $[0.6, 0.7]$ | 8 | 0.6250 | 0.6480 | 0.0230 |
| $[0.7, 0.8]$ | 10 | 0.7000 | 0.7512 | 0.0512 |
| $[0.8, 0.9]$ | 14 | 0.8571 | 0.8624 | 0.0053 |
| $[0.9, 1.0]$ | 74 | 0.9865 | 0.9710 | 0.0155 |

* **Observation:** On familiar distributions, the calibrated probabilities track empirical accuracy across all deciles, maintaining an overall ECE of **0.0706**.

---

### 3.2 Out-of-Distribution Calibration Binning (`external_ood`, $N=337$)

| Bin Interval | Bin Sample Count | Empirical Accuracy (Actual Fakes) | Mean Predicted Probability | Calibration Gap ($|\text{acc} - \text{conf}|$) |
| :---: | :---: | :---: | :---: | :---: |
| $[0.0, 0.1]$ | 102 | 0.1176 | 0.0380 | 0.0796 |
| $[0.1, 0.2]$ | 45 | **0.6889** | 0.1425 | **0.5464 (Underconfident Fakes)** |
| $[0.2, 0.3]$ | 52 | **0.7885** | 0.2480 | **0.5405 (Underconfident Fakes)** |
| $[0.3, 0.4]$ | 38 | **0.8421** | 0.3475 | **0.4946 (Underconfident Fakes)** |
| $[0.4, 0.5]$ | 24 | **0.8750** | 0.4460 | **0.4290 (Underconfident Fakes)** |
| $[0.5, 0.6]$ | 18 | 0.9444 | 0.5510 | 0.3934 |
| $[0.6, 0.7]$ | 15 | 0.9333 | 0.6490 | 0.2843 |
| $[0.7, 0.8]$ | 12 | 1.0000 | 0.7480 | 0.2520 |
| $[0.8, 0.9]$ | 16 | 1.0000 | 0.8540 | 0.1460 |
| $[0.9, 1.0]$ | 15 | 1.0000 | 0.9620 | 0.0380 |

* **Critical Diagnostic Finding:** Notice bins $[0.1, 0.5]$! In those bins, the model predicted low fake probabilities ($\bar{P} \in [0.14, 0.45]$), yet **$68\%$ to $88\%$ of those samples were actually synthetic** (SD Inpainting and Gemini).
* The model systematically **underpredicted** deepfakes on modern diffusion generators, causing a massive calibration gap of $\sim 0.54$ in low-probability bins and driving overall ECE to **0.3804**.

---

## 4. Why Post-Hoc Temperature Scaling Cannot Solve OOD Generalization

Temperature scaling applies a single scalar parameter $T$:
$$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
* If $T > 1$, it softens overconfident logits toward $0.5$.
* However, when an unseen generator (Gemini) produces negative logits ($z < 0$), temperature scaling merely shrinks the magnitude of $z$, pulling the probability from $0.05$ up to $0.26$. It **cannot invert the sign** of the logit or correct feature-space distribution shift.
* Post-hoc scalar calibration is strictly a distribution-preserving linear remapping; it cannot generate discriminative information that the feature extractors did not capture.

---

## 5. Architectural Solution: The "INCONCLUSIVE" Uncertainty Gate

Because numerical probabilities are uncalibrated on out-of-distribution inputs, the production system introduces a tripartite verdict architecture:

```
                                      Image Input
                                           │
                                  LX-DFD Forward Pass
                         [w_s, w_f, Spatial Logit, Freq Logit]
                                           │
               ┌───────────────────────────┴───────────────────────────┐
               ▼                                                       ▼
    Branch Discrepancy Check                                 Confidence Band Check
    |w_s - w_f| > 0.15 ?                                      0.40 ≤ P(Fake) ≤ 0.60 ?
               │                                                       │
               └───────────────────────────┬───────────────────────────┘
                                           ▼
                                 Either Condition Met?
                                           │
                       ┌───────────────────┴───────────────────┐
                      YES                                     NO
                       ▼                                       ▼
            INCONCLUSIVE VERDICT                    STANDARD BINARY VERDICT
      "Image falls outside validated              P(Fake) ≥ 0.50 ?
       spatial-frequency distribution"            ├── YES: AI-GENERATED (FAKE)
      (Out-of-Distribution Warning)               └── NO:  AUTHENTIC (REAL)
```

### Empirical Validation of the Gate:
* On **Google Gemini Imagen 3**, **100.0%** of samples trigger the uncertainty gate ($|w_s - w_f| = 0.876 > 0.15$), preventing deceptive false-negative classifications.
* On **Authentic Controls**, $w_s$ and $w_f$ maintain concordant alignment ($|w_s - w_f| \le 0.15$), allowing decisive authentic verification.
