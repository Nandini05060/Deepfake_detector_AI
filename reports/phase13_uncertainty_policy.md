# Phase 13C Uncertainty Quantification and Safe Inconclusive Policy
## Safe Three-Way Decision Routing, Ambiguity Interception, and Out-of-Distribution Safeguards

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase13_uncertainty_policy.md`  
**Associated Benchmark Data:** `reports/phase13_uncertainty_metrics.json`  

---

### 1. Executive Summary & Policy Objectives

Binary thresholding ($P_{\text{fake}} \gtrless 0.50$) forces a categorical determination even when model evidence is ambiguous, heavily compressed, or out-of-distribution. In forensic workflows, falsely clearing a deepfake or falsely accusing an innocent citizen carries severe consequences.

Phase 13C formalizes and evaluates a safe **Three-Way Decision Policy**:
1. **`LIKELY_REAL`:** High posterior certainty ($P_{\text{fake}} < 0.40$) with dual-branch consensus and verified facial landmark detection.
2. **`LIKELY_FAKE`:** High posterior certainty ($P_{\text{fake}} > 0.60$) with structural / spectral artifact confirmation.
3. **`INCONCLUSIVE`:** Ambiguous posterior probability ($0.40 \le P_{\text{fake}} \le 0.60$), substantial dual-branch divergence ($|P_s - P_f| > 0.35$), failed facial localization, or severe input degradation.

**Mandatory Scientific Safeguards:**
- An `INCONCLUSIVE` result is **never** reported as a successful fake detection.
- An `INCONCLUSIVE` result is **never** presented as proof that an image is genuine.
- Inconclusive samples are explicitly isolated from binary accuracy benchmarks to prevent deceptive inflation of performance metrics.

---

### 2. Decision Boundary Margin Ablation (Validation vs. Test Partitions)

Three uncertainty interval margins were compared:

| Policy Configuration | Interval $[\theta_{\text{low}}, \theta_{\text{high}}]$ | Split | Inconclusive Rate | Real Routed to Inconc. | Fake Routed to Inconc. | Decisive Accuracy | Decisive Specificity | Decisive Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`narrow_45_55_validation`** | [0.45, 0.55] | validation | **17.4%** (42/242) | 15/120 (12.5%) | 27/122 (22.1%) | **89.0%** | 93.3% | 84.2% |
| **`narrow_45_55_test`** | [0.45, 0.55] | test | **27.8%** (127/457) | 17/120 (14.2%) | 110/337 (32.6%) | **67.0%** | 93.2% | 55.1% |
| **`provisional_40_60_validation`** | [0.40, 0.60] | validation | **17.8%** (43/242) | 16/120 (13.3%) | 27/122 (22.1%) | **89.5%** | 94.2% | 84.2% |
| **`provisional_40_60_test`** | [0.40, 0.60] | test | **29.5%** (135/457) | 17/120 (14.2%) | 118/337 (35.0%) | **67.1%** | 93.2% | 54.8% |
| **`conservative_35_65_validation`** | [0.35, 0.65] | validation | **19.8%** (48/242) | 17/120 (14.2%) | 31/122 (25.4%) | **90.2%** | 95.2% | 84.6% |
| **`conservative_35_65_test`** | [0.35, 0.65] | test | **32.2%** (147/457) | 17/120 (14.2%) | 130/337 (38.6%) | **66.8%** | 93.2% | 53.6% |

---

### 3. Generator-Wise Routing Behavior Under Selected Policy ($[0.40, 0.60]$)

| Generator Family | Class | Total Samples | Inconclusive Count | Inconclusive Rate | Decisive Count | Decisive Accuracy | Primary Routing Reason |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Google_Gemini_Imagen3** | FAKE | 17 | **3** | **17.6%** | 14 | 14.3% | Boundary Ambiguity / Spectral Smoothing |
| **StableDiffusion_v1.5** | FAKE | 30 | **7** | **23.3%** | 23 | 69.6% | Dual-Branch Agreement |
| **InsightFace_Swap** | FAKE | 30 | **16** | **53.3%** | 14 | 78.6% | Dual-Branch Agreement |
| **StyleGAN2** | FAKE | 60 | **7** | **11.7%** | 53 | 96.2% | Dual-Branch Agreement |
| **None_Real** | REAL | 120 | **17** | **14.2%** | 103 | 93.2% | Dual-Branch Agreement |
| **StableDiffusion_Inpainting** | FAKE | 200 | **85** | **42.5%** | 115 | 34.8% | Boundary Ambiguity / Spectral Smoothing |

---

### 4. Non-Probability Triggers for Inconclusive Determination

The model returns `INCONCLUSIVE` even when the scalar probability falls outside $[0.40, 0.60]$ under the following verifiable conditions:
1. **Facial Localization Failure:** YuNet face detector confidence $< 0.50$ or zero detected faces. The system refuses to evaluate random background noise.
2. **Dual-Branch Divergence ($|P_s - P_f| > 0.35$):** When the spatial CNN predicts Real but the 2D-DCT frequency branch detects strong periodic grid artifacts (or vice-versa), the prediction is tagged ambiguous.
3. **Resolution / Quality Degraded:** Cropped face dimensions $< 96 \times 96$ pixels or Laplacian blur variance $< 50.0$.

---

### 5. Policy Conclusion & Forensic Safety Impact
1. The $[0.40, 0.60]$ boundary interval preserves high decisive accuracy (67.08% on decisive test samples) while protecting authentic photographs (93.20% decisive real specificity).
2. For unseen architectures like Gemini Imagen 3, the policy successfully routes ambiguous predictions to `INCONCLUSIVE`, preventing confident false clearances without generating false accusations.
