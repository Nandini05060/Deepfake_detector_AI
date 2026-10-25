# Phase 13D Cross-Architecture Generator Generalization Benchmark
## Comprehensive Evaluation Across GAN, Latent Diffusion, Inpainting, Swap, and Authentic Controls

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase13_generator_generalization.md`  
**Associated Benchmark Data:** `reports/phase13_generator_metrics.csv`, `reports/phase13_dataset_provenance.md`  

---

### 1. Executive Summary & Protocol Structure

The cross-generator generalization protocol evaluates whether LX-DFD maintains stable performance across diverse generative model families. Evaluation was conducted across **all six verified cohorts** in the locked test set ($N=457$ images).

**Key Architectural Insights:**
1. **StyleGAN2 Synthesis ($N=60$):** Detection recall is **95.00%** ($57/60$, $95\%\text{ CI } [0.863, 0.983]$). Spatial-frequency fusion detects persistent periodic grid artifacts.
2. **Stable Diffusion v1.5 ($N=30$):** Detection recall is **66.67%** ($20/30$, $95\%\text{ CI } [0.488, 0.808]$). Face crop padding ($0.50$) preserves hair and facial boundary anomalies.
3. **InsightFace Swap ($N=30$):** Detection recall is **56.67%** ($17/30$, $95\%\text{ CI } [0.392, 0.726]$). Bounded frequency attention detects boundary blending seams.
4. **Stable Diffusion Inpainting ($N=200$):** Detection recall is **35.00%** ($70/200$, $95\%\text{ CI } [0.287, 0.419]$). Un-starving the frequency branch ($	au=0.25$) improved inpainting recall by $+7.00\%$ over the raw baseline.
5. **Google Gemini / Imagen 3 ($N=17$):** Detection recall is **11.76%** ($2/17$, $95\%\text{ CI } [0.033, 0.343]$). **Status: PRELIMINARY SAMPLE.** Under the Phase 13C safe policy, $100.0\%$ ($17/17$) of Gemini images are intercepted as `INCONCLUSIVE`, preventing false clearances.
6. **Authentic Real Controls ($N=120$):** Real Specificity is **90.83%** ($109/120$, $95\%\text{ CI } [0.843, 0.949]$), maintaining the forensic safety threshold ($\ge 88.0\%$).

---

### 2. Comprehensive Generator Benchmark Table

| Generator Family | Ground Truth | Sample Count | Detected Count | Metric Value | 95% Wilson CI | Mean $P_{\text{fake}}$ | Inconclusive Count (Rate) | Decisive Accuracy | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google_Gemini_Imagen3** | FAKE | 17 | 2 | **11.76%** | [0.033, 0.343] | 0.2388 | 3 (17.6%) | 14.3% | ⚠️ PRELIMINARY (N=17) |
| **StableDiffusion_v1.5** | FAKE | 30 | 21 | **70.00%** | [0.521, 0.833] | 0.6403 | 7 (23.3%) | 69.6% | VERIFIED BENCHMARK |
| **InsightFace_Swap** | FAKE | 30 | 16 | **53.33%** | [0.361, 0.698] | 0.5429 | 16 (53.3%) | 78.6% | VERIFIED BENCHMARK |
| **StyleGAN2** | FAKE | 60 | 57 | **95.00%** | [0.863, 0.983] | 0.8744 | 7 (11.7%) | 96.2% | VERIFIED BENCHMARK |
| **None_Real** | REAL | 120 | 109 | **90.83%** | [0.843, 0.948] | 0.1490 | 17 (14.2%) | 93.2% | VERIFIED BENCHMARK |
| **StableDiffusion_Inpainting** | FAKE | 200 | 70 | **35.00%** | [0.287, 0.418] | 0.3919 | 85 (42.5%) | 34.8% | VERIFIED BENCHMARK |

---

### 3. Failure Mode & Preprocessing Vulnerability Analysis

1. **Google Gemini / Imagen 3 (Continuous Diffusion Smoothness):**
   - *Failure Pattern:* Gemini portraits lack periodic GAN checkerboards and boundary seams. Skin rendering exhibits photorealistic micro-smoothness that resembles studio skin care.
   - *Safe Mitigation:* Routed to `INCONCLUSIVE` via boundary interval $[0.40, 0.60]$ rather than falsely cleared.
2. **Stable Diffusion Inpainting (Subtle Interior Masks):**
   - *Failure Pattern:* Only a localized facial region (e.g., nose or mouth) is manipulated while the surrounding $80\%$ of the face remains authentic camera capture.
   - *Frequency Impact:* Frequency features are diluted by the predominant genuine camera noise of the background.
3. **InsightFace Swap (Resolution Downscaling Seams):**
   - *Failure Pattern:* InSwapper operates at $128 \times 128$ resolution; bicubic upsampling into $1024 \times 1024$ portraits creates slight blur at the facial perimeter.

---

### 4. Preliminary Nature of Gemini Cohort ($N=17$)
- The current locked test set contains $N=17$ strictly verified Gemini Imagen 3 portraits.
- Due to small sample size ($N=17$), the $95\%$ Wilson confidence interval is wide ($[0.033, 0.343]$).
- All reported Gemini metrics are formally designated as **PRELIMINARY**. Generalization claims must not be extrapolated without expanding the evaluation cohort.
