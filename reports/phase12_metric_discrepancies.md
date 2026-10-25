# Phase 12 Metric Discrepancies & Historical Audit Analysis
## Systematic Reconciliation Between Phase 10 Claims, Phase 11 Audit, and Phase 12 Independent Reproduction

**Document Version:** 1.0.0  
**Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase12_metric_discrepancies.md`  

---

## 1. Executive Summary of Discrepancies

During Phase 12 independent reproduction, three primary discrepancies and historical reporting nuances were identified and analyzed:

1. **The Gemini Recall Divergence (75.0% vs. 11.76%):**
   - **Historical Claim in Phase 10:** Held-out Gemini benchmark reported 75.0% recall (3/4 detected) with mean $P(\text{Fake}) = 0.7211$.
   - **Reported in Phase 11 Audit:** Gemini/Imagen 3 cohort reported 11.76% recall (2/17 detected) with mean $P(\text{Fake}) = 0.2134$.
   - **Root Cause Verified in Phase 12:**
     - **Sample size difference:** Phase 10 evaluated only $N=4$ images (`data/splits_phase10/test_gemini_heldout.csv`). Phase 11 expanded this to $N=17$ images by adding 13 locked images from `data/gemini_dataset/phase11/`.
     - **Preprocessing / Crop Geometry Mismatch:** In Phase 10, cropping was done via `Phase10Dataset` / `main.py` where margin 0.50 adds $0.50 \cdot \text{box\_dim}$ on both sides (effective crop width = $2.0 \times \text{bbox}$). In Phase 11, evaluation used `YuNetFacePreprocessor(margin=0.50)` which defines `side = int(round(max(fw, fh) * (1.0 + margin)))` (effective crop width = $1.5 \times \text{bbox}$).
     - **Empirical impact measured on the same 4 images:**
       - Sample 0 (`gemini_original_diagnostic.png`): Under $2.0\times$ crop (`main.py`), $P(\text{Fake}) = 0.9489$ (**FAKE**). Under $1.5\times$ crop (`YuNetFacePreprocessor`), $P(\text{Fake}) = 0.0341$ (**REAL**).
       - Sample 2 (`gemini_portrait_09`): Under $2.0\times$ crop, $P(\text{Fake}) = 0.9203$ (**FAKE**). Under $1.5\times$ crop, $P(\text{Fake}) = 0.4086$ (**INCONCLUSIVE**).
       - Under the wider crop ($2.0\times$), 3 out of 4 images exhibited $P(\text{Fake}) > 0.50$. Under the tighter crop ($1.5\times$), 3 out of 4 images dropped below the decision threshold.
     - **Conclusion:** The apparent "75% recall" in Phase 10 was an artifact of hair/background context inclusion on an extremely small sample ($N=4$). When evaluated with standardized facial cropping on the expanded $N=17$ set, true baseline recall is **11.76%** (2/17).

2. **The Frequency Branch Inversion Discrepancy:**
   - **StyleGAN2 vs. Modern Diffusion:**
     - On StyleGAN2 ($N=60$), spatial branch achieves 95.0% recall and frequency branch achieves 21.67%.
     - On SD Inpainting ($N=200$), spatial branch achieves only 25.50% recall, while frequency branch achieves **68.50%** recall (+43.0% boost).
     - On Google Gemini ($N=17$), spatial branch achieves 11.76% recall, while frequency branch achieves only **5.88%** (1/17) with mean $P(\text{Fake}) = 0.0771$.
   - **Root Cause:** Inpainting introduces sharp spatial seams and mask boundary phase discontinuities that produce strong 2D-DCT spectral anomalies. Latent diffusion decoders (Imagen 3) synthesize naturalistic $1/f$ spectral roll-offs without grid artifacts, causing the frequency branch to register near-zero anomaly scores unless resolution downsampling or spectral residuals are applied.

3. **The Downsampling Diagnostic Hazard (Real Control Sample Scarcity):**
   - **Phase 11 Finding:** In `scripts/run_phase11_gemini_diagnostic.py`, downsampling to $112 \times 112$ (Condition F) increased Gemini recall from 11.8% to 88.2% and reported 100% real control accuracy.
   - **Audit Finding in Phase 12:** Condition F was evaluated against only **$N=5$ real images** (`real_controls = sorted(glob.glob('data/external/IMDB_WIKI_real/*.jpg'))[:5]`). On those 5 real images, mean frequency fake probability surged from 0.4777 to **0.8910**!
   - **Implication:** Downsampling severely affects high-frequency spectral statistics. Adopting $112 \times 112$ naively without evaluating the entire 120-image real control set risks a catastrophic increase in false-positive rates on authentic photographs. A controlled resolution ablation across all 120 real controls is mandatory before any architectural change.

---

## 2. Table of Discrepancies and Reconciliations

| Metric / Phenomenon | Phase 10 Claim | Phase 11 Audit | Phase 12 Verified Truth | Discrepancy Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **Active Checkpoint SHA-256** | `13dc56...` | `13dc56...` | `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05` | Verified exact cryptographic match. |
| **Calibration Temperature** | $T = 1.6344$ | $T = 1.6344$ | $T = 1.634375$ ($1.6344$) | Verified exact parameter match. |
| **Gemini Held-Out Recall** | 75.0% ($3/4$) | 11.76% ($2/17$) | **11.76%** on standardized crop ($N=17$); $25.0\%$ on tighter crop ($N=4$); $75.0\%$ on wider crop ($N=4$). | Preprocessing crop boundary width + small-sample variance ($N=4$). |
| **In-Domain Accuracy** | 88.75% | 88.75% | **88.75%** ($213/240$) | Exact reproduction confirmed. |
| **In-Domain ROC-AUC** | 0.9709 | 0.9709 | **0.9709** | Exact reproduction confirmed. |
| **SD Inpainting Recall** | 42.0% ($84/200$) | 28.0% ($56/200$) | **28.0%** ($56/200$) under attention fusion; **68.5%** under frequency branch. | Phase 10 used single-branch or uncalibrated threshold; attention model clamps frequency branch to 0.05 floor, yielding 28.0%. |
| **InsightFace Recall** | 73.33% ($22/30$) | 53.33% ($16/30$) | **53.33%** under attention fusion; **76.67%** under frequency branch. | Attention fusion suppresses frequency branch ($w_f \approx 0.06$). |
| **Real Control Specificity** | 88.33% ($106/120$) | 89.17% ($107/120$) | **89.17%** ($107/120$) | Verified to four decimal places. |
| **Downsampled Gemini Recall** | N/A | 88.24% ($15/17$) | Under Condition F ($112\times 112$ downsample + upsample), recall reaches 88.24%, but real control false alarms surge on frequency branch. | Diagnostic experiment was evaluated on only $N=5$ real controls; systematic evaluation on $N=120$ real controls is required. |

---

## 3. Methodological Directives for Phase 12

1. **Standardize Face Preprocessing:** Reconcile `YuNetFacePreprocessor` and `app/backend/main.py` so that training, evaluation, and production share the exact same crop geometry.
2. **Never Generalize from $N=4$ or $N=17$ Unchecked:** All reported metrics must explicitly state $(k / N)$ and Wilson score 95% confidence intervals.
3. **Audit False Positives Rigorously:** Any resolution or frequency enhancement that improves Gemini recall must maintain authentic photograph specificity $\ge 88\%$.
