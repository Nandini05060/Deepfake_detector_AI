# Calibration and Uncertainty Audit: Why Confidence is Not Reliability
## Empirical Assessment of Probability Calibration, Brier Score, and Out-of-Distribution Overconfidence

**Author:** AI Forensics Scientific Evaluator & Risk Lead  
**Document Identifier:** `REP-CALIB-2026-V1`  
**Optimal Production Temperature:** $T = 1.6344$ (`checkpoints/calibration_params_phase10.json`)  
**Data Sources:** `reports/phase9_calibration_metrics.json` and `reports/phase10_calibration_report.md`  

---

## 1. Problem Statement: The Fallacy of Raw Softmax Confidence

A critical weakness of modern deep neural networks is **overconfident misclassification**. When an uncalibrated deep network is evaluated on an out-of-distribution image, its final linear layer frequently produces large logits ($|z| > 6.0$). Passing these through a sigmoid activation yields probabilities of $99.9\%$ or $0.01\%$.

In a forensic or judicial setting, an examiner seeing **"Fake Probability: 99.9%"** will assume that the platform is 99.9% accurate on this image. However:
$$\text{Softmax Output Probability } \hat{p} \neq \text{Empirical Reliability } P(Y = \hat{y} \mid \hat{p})$$

A model can output 99% probability on a sample while being right only 40% of the time. This audit evaluates calibration metrics to align model confidence with real-world accuracy.

---

## 2. Formal Calibration Metrics

Given a test dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ where $y_i \in \{0, 1\}$ and model predicted probabilities $\hat{p}_i = \sigma(z_i / T)$:

1. **Brier Score (Mean Squared Error in Probability Space):**
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2 \in [0, 1]$$
   Lower is better; penalizes overconfident wrong predictions quadratically.
2. **Negative Log-Likelihood (NLL / Cross-Entropy Loss):**
   $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
3. **Expected Calibration Error (ECE):**
   Group predictions into $M = 10$ equally spaced confidence bins $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   Measures the average gap between predicted probability and actual empirical accuracy.

---

## 3. In-Domain vs. External Generator Calibration Comparison

*All calibration parameters ($T$) were derived strictly on validation data, leaving final test sets untouched.*

### Table 1: In-Domain vs. External Calibration Metrics

| Evaluation Domain | Metric | Raw Uncalibrated Output | Calibrated ($T = 1.6344$) | Calibration Shift |
| :--- | :--- | :---: | :---: | :---: |
| **In-Domain Test (Phase 10)** | Expected Calibration Error (ECE) | 0.0576 (5.76%) | **0.0317 (3.17%)** | **-45.0% error reduction** |
| | Brier Score | 0.0770 | **0.0712** | -7.5% improvement |
| | Negative Log-Likelihood (NLL) | 0.2573 | **0.2251** | -12.5% loss reduction |
| **In-Domain Test (Phase 9)** | Expected Calibration Error (ECE) | 0.0227 (2.27%) | **0.0228 (2.28%)** | Near-optimal calibration |
| | Brier Score | 0.0168 | 0.0169 | Minimal drift |
| **External Out-of-Distribution** | Expected Calibration Error (ECE) | 0.5980 (59.8%) | **0.5963 (59.6%)** | **Severe Overconfidence!** |
| *(SD 1.5, Face Swaps, Gemini)* | Brier Score | 0.5805 | 0.5774 | High dispersion |
| | Negative Log-Likelihood (NLL) | 2.8207 | 2.7158 | Extreme loss penalties |

---

## 4. The External Calibration Breakdown: A Stark Warning

While Temperature Scaling successfully calibrated the in-domain test set (reducing ECE to **0.0317**), Table 1 reveals a startling forensic finding:

> **On external, unseen generators, the Expected Calibration Error remained at 59.63%!**

### Why Temperature Scaling Alone Cannot Fix Out-of-Distribution Overconfidence
Examining the reliability diagram bins for external generators (`reports/phase9_calibration_metrics.json`, lines 132–140):
* In **Bin 0** ($0.0 \le \hat{p} \le 0.1$, predictions where the model believed the image was authentic real with $>90\%$ certainty):
  * **Sample Count:** 217 samples.
  * **Model Average Confidence:** $1.61\%$ Fake probability (98.39% Real).
  * **Empirical Accuracy:** **82.49% were actually FAKE!**

Because latent diffusion models (SD 1.5, Gemini Imagen 3) lack high-frequency checkerboard harmonics, the network produced large negative logits ($z \ll -5.0$). Scaling these logits by $T = 1.6344$ softened the probability slightly (e.g., from 0.001 to 0.016), but **could not flip the sign of the logit**.

---

## 5. The LX-DFD Uncertainty Architecture & Operational Policy

To prevent users from mistaking model overconfidence for factual ground truth, LX-DFD enacted four mandatory safeguards across the API and frontend:

### 1. Empirical Reliability Levels
Raw percentages are translated into bounded forensic categories:
* **HIGH EMPIRICAL RELIABILITY:** Calibrated probability $\ge 0.85$ on validated in-domain generators (StyleGAN2, StyleGAN3, FFHQ).
* **MODERATE RELIABILITY (UNCERTAIN):** Probability between $0.40$ and $0.60$, or any prediction on an unvalidated generator family.
* **OUT-OF-DISTRIBUTION SCOPE WARNING:** When an image exhibits high spatial-frequency branch divergence ($|w_s - w_f| > 0.15$), the system triggers an amber uncertainty banner:
  > *"Branch divergence detected. Model attention indicates non-conforming frequency characteristics. Softmax confidence does not guarantee real-world accuracy."*

### 2. Separation of Branch Probabilities and Fusion Weights
The platform explicitly reports:
* Spatial Branch Probability: $P_s(\text{Fake})$
* Frequency Branch Probability: $P_f(\text{Fake})$
* Spatial Attention Weight: $w_s$
* Frequency Attention Weight: $w_f$
* Discrepancy Metric: $\delta_{\text{branch}} = |w_s - w_f|$

### 3. Temperature Scaling Deployed
All API inference responses now include the temperature-scaled probability alongside the raw logit:
```json
{
  "prediction": "FAKE",
  "fake_probability": 0.8635,
  "real_probability": 0.1365,
  "raw_fake_probability": 0.9412,
  "temperature": 1.6344,
  "calibration_status": "Temperature Scaled (T=1.634; In-Domain Validated)"
}
```

---

## 6. Scientific Conclusion

1. **Never equate 99% probability with 99% truth.**
2. Temperature scaling ($T=1.6344$) effectively calibrates in-domain predictions, reducing ECE to 3.17%.
3. Out-of-distribution generators require qualitative scope disclaimers and multi-crop diagnostic stability checks rather than blind trust in a single numerical probability.

---

*Certified by LX-DFD Scientific Calibration & Risk Taskforce.*
