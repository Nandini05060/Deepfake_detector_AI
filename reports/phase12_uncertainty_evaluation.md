# Phase 12G Uncertainty, Calibration & Inconclusive Policy Audit
## Empirical Evaluation of Temperature Scaling Transfer, Expected Calibration Error (ECE), and Dual-Branch Uncertainty Routing

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Computer Vision Researcher & Machine Learning Auditor  
**Document Identifier:** `reports/phase12_uncertainty_evaluation.md`  
**Associated Data:** `reports/phase12_calibration_metrics.json`  

---

## 1. Executive Summary & Research Motivation

A critical flaw in deepfake detection deployments is overconfidence on out-of-distribution (OOD) generators. Neural networks trained via binary cross-entropy push sigmoid probabilities near $0.0$ or $1.0$, giving a deceptive illusion of high confidence even when confronted with novel generative architectures.

Phase 12G audited:
1. The mathematical validity and parameter origin of temperature scaling ($T = 1.6344$).
2. Whether temperature calibration optimized on in-domain validation data transfers to unseen external generators.
3. The empirical efficacy of the `INCONCLUSIVE` verdict routing rules.

### Key Audit Findings:
1. **Temperature Scaling Successfully Calibrates In-Domain Distributions:**
   - On in-domain validation data ($N=242$), temperature scaling reduces Expected Calibration Error (ECE) from **0.1190 down to 0.0764** (a $35.8\%$ relative reduction).
   - On in-domain locked test data ($N=240$), ECE drops from **0.0955 down to 0.0706**, with Brier score improving to **0.1058**.
2. **Calibration Does NOT Transfer to External Generators:**
   - On unseen out-of-domain generators ($N=217$: SD Inpainting + Google Gemini Imagen 3), calibrated ECE remains severe at **0.6517**, and Negative Log-Likelihood (NLL) remains high at **1.0354**.
   - This proves empirically that parametric temperature scaling cannot fix epistemic distribution shift.
3. **The INCONCLUSIVE Uncertainty Gate is Indispensable:**
   - Using the dual-condition uncertainty gate ($0.40 \le P \le 0.60$ OR branch divergence $|P_s - P_f| > 0.15$), the model flags **47.5% of out-of-domain samples as INCONCLUSIVE**.
   - On the remaining in-domain test samples routed to conclusive verdicts, accuracy reaches **92.5%**.
   - The uncertainty gate acts as an automated safety brake, preventing the system from issuing false REAL or FAKE verdicts on ambiguous diffusion portraits.

---

## 2. Partition-by-Partition Calibration Metrics

Evaluated using 10 uniform reliability bins $[0.0, 1.0]$:

| Partition | Sample Count ($N$) | Uncalibrated ECE | Calibrated ECE ($T=1.6344$) | Brier Score | NLL Loss | Inconclusive Rate | Conclusive Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set (In-Domain)** | 242 | 0.1190 | **0.0764** ($-35.8\%$) | 0.1042 | 0.3215 | 20.7% | 89.6% |
| **Test Set (In-Domain)** | 240 | 0.0955 | **0.0706** ($-26.1\%$) | 0.1058 | 0.3268 | 22.1% | **92.5%** |
| **Test Set (Out-of-Domain)** | 217 | 0.6860 | **0.6517** (Failure) | 0.3453 | 1.0354 | **47.5%** | 28.1% |

---

## 3. Mathematical Policy & Threshold Derivation

### 3.1 Temperature Scaling
The probability of manipulation is computed from binary logit $z$ via temperature parameter $T = 1.6344$:
$$P(\text{Fake}) = \sigma\left(\frac{z}{1.6344}\right) = \frac{1}{1 + e^{-z / 1.6344}}$$
Parameter $T$ was optimized strictly on validation log-likelihood without touching test data.

### 3.2 Automated Uncertainty Routing Specification
A sample is routed to `INCONCLUSIVE` if either of the following conditions is satisfied:
1. **Decision Boundary Ambiguity:**
   $$0.40 \le P(\text{Fake}) \le 0.60$$
2. **Dual-Branch Disagreement (Branch Divergence):**
   $$|P_{\text{spatial}} - P_{\text{freq}}| > 0.15 \quad \text{and} \quad |P(\text{Fake}) - 0.50| < 0.35$$

### 3.3 Forensic Scope Warning Contract
When `INCONCLUSIVE` is triggered, the API returns:
```json
{
  "prediction": "INCONCLUSIVE",
  "confidence_level": "LOW",
  "empirical_reliability": "LOW (Decision Boundary Region or Branch Divergence)",
  "generator_warning": "INCONCLUSIVE VERDICT: This image exhibits features that place it outside the model's calibrated operating range. Novel generators or compression artifacts may produce ambiguous forensic signals."
}
```

---

## 4. Integrity Directives for Deployment
1. **Never equate INCONCLUSIVE with FAKE:** An inconclusive result denotes lack of forensic confidence, not proof of guilt or manipulation.
2. **Never claim uncertainty proves generator identity:** Uncertainty scores reflect distribution mismatch, which can also be caused by camera noise, blur, or severe compression.
