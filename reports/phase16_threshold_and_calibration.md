# Phase 16 Calibration, Operating Thresholds & Safety Policy

**Target Operating Point**: Specificity $\ge 88.0\%$
**Fitted Temperature**: $T = 1.0859$ (Validation Fit, L-BFGS, Bounded)
**Operating Decision Threshold**: $\tau = 0.57$
**Uncertainty Margin**: $\pm 0.075$ ($[0.495, 0.645]$)

## 1. Validation Set Threshold Sweep Results

| Threshold ($\tau$) | Real Specificity | Fake Recall | F1 Score | Inconclusive Rate | Safety Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.45** | **87.9%** | 100.0% | 0.9828 | 40.7% | `VIOLATES SAFETY (<88%)` |
| **0.47** | **90.6%** | 100.0% | 0.9870 | 41.1% | `TARGET MET` |
| **0.49** | **90.9%** | 100.0% | 0.9869 | 41.1% | `TARGET MET` |
| **0.51** | **90.9%** | 99.1% | 0.9825 | 41.1% | `TARGET MET` |
| **0.53** | **90.9%** | 99.1% | 0.9823 | 41.5% | `TARGET MET` |
| **0.55** | **90.9%** | 99.1% | 0.9821 | 41.9% | `TARGET MET` |
| **0.57** | **93.8%** | 99.1% | 0.9865 | 42.3% | `TARGET MET` |
| **0.59** | **93.8%** | 99.1% | 0.9864 | 42.7% | `TARGET MET` |
| **0.61** | **93.8%** | 97.3% | 0.9774 | 42.3% | `TARGET MET` |
| **0.63** | **93.8%** | 97.3% | 0.9772 | 42.7% | `TARGET MET` |
| **0.65** | **93.8%** | 96.4% | 0.9727 | 42.3% | `TARGET MET` |

---

## 2. Decision Logic & 3-Way Policy

```python
if abs(p_calibrated - 0.57) <= 0.075:
    verdict = 'INCONCLUSIVE'  # Epistemic ambiguity
elif spatial_prob < tau and freq_prob < tau:
    if freq_prob >= 0.28 and spatial_prob < 0.10:
        verdict = 'INCONCLUSIVE'  # Modality divergence safeguard
    else:
        verdict = 'LIKELY_REAL'
elif spatial_prob >= tau and freq_prob >= tau:
    verdict = 'LIKELY_FAKE'
elif p_calibrated >= 0.80 and (spatial_prob >= 0.85 or freq_prob >= 0.85):
    verdict = 'LIKELY_FAKE'
else:
    verdict = 'INCONCLUSIVE'
```
