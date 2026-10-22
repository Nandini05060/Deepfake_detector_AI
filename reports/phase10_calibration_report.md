# Phase 10 Calibration & Uncertainty Report

**Optimal Temperature Parameter ($T$)**: `1.6344`

## Calibration Comparison on In-Domain Test Data

| Calibration State | Expected Calibration Error (ECE) | Brier Score | Negative Log-Likelihood |
|---|---|---|---|
| **Raw Model Output** | 0.0576 | 0.0770 | 0.2573 |
| **Temperature Scaled ($T=1.634$)** | **0.0317** | **0.0712** | **0.2251** |

## Empirical Reliability Policy

- Predicted fake probability is scaled using temperature $T$, aligning probability with test accuracy on evaluated domains.
- On unseen or non-evaluated generators, confidence is restricted to `MODERATE` with an active Out-of-Distribution Scope Advisory.
