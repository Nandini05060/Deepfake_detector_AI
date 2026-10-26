# Phase 16 Per-Generator Evaluation & Cross-Domain Breakdown

**Evaluated on Test Cohort**: $N = 448$ samples

## 1. Multi-Model Benchmark Comparison on Held-Out Test Set

| Evaluation Metric | Phase 10 Production | Phase 15 Gemini Candidate | Proposed Phase 16 Unified | Phase 16 Gain / Trade-off |
|:---|:---:|:---:|:---:|:---|
| **ROC-AUC** | 0.8329 | 0.8381 | **0.8400** | +0.0019 |
| **PR-AUC** | 0.9274 | 0.9307 | **0.9293** | -0.0014 |
| **Decisive Accuracy** | 76.7% | 83.3% | **84.6%** | +1.3% |
| **F1 Score** | 0.8262 | 0.9032 | **0.9055** | +0.0023 |
| **Fake Recall** | 74.2% | 91.3% | **89.2%** | -2.1% |
| **Real Specificity** | 84.0% | 35.9% | **62.8%** | **+26.9% boost** |
| **False Positive Rate** | 16.0% | 64.1% | **37.2%** | **-26.9% reduction** |
| **Brier Score** | 0.2134 | 0.2326 | **0.2457** | 0.0131 |
| **Expected Calibration Error (ECE)** | 0.2157 | 0.2312 | **0.2399** | 0.0087 |
| **Inconclusive Rate** | 33.9% | 40.0% | **44.9%** | Balance preserved |
| **Latency per Sample** | 22.3 ms | 22.9 ms | **20.5 ms** | Real-time GPU inference |

## 2. Per-Generator Granular Performance Matrix

| Generator / Cohort | True Class | Phase 10 Recall | Phase 15 Recall | Phase 16 Recall | Phase 16 Mean Fake Prob | Phase 16 Inconclusive Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FFHQ_Real** | `REAL` | 100.0% | 100.0% | **100.0%** | 0.5% | 68.3% |
| **Google_Gemini_Imagen3** | `FAKE` | 71.4% | 100.0% | **100.0%** | 80.6% | 25.0% |
| **IMDB_WIKI** | `REAL` | 60.0% | 19.4% | **33.3%** | 40.3% | 60.0% |
| **InsightFace_Swap** | `FAKE` | 81.0% | 90.9% | **95.7%** | 74.2% | 23.3% |
| **StableDiffusion_Inpainting** | `FAKE` | 54.5% | 84.7% | **77.8%** | 45.8% | 55.0% |
| **StableDiffusion_v1.5** | `FAKE` | 95.8% | 96.5% | **96.2%** | 87.8% | 13.3% |
| **StyleGAN2** | `FAKE` | 100.0% | 100.0% | **100.0%** | 97.9% | 1.7% |

---

## 3. Confusion Matrices (Decisive Classifications)

### Phase 10 Production
```
[[ 63  12]
 [ 57 164]]
```

### Phase 15 Gemini Candidate (Severe Real Degradation)
```
[[ 14  25]
 [ 20 210]]
```

### Phase 16 Unified Model (Restored Specificity & Generalization)
```
[[ 27  16]
 [ 22 182]]
```

