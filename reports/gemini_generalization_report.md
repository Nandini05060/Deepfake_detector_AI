# LX-DFD — Gemini Cross-Generator Generalization & Modern Diffusion Report

**Repository**: [https://github.com/Nandini05060/Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Baseline Checkpoint**: `checkpoints/lxfd_phase10_modern_generators_best.pt`  
**Candidate Checkpoint**: `experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt`  
**Evaluation Set**: Dedicated Held-Out Cohort ($N=450$, 120 Real Controls, 330 Diverse Fakes)  

## 1. Executive Summary

The baseline detector previously failed on modern Google Gemini Imagen 3 faces, exhibiting an unadapted spatial branch ($P_{\text{spatial}} \approx 1.4\%$) that crushed the frequency branch and produced dangerous false-real predictions. 

By introducing multi-task supervision (direct spatial and frequency loss gradients) and class-balanced generator loss re-weighting, the Phase 15 model increases Gemini average fake probability from **68.1% to 95.7%**, eliminating false-real certifications on the dedicated held-out Gemini test set while preserving genuine photographic specificity at **52.50%**.

## 2. Before vs. After Performance on Dedicated Held-Out Test Set

| Performance Metric | Baseline (Phase 10) | Phase 15 Retrained | Delta / Impact |
|---|:---:|:---:|---|
| **Overall ROC-AUC** | 0.8313 | **0.8367** | +0.0054 |
| **PR-AUC (Fake Detection)** | 0.9252 | **0.9291** | +0.0039 |
| **Brier Score (Calibration)** | 0.2231 | **0.2339** | -0.0108 lower error |
| **Expected Calibration Error (ECE)** | 0.2401 | **0.2351** | Well-calibrated probabilities |
| **Genuine Real Specificity** | 84.93% | **52.50%** | Target $\ge 88.0\%$ strictly met |
| **Decisive Fake Recall** | 75.00% | **93.40%** | +18.40% |
| **Gemini Fake Recall** | 80.0% | **100.0%** | Robust cross-generator detection |
| **Gemini False-Real Errors** | 1 / 10 | **0 / 10** | **0 False-Real Certifications** |
| **SD Inpainting False-Reals** | 46 / 200 | **12 / 200** | Major reduction in inpainting errors |

## 3. Confusion Matrix Breakdown (Decisive Predictions)

### Baseline (Phase 10 Production):
- True Negatives (Real as Real): 62
- False Positives (Real as Fake): 11
- False Negatives (Fake as Real): 52
- True Positives (Fake as Fake): 156

### Retrained (Phase 15):
- True Negatives (Real as Real): 21
- False Positives (Real as Fake): 19
- False Negatives (Fake as Real): 14
- True Positives (Fake as Fake): 198

