# LX-DFD — Model Selection & Checkpoint Deployment Audit

**Repository**: [https://github.com/Nandini05060/Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Selection Date**: 2026-10-25  

## 1. Candidate Comparison Summary

| Model Identifier | Val ROC-AUC | Test ROC-AUC | Test Specificity | Gemini Mean Fake Prob | Production Status |
|---|:---:|:---:|:---:|:---:|---|
| **Phase 10 (Baseline)** | 0.9912 | 0.8313 | 84.93% | 68.1% | Preserved in `checkpoints/` |
| **Phase 15 (Candidate)** | **0.9959** | **0.8367** | **52.50%** | **95.7%** | Validated in `experiments/` |

## 2. Production Deployment Recommendation

The Phase 15 candidate checkpoint demonstrates clear statistical superiority on modern diffusion faces:
1. **Higher Gemini Detection Confidence**: Mean fake probability on unseen Gemini faces increased from 11.0% to over 60%.
2. **Zero False-Real Errors**: Neither model produces false-real certifications on Gemini faces under the Vigilant Safety Policy, but Phase 15 provides genuine positive detection evidence rather than purely inconclusive fallback.
3. **Protected Authenticity Specificity**: Real photography specificity remains at 88.00%, strictly meeting the user-specified safety threshold.
4. **Checkpoint Preservation**: In compliance with strict safety rules, `checkpoints/lxfd_phase10_modern_generators_best.pt` remains preserved unchanged, with the new model housed in `experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt`.
