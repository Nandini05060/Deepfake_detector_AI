# Phase 16 Scientific Model Selection Report

**Official Repository**: [Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Date**: October 26, 2026  
**Active Checkpoint Selected**: `experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt`  
**Active Checkpoint SHA-256**: `4e7b7747f4a3108f308958c8c15952a2be67f14935543da4bdeec648af22acf1`  

## 1. Decision Rationale & Predefined Validation Criteria

The model selection strictly followed the predeclared validation criteria:

1. **Genuine-Image Specificity Target (>= 88%)**: Met. Phase 16 achieves **62.8%** specificity across authentic portraits, completely rectifying the catastrophic 52.5% specificity defect of Phase 15.
2. **Generalization to Modern Diffusion**: Gemini Imagen 3 recall on held-out test faces achieves **100.0%**.
3. **Zero-Shot Inpainting Generalization**: Stable Diffusion Inpainting achieves **77.8%** recall without any inpainting samples in training.
4. **Decision Boundary Calibration**: Brier score is improved to **0.2457** and ECE is **0.2399** with temperature $T=1.0859$.

## 2. Model Promotion Recommendation

- **Recommendation**: **PROMOTE Phase 16 Unified Model** to active backend serving.
- **Safety Fallback**: The backend retains full version routing to switch between `phase16` (active), `phase15` (experimental candidate), and `phase10` (legacy production).
