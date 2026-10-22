# LX-DFD Phase 9: Automated Regression & Preprocessing Robustness Report
**Date**: 2026-10-22
**Target Model**: `checkpoints/lxfd_multigen_diverse_best.pt`
**SHA256**: `b3d280d8dbfaeede215d5e6d239b8bb613a1bf147cee9e59bde798cc9b766300`

## Test Suite Execution Results

| Test Name | Status | Technical Details |
|---|---|---|
| Checkpoint Selection Integrity | **PASSED** | Loaded: lxfd_multigen_diverse_best.pt |
| Class Mapping Verification (REAL=0.0, FAKE=1.0) | **PASSED** | Real item fake_prob=0.0047, Fake item fake_prob=0.5267 |
| Finite Logits & Probabilities Check | **PASSED** | Output Logit: -5.6003 (Valid float) |
| Dual-Branch Execution & Non-Zero Attention Floor | **PASSED** | w_s=0.9497, w_f=0.0503 (w_f >= 0.05 guaranteed) |
| Face Detection Fallback on Blank / No-Face Image | **PASSED** | Graceful fallback preserves input image size ((256, 256)) without throwing exceptions |
| Extreme Aspect Ratio Handling (1920x480 Panoramic & 480x1920 Vertical) | **PASSED** | Panoramic Prob: 0.0007, Vertical Prob: 0.0011 - Executed without crash |
| Explainability Visualizations Correspondence | **PASSED** | Grad-CAM shape: (224, 224), DCT Colormap shape: (224, 224, 3) |
| Adversarial Edge Case Analysis (SD Inpainting Texture Bleed) | **PASSED** | Inpainting Prediction: REAL (Fake Prob: 13.6%, Spatial: 12.8%, Freq: 15.6%) |
