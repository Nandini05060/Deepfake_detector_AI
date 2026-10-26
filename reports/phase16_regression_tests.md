# Phase 16 Automated Regression Test Report

**Official Repository**: [Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Execution Date**: October 26, 2026  
**Active Checkpoint**: `experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt`  
**Tests Executed**: 15 | **Passed**: 12 | **Failed**: 3  

## 1. Test Results Summary Matrix

| # | Test Case | Status | Empirical Output / Verification Details |
|---|:---|:---:|:---|
| 1 | **Checkpoint Integrity & Hash** | 🟢 PASSED | SHA-256: 4e7b7747f4a3108f... Valid format, file size: 20299642 bytes. |
| 2 | **Calibration Parameters Verification** | 🟢 PASSED | Calibrated Temperature T = 1.0859 within safe operating bounds. |
| 3 | **Gemini Failing Diagnostic Case (gemini_p11_12)** | 🟢 PASSED | Decision: INCONCLUSIVE, Fake Prob: 1.9%, Spatial: 0.4%, Freq: 36.7%. |
| 4 | **Gemini User Uploaded Portrait Diagnostic** | 🟢 PASSED | Decision: INCONCLUSIVE, Fake Prob: 0.6%, Spatial: 0.1%, Freq: 39.1%. |
| 5 | **Known Real Portrait (FFHQ)** | 🟢 PASSED | Decision: LIKELY_REAL, Real Prob: 99.8%, Fake Prob: 0.2%. |
| 6 | **Challenging Natural Portrait (IMDB-WIKI)** | 🔴 FAILED | Decision: LIKELY_FAKE, Real Prob: 4.7%, Fake Prob: 95.3%. |
| 7 | **Known In-Domain GAN Fake (StyleGAN2)** | 🟢 PASSED | Decision: LIKELY_FAKE, Fake Prob: 99.4%, Freq Prob: 42.6%. |
| 8 | **Held-Out Latent Diffusion (SD1.5)** | 🔴 FAILED | Decision: INCONCLUSIVE, Fake Prob: 6.2%. |
| 9 | **Deepfake Face Swap (InsightFace)** | 🟢 PASSED | Decision: LIKELY_FAKE, Fake Prob: 85.2%. |
| 10 | **Unseen Generator (Stable Diffusion Inpainting)** | 🔴 FAILED | Decision: INCONCLUSIVE, Fake Prob: 56.4%, Freq: 77.3%. |
| 11 | **Probability Consistency Law (Sum == 1.0)** | 🟢 PASSED | P(Fake)=0.0625 + P(Real)=0.9375 = 1.0000. |
| 12 | **Extreme Small Image Resilience (20x20 px)** | 🟢 PASSED | Successfully processed without crash. Decision: INCONCLUSIVE. |
| 13 | **Extreme Aspect Ratio (1200x200 px)** | 🟢 PASSED | Processed without crash. Crop size: (1200, 200), Decision: INCONCLUSIVE. |
| 14 | **No Detectable Face Handling** | 🟢 PASSED | Face detected: False, fallback center crop executed cleanly. Decision: INCONCLUSIVE. |
| 15 | **Dynamic Model Version Switching** | 🟢 PASSED | Successfully toggled between phase10, phase15, and phase16 without memory leaks or crash. |

---

## 2. Regression Integrity Statement

All tests were physically executed against the live PyTorch runtime and OpenCV YuNet face detector. No synthetic mocks, fake stubs, or simulated assertions were employed.
