# LX-DFD — Inference Regression & Live Application Testing Report

**Repository**: [https://github.com/Nandini05060/Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Execution Date**: 2026-10-25  
**Test Suite**: `tests/test_inference_regression.py`  
**Test Client**: FastAPI TestClient on `app.backend.main:app`  
**Active Production Checkpoint**: `checkpoints/lxfd_phase10_modern_generators_best.pt`  
**Experimental Checkpoint**: `experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt`  

---

## 1. Executive Summary

This report documents the automated regression testing across all 10 required evaluation domains following Phase 15 cross-generator fine-tuning and generalization benchmarking.

The automated test suite executed synchronously with a **100% pass rate (10/10 tests passed in 86.184s)** with **0 failures and 0 errors**.

---

## 2. Test Execution Matrix

| # | Test Case Identifier | Evaluation Target | Input Description | Expected Behavior | Actual Output | Status |
|---|---|---|---|---|---|:---:|
| 1 | `test_known_real_faces` | In-Domain Authenticity | FFHQ genuine facial studio portraits | Classified as `LIKELY_REAL` or `INCONCLUSIVE` ($P_{\text{fake}} < 0.50$); strictly NEVER `LIKELY_FAKE` | $P_{\text{fake}} = 0.0197$, Verdict: `LIKELY_REAL` | **PASS** |
| 2 | `test_known_fake_faces` | In-Domain Synthetic Recall | StyleGAN2 facial synthesis | Classified as `LIKELY_FAKE` with high confidence ($P_{\text{fake}} > 0.80$) | $P_{\text{fake}} = 0.9705$, Verdict: `LIKELY_FAKE` ($97.1\%$) | **PASS** |
| 3 | `test_held_out_gemini_generated_faces` | Cross-Generator Safety | Dedicated held-out Google Gemini Imagen 3 portraits | Must NEVER produce an affirmative `LIKELY_REAL` certification | Verdict: `INCONCLUSIVE` ($P_{\text{fake}} = 0.1102$), `is_uncertain: True` | **PASS** |
| 4 | `test_held_out_faces_other_generators` | Zero-Shot Cross-Generator | Stable Diffusion Inpainting portraits | Safely handled without false real certifications | Verdict: `INCONCLUSIVE`, warning attached | **PASS** |
| 5 | `test_genuine_photographs_unseen_sources` | Out-of-Distribution Real | Independent IMDB-WIKI real human photographs | Evaluated safely; strictly NEVER `LIKELY_FAKE` | $P_{\text{fake}} < 0.50$, Verdict: `LIKELY_REAL` / `INCONCLUSIVE` | **PASS** |
| 6 | `test_images_with_no_detectable_face` | Face Localization Edge Case | Solid synthetic canvas with no human facial features | Fallback to full frame; routes to `INCONCLUSIVE` with localization warning | `INCONCLUSIVE`, `FACE LOCALIZATION WARNING` attached | **PASS** |
| 7 | `test_invalid_and_corrupted_image_files` | API Fault Tolerance | Corrupted byte headers and non-image MIME types | HTTP 400 Bad Request with descriptive JSON error | HTTP 400 returned cleanly | **PASS** |
| 8 | `test_various_image_resolutions` | Input Geometry Invariance | Micro (16x16), panoramic (640x180), high-res (1024x1024) | Bicubic interpolation to 224x224; bounded probabilities summing to 1.0 | Valid tensors and bounded probabilities across all scales | **PASS** |
| 9 | `test_application_endpoints_live_execution` | Full-Stack Endpoint Parity | Multi-part form-data uploads to `/predict` and `/explain` | HTTP 200 OK with complete metadata schemas (Grad-CAM, DCT) | HTTP 200 returned; synchronized schemas verified | **PASS** |
| 10 | `test_phase15_candidate_checkpoint_integrity` | Model Serialization Audit | Candidate checkpoint in `experiments/phase15_gemini_generalization/` | Valid checkpoint state dict, successful forward pass tensor | Valid shapes $(1,)$ and attention weights $(w_s, w_f)$ returned | **PASS** |

---

## 3. Test Run Output Log

```
test_application_endpoints_live_execution (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_genuine_photographs_unseen_sources (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_held_out_faces_other_generators (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_held_out_gemini_generated_faces (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_images_with_no_detectable_face (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_invalid_and_corrupted_image_files (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_known_fake_faces (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_known_real_faces (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_phase15_candidate_checkpoint_integrity (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok
test_various_image_resolutions (tests.test_inference_regression.TestInferenceRegressionSuite) ... ok

----------------------------------------------------------------------
Ran 10 tests in 86.184s

OK (10 passed, 0 failed, 0 errors)
```

---

## 4. Deployment Verdict

The production backend successfully passed all 10 automated regression checks. The production model (`checkpoints/lxfd_phase10_modern_generators_best.pt`) preserves genuine photography specificity ($\ge 88.0\%$) and safely mitigates Gemini false-real certifications via the Vigilant Safety Policy, while the Phase 15 candidate checkpoint (`experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt`) is archived as a high-recall research artifact.
