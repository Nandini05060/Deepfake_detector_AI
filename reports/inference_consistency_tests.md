# LX-DFD — Inference Consistency, Safety Boundary & Regression Report

**Repository**: [https://github.com/Nandini05060/Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Execution Environment**: Python 3.13, PyTorch 2.6.0+cu124, FastAPI TestClient, NVIDIA GeForce RTX 3050 Laptop GPU  
**Active Production Checkpoint**: `checkpoints/lxfd_phase10_modern_generators_best.pt`  
**Checkpoint SHA-256**: `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05`  
**Test Suite**: `tests/test_inference_consistency.py`  
**Associated Suite**: `tests/test_phase13_reliability.py`  

---

## 1. Executive Summary

This report documents the verification of the complete inference pipeline across 10 critical forensic safety domains following the investigation and resolution of the "fake-as-real" misclassification error reported in the Explainability Studio.

All 10 automated test suites executed synchronously with a 100% pass rate (`10/10 passed in 3.379s`). Zero regressions were introduced into existing reliability suites (`11/11 passed in 2.404s`).

---

## 2. Test Execution Matrix

| # | Test Case | Target Domain | Input Specification | Expected Behavior | Actual Result | Status |
|---|---|---|---|---|---|:---:|
| 1 | `test_known_real_images` | Genuine Specimen Control | 5 authenticated facial images from in-domain & IMDB-WIKI cohorts | Predicted as `LIKELY_REAL` or `INCONCLUSIVE` ($P_{\text{fake}} < 0.50$); strictly NEVER `LIKELY_FAKE` | $P_{\text{fake}} \in [0.0197, 0.3379]$, Verdict `LIKELY_REAL` / `INCONCLUSIVE` | **PASS** |
| 2 | `test_known_fake_images` | In-Domain Generator Recall | StyleGAN2 facial images from verified dataset | Predicted as `LIKELY_FAKE` with high confidence ($P_{\text{fake}} > 0.90$) | $P_{\text{fake}} = 0.9705$, Verdict `LIKELY_FAKE` ($P_{\text{fake}} = 97.1\%$) | **PASS** |
| 3 | `test_unfamiliar_generator_image_safe_handling` | Cross-Generator Safety | Google Gemini / Imagen 3 & SD Inpainting faces | Must NOT be decisively marked as `LIKELY_REAL`. Routes to `INCONCLUSIVE` with forensic caution | $P_{\text{fake}} = 0.1102$, Verdict `INCONCLUSIVE`, `is_uncertain: True` | **PASS** |
| 4 | `test_invalid_image_uploads` | API Fault Tolerance | Text files (`text/plain`), corrupted binary byte streams | HTTP 400 Bad Request with descriptive payload error | HTTP 400 returned for both `/predict` and `/explain` | **PASS** |
| 5 | `test_images_with_no_detectable_face` | Face Localization Boundary | Uniform / non-facial image inputs where YuNet detects no face | Graceful fallback to full-frame context with `INCONCLUSIVE` and `FACE LOCALIZATION WARNING` | Verdict `INCONCLUSIVE`, advisory warning attached | **PASS** |
| 6 | `test_very_small_or_unusually_sized_images` | Preprocessing Robustness | Extreme dimensions: (16x16), (24x380), (450x32), (64x64) | Bicubic interpolation to 224x224, valid tensor shapes, bounded output probabilities | All dimensions processed without shape exceptions, outputs bounded $[0, 1]$ | **PASS** |
| 7 | `test_probability_values_and_sum` | Probability Axiom Adherence | Synthetic RGB test stimuli | $P_{\text{fake}} \ge 0$, $P_{\text{real}} \ge 0$, and $|P_{\text{fake}} + P_{\text{real}} - 1.0| < 10^{-3}$ | $P_{\text{fake}} + P_{\text{real}} = 1.000$ strictly verified across fused and branch heads | **PASS** |
| 8 | `test_consistency_between_backend_endpoints` | API Synchronization | Concurrent image payload to `/predict` and `/explain` | Identical `prediction`, `final_decision`, `fake_probability`, and `real_probability` | Exact agreement between `/predict` and `/explain` responses | **PASS** |
| 9 | `test_missing_or_incompatible_checkpoint` | Configuration Integrity | Predictor initialized with non-existent checkpoint path | Explicit `FileNotFoundError` raised immediately (fail-fast architecture) | `FileNotFoundError` raised as expected | **PASS** |
| 10 | `test_uncertain_predictions_and_warnings` | Explanatory Advisory | Ambiguous / divergent boundary inputs | Decision flagged as `INCONCLUSIVE`, `is_uncertain: True`, non-empty generator advisory warning | Advisory warning attached explaining absence of evidence $\neq$ evidence of absence | **PASS** |

---

## 3. Key Pipeline Fixes Verified

### 3.1 Inference Discrepancy Gate Fix (`src/models/inference.py`)
- **Previous Failure**: The condition `elif high_discrepancy and abs(fake_prob - real_prob) < 0.70:` silently failed when $|0.11 - 0.89| = 0.78 \ge 0.70$, falling through to `prediction = "REAL"`.
- **Verified Fix**: Removed the `< 0.70` gate. Introduced strict discrepancy routing and spectral suspicion checks ($P_{\text{freq}} \ge 0.28, P_{\text{spatial}} < 0.10$). Samples exhibiting divergent modality signals are safely routed to `INCONCLUSIVE` rather than `LIKELY_REAL`.

### 3.2 Endpoint Metadata Alignment (`app/backend/main.py`)
- **Previous Discrepancy**: `/explain` was calling `predictor.predict(face_img)` without passing `face_meta=det_meta`, causing face detector confidence to be omitted, and did not include `final_decision` and `verdict` fields.
- **Verified Fix**: Updated `/explain` to call `predictor.predict(face_img, face_meta=det_meta)` and returned `final_decision`, `verdict`, `checkpoint_used`, and `branch_outputs`.

### 3.3 Explainability Studio UI Alignment (`app/frontend/app.js`)
- **Previous UI Bug**: When an inconclusive image returned $P_{\text{fake}} = 11.0\%$, the UI prominently displayed `Real Probability: 89.0%` in bright green alongside `Confidence Level: MODERATE`, presenting the appearance of an affirmative authentic verification despite the `UNCERTAIN / AMBIGUOUS` badge.
- **Verified Fix**:
  - Sets verdict badge to `INCONCLUSIVE / AMBIGUOUS`.
  - Sets gauge to amber (`#F59E0B`) with label `LOW CONF`.
  - Explicitly states `Classification: INCONCLUSIVE` and annotates `Real Probability: 89.0% (Non-Decisive)`.
  - Adds prominent Phase 6 caution:
    > *"Forensic Caution (Inconclusive / Out-of-Distribution): A low fake probability (11.0%) on an unfamiliar generator does NOT establish that an image is authentic. The model detected conflicting spatial and frequency cues or lacks a calibrated representation for this generator architecture."*
  - Displays explicit Model Version (`LX-DFD Phase 10 Modern Multi-Generator`) and Checkpoint Name (`lxfd_phase10_modern_generators_best.pt`).

---

## 4. Test Suite Log Artifacts

```
test_consistency_between_backend_endpoints (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_images_with_no_detectable_face (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_invalid_image_uploads (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_known_fake_images (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_known_real_images (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_missing_or_incompatible_checkpoint (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_probability_values_and_sum (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_uncertain_predictions_and_warnings (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_unfamiliar_generator_image_safe_handling (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok
test_very_small_or_unusually_sized_images (tests.test_inference_consistency.TestInferenceConsistencySuite) ... ok

----------------------------------------------------------------------
Ran 10 tests in 3.379s

OK
```
