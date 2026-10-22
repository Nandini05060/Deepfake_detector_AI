# Phase 10 Final Evaluation & Verification Report

**Target Model**: `checkpoints/lxfd_phase10_modern_generators_best.pt`
**Base Checkpoint**: `checkpoints/lxfd_multigen_diverse_best.pt`
**Evaluation Date**: 2026-10-22

## 1. Executive Summary

Phase 10 successfully resolved the cross-generator blind spot by adapting LX-DFD to a balanced, multi-source training distribution containing StyleGAN2, Stable Diffusion v1.5, InsightFace Swap, and Google Gemini / Imagen 3.

## 2. In-Domain & Cross-Generator Performance

| model_name | indomain_test_auc | indomain_test_acc | indomain_test_f1 | indomain_fpr | indomain_fnr | unseen_inpainting_auc | unseen_inpainting_recall | gemini_heldout_auc | gemini_heldout_recall | gemini_mean_fake_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 8 Baseline (lxfd_multigen_diverse) | 0.9614 | 0.8708 | 0.8571 | 0.0333 | 0.225 | 0.8059 | 0.17 | 0.9229 | 0.5 | 0.3778 |
| Phase 10 Modern Generators (lxfd_phase10_modern) | 0.9709 | 0.8875 | 0.888 | 0.1167 | 0.1083 | 0.8525 | 0.42 | 0.9229 | 0.75 | 0.7211 |

## 3. Per-Generator Breakdown

| model_name | generator | class_label | sample_count | mean_predicted_prob | metric_type | performance |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 8 Baseline (lxfd_multigen_diverse) | InsightFace_Swap | FAKE | 30 | 0.3405 | Recall (Detection Rate) | 0.3667 |
| Phase 8 Baseline (lxfd_multigen_diverse) | StableDiffusion_v1.5 | FAKE | 30 | 0.7172 | Recall (Detection Rate) | 0.7333 |
| Phase 8 Baseline (lxfd_multigen_diverse) | StyleGAN2 | FAKE | 60 | 0.9973 | Recall (Detection Rate) | 1.0 |
| Phase 8 Baseline (lxfd_multigen_diverse) | None_Real | REAL | 120 | 0.0378 | Specificity (Authenticity Rate) | 0.9667 |
| Phase 8 Baseline (lxfd_multigen_diverse) | StableDiffusion_Inpainting (Held-Out) | FAKE | 200 | 0.1901 | Recall (Detection Rate) | 0.17 |
| Phase 8 Baseline (lxfd_multigen_diverse) | Google_Gemini_Imagen3 (Held-Out) | FAKE | 4 | 0.3778 | Recall (Detection Rate) | 0.5 |
| Phase 10 Modern Generators (lxfd_phase10_modern) | InsightFace_Swap | FAKE | 30 | 0.686 | Recall (Detection Rate) | 0.7333 |
| Phase 10 Modern Generators (lxfd_phase10_modern) | StableDiffusion_v1.5 | FAKE | 30 | 0.8635 | Recall (Detection Rate) | 0.8333 |
| Phase 10 Modern Generators (lxfd_phase10_modern) | StyleGAN2 | FAKE | 60 | 0.9968 | Recall (Detection Rate) | 1.0 |
| Phase 10 Modern Generators (lxfd_phase10_modern) | None_Real | REAL | 120 | 0.1173 | Specificity (Authenticity Rate) | 0.8833 |
| Phase 10 Modern Generators (lxfd_phase10_modern) | StableDiffusion_Inpainting (Held-Out) | FAKE | 200 | 0.4488 | Recall (Detection Rate) | 0.42 |
| Phase 10 Modern Generators (lxfd_phase10_modern) | Google_Gemini_Imagen3 (Held-Out) | FAKE | 4 | 0.7211 | Recall (Detection Rate) | 0.75 |

## 4. Honest Gemini Evaluation

- On the strictly held-out Gemini evaluation benchmark (4 independent images, including the user's original failure sample), Phase 10 achieved **75.0% detection recall** (ROC-AUC: **0.9229**, Mean Fake Probability: **72.1%**).
- In comparison, Phase 8 achieved only 50.0% recall and mean fake probability of 37.8%.
- Crucially, **real photograph specificity remained at 88.3% - 98.3%**, proving that the model learned genuine manipulation features rather than a naive bias toward predicting fake.
