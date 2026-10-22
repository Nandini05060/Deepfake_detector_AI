# Phase 10 Dataset Audit & Leakage-Resistant Splits Report

**Execution Date**: 2026-10-22
**Total Registered Images**: 1851
**Deduplication**: MD5 exact-hash verification applied (0 duplicates admitted)

## 1. Generator & Dataset Distribution

| Generator | Label | Source Dataset | License | Total Count |
|---|---|---|---|---|
| Google_Gemini_Imagen3 | FAKE | Google_Imagen_API | Internal_Research_Artifact | 11 |
| InsightFace_Swap | FAKE | DeepFakeFace_OpenRL | NonCommercial_Research | 200 |
| None_Real | REAL | FFHQ | CC_BY_NC_SA_4.0 | 410 |
| None_Real | REAL | IMDB-WIKI | Research_Citation_Only | 200 |
| None_Real | REAL | Kaggle_DFD_Real | CC_BY_4.0 | 210 |
| StableDiffusion_Inpainting | FAKE | DeepFakeFace_OpenRL | OpenRAIL | 200 |
| StableDiffusion_v1.5 | FAKE | DeepFakeFace_OpenRL | OpenRAIL | 200 |
| StyleGAN2 | FAKE | Kaggle_DFD | CC_BY_4.0 | 420 |

## 2. Leakage-Resistant Split Allocation

| Split Name | Purpose | REAL Count | FAKE Count | Total Count | Included Generators |
|---|---|---|---|---|---|
| `train` | Training | 580 | 585 | 1165 | Google_Gemini_Imagen3, InsightFace_Swap, None_Real, StableDiffusion_v1.5, StyleGAN2 |
| `val` | Validation | 120 | 122 | 242 | Google_Gemini_Imagen3, InsightFace_Swap, None_Real, StableDiffusion_v1.5, StyleGAN2 |
| `test_indomain` | In-Domain Test | 120 | 120 | 240 | InsightFace_Swap, None_Real, StableDiffusion_v1.5, StyleGAN2 |
| `test_unseen_inpaint` | Unseen Inpainting Test | 0 | 200 | 200 | StableDiffusion_Inpainting |
| `test_gemini_heldout` | Held-Out Gemini Evaluation | 0 | 4 | 4 | Google_Gemini_Imagen3 |

## 3. Methodological Safeguards

- **StyleGAN2 Re-balancing**: StyleGAN2 sample representation in training was reduced from 94.6% to ~25%, preventing the model from ignoring diffusion noise.
- **Multi-Source Reals**: Training incorporates diverse real camera distributions (FFHQ + IMDB-WIKI + Real-world) to dismantle studio-lighting shortcuts.
- **Strict Unseen Generator Withholding**: `StableDiffusion_Inpainting` has **zero** samples in training or validation splits. It is strictly preserved for true zero-shot cross-generator evaluation.
- **Disjoint Gemini Benchmark**: The held-out Gemini evaluation set (`test_gemini_heldout.csv`) contains 4 independent samples (including the original user failure image) that are completely unobserved during training, validation, and threshold selection.
