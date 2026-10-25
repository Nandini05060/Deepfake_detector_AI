# Phase 13D Dataset Provenance & Split Integrity Audit
## Cryptographic Manifest Verification, Source Attribution, and Zero-Contamination Guarantee

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase13_dataset_provenance.md`  
**Locked Manifest:** `data/manifests/phase11_generator_manifest.csv`  

---

### 1. Dataset Integrity & Cryptographic Hashes

- **Locked Manifest Path:** `data/manifests/phase11_generator_manifest.csv`
- **Manifest SHA-256:** `b21d9d4c2bbd4860630ee2765b3a20f3846f3ebaaf1c3f2aabac24566bc08304`
- **Total Registered Images:** 1,864 samples
- **Split Partitions:**
  - `train`: 1,165 images (580 Real, 585 Fake)
  - `val`: 242 images (120 Real, 122 Fake)
  - `test_indomain`: 240 images (120 Real, 120 Fake)
  - `test_unseen_inpaint`: 200 images (200 Fake SD Inpainting)
  - `test_locked`: 13 images (13 Fake Gemini Imagen 3)
  - `test_gemini_heldout`: 4 images (4 Fake Gemini Imagen 3)
- **Total Test Images:** $240 + 200 + 13 + 4 = 457$ images ($120$ Real, $337$ Fake).

---

### 2. Generator Attribution & Source Provenance

| Generator Family | Total Images | Splits Present | Underlying Synthesis Engine | Provenance Notes |
| :--- | :---: | :--- | :--- | :--- |
| **Authentic Real Faces (`None_Real`)** | 820 | `train`, `val`, `test_indomain` | Genuine Optical Camera Sensors | Derived from FFHQ & CelebA-HQ photography. Checked for duplicate removal. |
| **StyleGAN2 Synthesis** | 420 | `train`, `val`, `test_indomain` | Official StyleGAN2-ADA FFHQ | Fully synthetic 1024x1024 faces generated from random seeds. |
| **Stable Diffusion v1.5** | 200 | `train`, `val`, `test_indomain` | CompVis/stable-diffusion-v1-5 | Prompt-engineered human portraits generated at 512x512. |
| **InsightFace Swap** | 200 | `train`, `val`, `test_indomain` | InSwapper-128 ONNX Pipeline | Target faces swapped onto source portraits; color-corrected. |
| **SD Inpainting (Held-Out)** | 200 | `test_unseen_inpaint` | RunwayML SD-Inpainting v1.5 | Unseen in training; localized polygon inpainting on facial features. |
| **Google Gemini Imagen 3** | 24 | `train` (5), `val` (2), `test` (17) | Google Imagen 3 API | State-of-the-art continuous diffusion portraits. $N=17$ in locked test. |

---

### 3. Leakage & Partition Disjointness Verification
1. **Cryptographic Deduplication:** All 1,864 images possess distinct SHA-256 hashes. 0 identical duplicates cross partition boundaries.
2. **Subject Disjointness:** Authentic real images originate from disjoint subject pools across train, val, and test splits.
3. **Partition Isolation:** The 17 test Gemini images are completely quarantined from training ($N=5$) and validation ($N=2$) sets.
