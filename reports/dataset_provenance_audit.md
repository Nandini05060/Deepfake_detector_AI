# LX-DFD — Dataset Provenance, Split & Licensing Audit

**Repository**: [https://github.com/Nandini05060/Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Total Verified Images**: 1878  
**Manifest File**: [`data/verified_faces/verified_manifest.csv`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/data/verified_faces/verified_manifest.csv)  

## 1. Dataset Split Summary

| Split | Real Count | Fake Count | Total | Generators Included |
|---|:---:|:---:|:---:|---|
| **Train** | 590 | 590 | **1180** | FFHQ_Real (490), StyleGAN2 (300), InsightFace_Swap (140), StableDiffusion_v1.5 (140), IMDB_WIKI (100), Google_Gemini_Imagen3 (10) |
| **Validation** | 124 | 124 | **248** | FFHQ_Real (84), StyleGAN2 (60), IMDB_WIKI (40), StableDiffusion_v1.5 (30), InsightFace_Swap (30), Google_Gemini_Imagen3 (4) |
| **Test** | 120 | 330 | **450** | StableDiffusion_Inpainting (200), StyleGAN2 (60), IMDB_WIKI (60), FFHQ_Real (60), StableDiffusion_v1.5 (30), InsightFace_Swap (30), Google_Gemini_Imagen3 (10) |

---

## 2. Generator Provenance & Licensing Matrix

| Generator / Cohort | Class | Source / Reference | License | Role in Benchmark |
|---|---|---|---|---|
| **FFHQ_Real** | `REAL` | Flickr-Faces-HQ (FFHQ) / Kaggle Final Dataset | Creative Commons BY-NC-SA 4.0 | High-quality genuine human facial photographs |
| **IMDB_WIKI** | `REAL` | IMDB-WIKI Facial Dataset | Academic Research Use Only (Rothe et al.) | Diverse authentic photographic portraits with varying ages, poses, and lighting |
| **StyleGAN2** | `FAKE` | NVIDIA StyleGAN2 (Karras et al.) / Kaggle Final Dataset | NVIDIA Source Code License / Non-commercial | GAN-generated photorealistic synthetic human faces |
| **StableDiffusion_v1.5** | `FAKE` | RunwayML / StabilityAI SD v1.5 | CreativeML OpenRAIL-M | Latent diffusion text-to-image synthetic portraits |
| **InsightFace_Swap** | `FAKE` | InsightFace SimSwap / AntelopeV2 | MIT / Academic Research Use | Deepfake identity manipulation via facial feature swapping |
| **StableDiffusion_Inpainting** | `FAKE` | StabilityAI SD Inpainting | CreativeML OpenRAIL-M | Partial facial manipulation and synthetic inpainting |
| **Google_Gemini_Imagen3** | `FAKE` | Google DeepMind Gemini / Imagen 3 API | Internal Research / Diagnostic Artifacts | State-of-the-art modern latent diffusion photorealistic human faces |

---

## 3. Leakage Prevention & Integrity Guarantees

1. **Zero Exact Duplicates**: Every file hash (SHA-256 and MD5) was deduplicated globally.
2. **Dedicated Held-Out Gemini Cohort**: 10 distinct Gemini Imagen 3 portraits are strictly reserved for the final test set and were NEVER exposed to training or threshold tuning.
3. **Completely Unseen Inpainting Benchmark**: Stable Diffusion Inpainting (N=200) has 0 samples in train and 0 in validation; it serves as a pure zero-shot cross-generator test.
4. **Independent Genuine Face Controls**: Genuine photographic faces are sourced from two distinct non-overlapping distributions (FFHQ photographic studio portraits and IMDB-WIKI natural lighting portraits).
