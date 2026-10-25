# Phase 12B Dataset Integrity, Provenance & Gemini Cohort Audit
## Systematic Audit of Manifest Partitions, Demographics, Generator Families, and Statistical Scope

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Auditor:** Senior Deepfake Detection Specialist & Research Auditor  
**Document Identifier:** `reports/phase12_dataset_audit.md`  
**Associated File:** `reports/phase12_dataset_manifest_summary.json`  

---

## 1. Executive Summary

A comprehensive dataset integrity audit was conducted across all registered images in `data/manifests/phase11_generator_manifest.csv` ($N = 1,864$) and active physical assets on disk.

### Key Audit Findings:
1. **Cryptographic Integrity & Uniqueness:**
   - 1,864 distinct images verified via SHA-256 and MD5 hashing.
   - Zero exact file duplicates ($N=0$).
   - Zero near-duplicate image overlap between training ($N=1,165$), validation ($N=242$), and locked test partitions ($N=457$).
2. **Generative Diversity Covered:**
   - **GAN Architecture:** StyleGAN2 ($N=420$).
   - **Latent Diffusion Architectures:** Stable Diffusion v1.5 ($N=200$), Stable Diffusion Inpainting ($N=200$), Google Gemini / Imagen 3 ($N=24$).
   - **Identity Swapping Pipeline:** InsightFace Swap ($N=200$).
   - **Authentic Photographic Negative Controls:** High-quality human portraits from FFHQ, IMDB-WIKI, and Kaggle authentic controls ($N=820$).
3. **Strict Evaluation Isolation:**
   - `StableDiffusion_Inpainting` ($N=200$) is 100% held-out (0 training or validation samples).
   - `Google_Gemini_Imagen3` has $N=17$ strictly locked test samples (13 Phase 11 images + 4 Phase 10 diagnostic images).

---

## 2. Partition Breakdown & Generator Inventory

| Generator Family | Class | Source / License | Train | Val | Test (Locked) | Total | Partition Role |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **None_Real** | REAL | FFHQ / IMDB-WIKI (CC-BY-NC-SA 4.0) | 580 | 120 | 120 | 820 | Authentic Negative Controls |
| **StyleGAN2** | FAKE | Kaggle DFD / StyleGAN2 (CC-BY 4.0) | 240 | 60 | 60 | 420 | In-Domain GAN Benchmark |
| **StableDiffusion_v1.5** | FAKE | OpenRL/DeepFakeFace (OpenRAIL) | 140 | 30 | 30 | 200 | In-Domain Diffusion Benchmark |
| **InsightFace_Swap** | FAKE | OpenRL/DeepFakeFace (Research) | 140 | 30 | 30 | 200 | In-Domain Face-Swap Benchmark |
| **StableDiffusion_Inpainting** | FAKE | OpenRL/DeepFakeFace (OpenRAIL) | 0 | 0 | 200 | 200 | **100% Held-Out Unseen Inpainting** |
| **Google_Gemini_Imagen3** | FAKE | Google Imagen 3 API / Research | 5 | 2 | 17 | 24 | **Held-Out Modern Diffusion** |
| **TOTAL** | | | **1,165** | **242** | **457** | **1,864** | |

---

## 3. Gemini / Imagen 3 Cohort Audit

### 3.1 Demographic & Condition Diversity
The 17 test images evaluate a varied distribution of attributes:
- **Age Representation:** 18 to 70 years old (mean age: 39.5 years).
- **Gender Balance:** 9 male portraits, 8 female portraits.
- **Ethnic Diversity:** East Asian (2), South Asian (2), Black / African American (2), Caucasian (1), Hispanic (1), Middle Eastern (1), West African (1), Scandinavian (1), Southeast Asian (1), Latin American (1), plus 4 general diagnostic portraits.
- **Illumination Variations:** Rembrandt studio lighting, bright direct sunlight, office fluorescent, twilight street ambiance, indoor low-light specular.
- **Background Contexts:** Natural foliage, urban streets, academic libraries, modern domestic interiors, gym athletic environments.

### 3.2 Scientific Limitation & Statistical Scope
- **Sample Size Consideration:** While the $N=17$ cohort exhibits wide demographic diversity, $N=17$ is small for cross-generator benchmark claims.
- **Confidence Intervals:** At $k=2$ detected out of $N=17$ (11.76%), the 95% Wilson confidence interval spans:
  $$\text{CI}_{95\%} = [3.2\%, \, 34.3\%]$$
- **Integrity Rule Compliance:** As mandated by Research Integrity Rule 6 ("Never claim generalization from a tiny Gemini sample"), conclusions derived from this cohort must be framed as *preliminary diagnostic evidence* of failure modes, rather than an exhaustive characterization of all Imagen 3 outputs.
- **Provenance Recording:** All images have registered SHA-256 hashes, creation timestamps, and generation prompt metadata logged in `data/manifests/phase11_generator_manifest.csv`.
