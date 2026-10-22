# Phase 7: Dataset Provenance, Licensing & Manifest Schema Audit

**Author**: Senior Deepfake Detection Researcher  
**Project**: LX-DFD (Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning)  
**Date**: October 2026  
**Status**: Formal Audit & Compliance Document  

---

## 1. Executive Summary & Audit Mandate

The primary objective of Phase 7 is to transition LX-DFD from a single-generator training distribution (StyleGAN2 fakes and FFHQ reals) to a scientifically valid **multi-generator paradigm**.

In accordance with strict experimental protocols, no external dataset may be ingested without verifying:
1. **Official Provenance & Peer-Reviewed Publication Source**.
2. **Legal Licensing, Terms of Use & Academic Redistribution Rights**.
3. **Physical Accessibility & Download Mechanism** (avoiding unauthenticated scraping).
4. **Facial Alignment Suitability** (distinguishing human portraits from general scenes).
5. **Storage Footprint & Local Resource Safety** (ensuring local machine disk capacity is respected).
6. **Ground-Truth Label Rigor** (zero synthesis-type hallucination or visual guessing).

---

## 2. In-Depth Audit of Candidate Dataset Sources

### 2.1 OpenRL / DeepFakeFace (DFF) (arXiv:2309.02218, 2023)
- **Publication & Authors**: OpenRL Team, *"DeepFakeFace: A Comprehensive High-Quality Deepfake Face Benchmark"*, arXiv preprint arXiv:2309.02218, 2023.
- **Repository URL**: [https://huggingface.co/datasets/OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace).
- **Generators Present**:
  - `StableDiffusion_v1.5` (Latent Text-to-Image Diffusion)
  - `InsightFace_Swap` (Facial Identity Replacement)
  - `StableDiffusion_Inpainting` (Mask-Conditioned Facial Inpainting)
  - Authentic Controls: `IMDB-WIKI` celebrity portrait photography
- **Licensing & Research Restrictions**: Permissive Apache-2.0 and OpenRAIL licenses for open scientific research.
- **Image Characteristics**: Authentic and synthetic human portraits.
- **Download Safety & Storage Management**:
  - Full archive: $>15$ GB.
  - Curated download implementation: [`scripts/download_external_eval_dataset.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/scripts/download_external_eval_dataset.py) utilizes HTTP 206 Partial Content byte-range streaming via `fsspec`, extracting only the required $800$ verified images without downloading full remote archives.
  - Local disk footprint: $\sim 115$ MB.
- **Audit Verdict**: **APPROVED FOR INGESTION**.

---

### 2.2 Kaggle 140k Real and Fake Faces (Phase 1 Baseline Dataset)
- **Provenance & Authors**: Kshitiz Bhargava, Kaggle Research Dataset; source images derived from Flickr-Faces-HQ (FFHQ) and StyleGAN2 ($256 \times 256$ aligned crops).
- **Generators Present**:
  - Authentic Controls: `FFHQ` authentic human portraits ($N=6,000$).
  - Synthetic Forgeries: `StyleGAN2` unconditional convolutional GAN fakes ($N=6,890$).
- **Licensing & Research Restrictions**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Storage Footprint**: Cached locally in `.cache/kagglehub/` ($12,890$ images, $\sim 1.4$ GB).
- **Audit Verdict**: **APPROVED FOR INGESTION** (Core in-domain baseline).

---

### 2.3 GenImage Benchmark (NeurIPS 2023)
- **Publication & Authors**: Mingran Zhu et al., *"GenImage: A Large-Scale Benchmark for Image Generation Detection"*, NeurIPS 2023.
- **Repository URL**: [https://github.com/SysCV/GenImage](https://github.com/SysCV/GenImage) | `OpenGVLab/GenImage`.
- **Generators Present**: Stable Diffusion v1.4, Stable Diffusion v1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN.
- **Licensing & Research Restrictions**: CC BY-NC 4.0.
- **Accessibility & Access Restrictions**:
  - Direct unauthenticated access to `OpenGVLab/GenImage` on Hugging Face Hub returns **`HTTP 401: Unauthorized`**; downloading requires explicit user authentication tokens.
- **Image Content Suitability**:
  - *Critical Limitation*: GenImage is derived from ImageNet-1K, consisting primarily of general object classes (animals, vehicles, instruments, landscapes).
  - Merely containing AI-generated images does **not** mean it contains usable human face images. Indiscriminate ingestion would inject non-face objects into face forensic models.
- **Audit Verdict**: **BLOCKED FROM BULK DOWNLOAD**. Must not be ingested until an authenticated token is configured and images are filtered using YuNet face detection on human-centric synsets.

---

### 2.4 FaceForensics++ (ICCV 2019) & Celeb-DF v2 (CVPR 2020)
- **Licensing & Restrictions**: **Gated Academic Licenses**. Access to raw archives requires formal application and signed terms of use with institutional credentials (`@university.edu`).
- **Storage Requirements**: Raw video archives exceed $500$ GB for FF++ and $32$ GB for Celeb-DF v2.
- **Current Disk Availability**: Local machine drive `C:` has $39.17$ GB of free storage. Attempting to download raw video datasets would cause disk exhaustion.
- **Audit Verdict**: **BLOCKED**. Gated licensing and storage constraints prevent automated ingestion.

---

## 3. Storage Footprint & System Resources Summary

| Dataset Candidate | Licensing | Accessibility | Disk Required | Action in Phase 7 |
|---|:---:|:---:|:---:|:---:|
| **Kaggle 140k (FFHQ / StyleGAN2)** | CC BY-NC-SA 4.0 | Local Cache | $1.4$ GB | **Active (In-Domain)** |
| **OpenRL / DeepFakeFace ($N=800$)** | Apache-2.0 / OpenRAIL | HTTP Streamed | $115$ MB | **Active (External Diversity)** |
| **GenImage (Human Subsets)** | CC BY-NC 4.0 | Gated (Token 401) | $>12$ GB | **Deferred (Authentication Required)** |
| **FaceForensics++ (C23)** | Gated Academic | Blocked (TUM Form) | $>40$ GB | **Blocked (Licensing/Storage)** |
| **Celeb-DF v2** | Gated Academic | Blocked (Albany Form) | $>32$ GB | **Blocked (Licensing/Storage)** |

**Storage Budget**: Total active dataset footprint is $\sim 1.52$ GB, leaving $>37.6$ GB of free disk space on drive `C:`.

---

## 4. Multi-Generator Metadata Manifest Schema Audit

The unified metadata manifest ([`data/manifests/phase7_multigen_manifest.csv`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/data/manifests/phase7_multigen_manifest.csv)) was generated via [`scripts/build_phase7_manifest.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/scripts/build_phase7_manifest.py).

### 4.1 Schema Verification

| Field Name | Data Type | Null Count | Description & Constraints |
|---|:---:|:---:|---|
| `path` | `string` | $0$ ($0.0\%$) | Absolute or canonical path to local image on disk. |
| `label` | `string` | $0$ ($0.0\%$) | Binary ground truth: strictly `REAL` or `FAKE`. |
| `source_dataset` | `string` | $0$ ($0.0\%$) | Provenance dataset: `FFHQ`, `Kaggle_DFD`, `IMDB-WIKI`, `DeepFakeFace`. |
| `generator` | `string` | $0$ ($0.0\%$) | Ground-truth generative architecture: `None`, `StyleGAN2`, `StableDiffusion_v1.5`, `InsightFace_Swap`, `StableDiffusion_Inpainting`. |
| `identity_id` | `string` | $0$ ($0.0\%$) | Subject identity cluster identifier to prevent identity cross-split leakage. |
| `original_group_id` | `string` | $0$ ($0.0\%$) | Augmentation and perceptual hash grouping identifier. |
| `split` | `string` | $0$ ($0.0\%$) | Partition assignment: `train`, `val`, `test_indomain`, `test_unseen_logo`, `test_unseen_eval`. |
| `face_detection_status` | `string` | $0$ ($0.0\%$) | Preprocessing verification status: `pre_aligned_face` or `yunet_candidate`. |
| `preprocessing_version` | `string` | $0$ ($0.0\%$) | Active preprocessing version tag: strictly `phase7_v1`. |

### 4.2 Generator Label Integrity
- **Zero Hallucination Rule**: No generator label was inferred from visual appearance or CNN predictions.
- All generator labels (`StyleGAN2`, `StableDiffusion_v1.5`, `InsightFace_Swap`, `StableDiffusion_Inpainting`) derive strictly from original generator manifests and directory structures.

---

## 5. Audit Conclusion

The dataset manifest satisfies all provenance, legal licensing, facial suitability, storage safety, and schema integrity criteria. Ingestion into training and evaluation pipelines is formally authorized.
