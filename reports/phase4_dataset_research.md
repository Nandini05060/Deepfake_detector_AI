# Phase 4: External Evaluation Dataset Research and Suitability Audit

**Author**: Senior Deepfake Detection Researcher  
**Project**: LX-DFD (Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning)  
**Date**: October 2026  
**Status**: Comprehensive Research Audit Completed  

---

## 1. Executive Summary & Research Objectives

Conventional deepfake detectors exhibit near-perfect detection accuracy when evaluated on the same distribution or generator family used during training, yet suffer precipitous performance collapses when exposed to novel, unseen generative models (e.g., transitioning from StyleGAN to Latent Diffusion or FLUX). 

In Phase 2 and Phase 3 of the LX-DFD project:
- **Phase 2** established an in-domain, zero-leakage split (`data/splits_v2/test.csv`, $N=1,956$), demonstrating that LX-DFD achieves **99.39% Accuracy** and **0.9998 ROC-AUC** on in-domain StyleGAN/FFHQ test images.
- **Phase 3** trained a robustness-aware model (`robustness_v2`), demonstrating that training with Gaussian noise, JPEG compression, and blur mitigates severe performance degradation under real-world transmission distortions (reducing mean AUC drop from $0.0815$ to $0.0353$).
- **Critical Research Gap**: Generalization to genuinely unseen AI image generators has remained unverified because the local Kaggle dataset contains only binary labels without generator metadata, and no external benchmark images have been ingested.

This report conducts a rigorous, multi-dimensional academic evaluation of public deepfake and AI-generated face datasets to identify optimal candidates for physical cross-dataset and cross-generator benchmarking.

---

## 2. In-Depth Candidate Dataset Analysis

### 2.1 FaceForensics++ (FF++)
- **Authors & Publication**: Rössler et al., *"FaceForensics++: Learning to Detect Manipulated Facial Images"*, IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
- **Dataset Contents**:
  - 1,000 pristine source video sequences downloaded from YouTube under Creative Commons / standard YouTube licenses.
  - 4 automated facial manipulation methods:
    1. **Deepfakes** (Autoencoder-based face replacement)
    2. **Face2Face** (Facial reenactment transfer of expressions)
    3. **FaceSwap** (Classical graphics-based 3D model fitting face replacement)
    4. **NeuralTextures** (Facial reenactment with learned neural texture representations)
  - Follow-up additions include **FaceShifter** (two-stage identity-preserving face-swap) and **DeepFakeDetection** (consensual actor pairs).
  - 3 video compression levels: Raw (uncompressed), c23 (H.264 high quality, light compression), and c40 (H.264 heavy compression).
- **Generator Families**: Facial Replacement (Autoencoders, FaceSwap, FaceShifter) and Facial Reenactment (Face2Face, NeuralTextures).
- **Ground-Truth Generator Annotations**: Strictly structured into distinct folders per manipulation method (`Deepfakes/`, `Face2Face/`, `FaceSwap/`, `NeuralTextures/`, `original_sequences/`).
- **Provenance of Real Images**: Real face videos harvested from YouTube with consent/creative commons, completely disjoint from the Flickr-Faces-HQ (FFHQ) static photo collection.
- **Licensing & Access Requirements**:
  - Gated Academic Research License issued by Technical University of Munich (TUM).
  - Access requires completing an online Terms of Use agreement form with institutional email (`@university.edu`).
  - Authors supply an automated Python script (`download_faceforensics.py`) with an expiring download token.
- **Storage Constraints & Bandwidth**:
  - Full raw video dataset: $>500$ GB.
  - Compressed c23 video dataset: $\sim 40$ GB.
  - Extracted, face-cropped frame benchmark (10 frames per video, $\sim 50,000$ images): $\sim 12$ GB.
- **Download Instructions**:
  ```bash
  python download_faceforensics.py data/external/FaceForensics/ \
      -d FaceForensics++ -c c23 -t videos --server EU2
  ```
- **Suitability for LX-DFD**:
  - **Strengths**: Benchmark gold standard in forensic literature. Evaluates identity manipulation and boundary blending rather than unconditional generation. High domain independence.
  - **Limitations**: Requires academic application sign-off; video data requires OpenCV face tracking/cropping (`yunet.onnx`) to extract aligned 224x224 facial frames.

---

### 2.2 Celeb-DF (v2)
- **Authors & Publication**: Li et al., *"Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics"*, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
- **Dataset Contents**:
  - 590 real video sequences from YouTube featuring diverse celebrities.
  - 5,639 high-quality deepfake video clips synthesized using an improved deepfake synthesis algorithm that addresses common artifacts (e.g. low resolution, color boundaries, temporal flickering).
  - Average video length $\sim 13$ seconds at 30 fps (over 2 million video frames).
- **Generator Families**: High-quality DeepFake Autoencoder face-swapping with post-blending refinement.
- **Ground-Truth Generator Annotations**: Two classes: `Celeb-real/` and `Celeb-synthesis/`, mapped via a comprehensive `List_of_testing_videos.txt` test split protocol.
- **Provenance of Real Images**: Public celebrity interviews and press junkets on YouTube.
- **Licensing & Access Requirements**:
  - Non-commercial academic research agreement managed by the Computer Vision and Machine Learning Lab at University at Albany, SUNY.
  - Researchers submit a signed application form to receive a Box / Google Drive direct access link.
- **Storage Constraints**: $\sim 32$ GB compressed archive (videos). Cropped test frame subset: $\sim 4$ GB.
- **Download Instructions**:
  Download authorized zip archives via provided institutional Box link:
  ```bash
  unzip Celeb-DF-v2.zip -d data/external/Celeb-DF-v2/
  ```
- **Suitability for LX-DFD**:
  - **Strengths**: Specifically engineered to minimize spatial boundary artifacts, making it a critical test for LX-DFD's Frequency Branch (detecting subtle upsampling harmonics in facial regions).
  - **Limitations**: Manual access form approval required; only evaluates one primary generation method (improved face-swap), so it cannot assess diffusion or GAN generator diversity alone.

---

### 2.3 GenImage
- **Authors & Publication**: Zhu et al., *"GenImage: A Large-Scale Benchmark for Image Generation Detection"*, Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS Datasets and Benchmarks Track), 2023.
- **Dataset Contents**:
  - 1,331,167 total images across 8 diverse modern generative architectures:
    1. **Stable Diffusion v1.4** (Latent Diffusion)
    2. **Stable Diffusion v1.5** (Latent Diffusion)
    3. **Midjourney** (Proprietary Diffusion)
    4. **ADM** (Guided Diffusion)
    5. **GLIDE** (Cascaded Diffusion with Text Guidance)
    6. **Wukong** (Multilingual Latent Diffusion)
    7. **VQDM** (Vector Quantized Diffusion)
    8. **BigGAN** (Large-Scale GAN)
  - Paired with real photographs from the ImageNet (ILSVRC-2012) dataset.
- **Generator Families**: Broad coverage of **Diffusion Models** (Latent, Guided, Cascaded, VQ) and **GANs** (BigGAN).
- **Ground-Truth Generator Annotations**: Explicitly partitioned into distinct folders per generator (`imagenet_ai_0419_sdv4`, `imagenet_ai_0419_midjourney`, etc.), with dedicated `train/` and `val/` splits.
- **Provenance of Real Images**: ImageNet (ILSVRC2012) natural photographs.
- **Licensing & Access Requirements**:
  - Permissive academic open access hosted on OpenDataLab and Hugging Face.
  - Open download via Hugging Face Hub (with standard user authentication token).
- **Storage Constraints**: Total dataset $>120$ GB. Individual generator validation splits: $\sim 3$–$6$ GB per generator.
- **Download Instructions**:
  ```bash
  huggingface-cli download OpenGVLab/GenImage --repo-type dataset --include "imagenet_ai_0419_midjourney/val/*" --local-dir data/external/GenImage_Midjourney
  ```
- **Suitability for LX-DFD**:
  - **Strengths**: Premier academic benchmark for testing generalization to modern diffusion architectures (SD, Midjourney).
  - **Limitations**: Sourced from ImageNet classes (animals, objects, vehicles, humans). To evaluate facial detectors like LX-DFD, images must be pre-filtered using a face detector (`yunet.onnx`) to extract human face crops, or filtered by human-related ImageNet synsets (e.g., `groom`, `scuba_diver`, `baseball_player`).

---

### 2.4 StyleGAN3 Synthetic Faces Benchmark (NVIDIA / Ojha et al., CVPR 2023)
- **Authors & Publication**: Karras et al., *"Alias-Free Generative Adversarial Networks (StyleGAN3)"*, NeurIPS 2021; Ojha et al., *"Towards Universal Fake Image Detectors that Generalize Across Generative Models"*, CVPR 2023.
- **Dataset Contents**:
  - Unconditional synthetic human face portraits generated using official StyleGAN3 checkpoints (`stylegan3-t-ffhqu-1024x1024.pkl` and `stylegan3-r-ffhqu-1024x1024.pkl`).
  - StyleGAN3 introduces rotational and translational equivariance to eliminate "texture sticking," completely re-engineering the convolutional upsampling layers and continuous alias-free filter designs.
- **Generator Families**: Alias-free Generative Adversarial Networks (StyleGAN3-T, StyleGAN3-R).
- **Ground-Truth Generator Annotations**: Verified StyleGAN3 synthesis seeds with zero post-processing.
- **Provenance of Real Images**: FFHQ test set (requires verification of split boundaries to avoid training leakage).
- **Licensing & Access Requirements**: NVIDIA Research License (permissive for research and educational benchmarking). Pre-generated batches available on Kaggle and Hugging Face.
- **Storage Constraints**: $\sim 200$ MB for 1,000 high-resolution images.
- **Suitability for LX-DFD**:
  - **Strengths**: **Crucial scientific benchmark for LX-DFD's Frequency Branch**. Conventional GAN detectors identify StyleGAN1/2 by exploiting high-frequency spectral grid peaks produced by transposed convolutions. Because StyleGAN3 removes these periodic spectral artifacts with continuous low-pass sinc filters, testing on StyleGAN3 directly answers whether LX-DFD relies on brittle aliasing artifacts or genuine facial synthesis cues.
  - **Limitations**: Fake faces only; must be paired with independent authentic faces.

---

### 2.5 WildDeepfake
- **Authors & Publication**: Zi et al., *"WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection"*, ACM International Conference on Multimedia (ACM MM), 2020.
- **Dataset Contents**:
  - 7,314 face video sequences harvested from diverse internet sources (video-sharing platforms, social media, online forums).
  - Contains rich diversity in scene lighting, facial head poses, occlusions, and varied, unknown real-world deepfake software packages.
- **Generator Families**: In-the-wild manipulations (Autoencoders, Faceswap, Wav2Lip reenactments).
- **Ground-Truth Generator Annotations**: Binary labels (`real` vs. `fake`); specific generator software not documented per video.
- **Provenance of Real Images**: Internet public videos.
- **Licensing & Access Requirements**: Academic research agreement; accessible via Google Drive request.
- **Storage Constraints**: $\sim 10$ GB.
- **Suitability for LX-DFD**:
  - **Strengths**: Realistic evaluation of in-the-wild transmission noise and uncontrolled compression.
  - **Limitations**: Lacks generator metadata, preventing disaggregated cross-generator analysis.

---

### 2.6 DiffusionFace / Text-to-Image Facial Benchmarks (SDXL, Midjourney v6, FLUX.1)
- **Description**: Curated test collections of state-of-the-art text-to-image synthesized portraits generated via prompts such as *"close-up photographic portrait of a person, studio lighting, highly detailed, 8k"*.
- **Generators Represented**:
  - **Stable Diffusion XL (SDXL)**: 3.5B parameter latent diffusion model.
  - **Midjourney v6**: Proprietary diffusion synthesizer known for photorealistic skin textures.
  - **FLUX.1 [dev/schnell]**: 12B parameter flow-matching transformer synthesizer.
- **Ground-Truth Generator Annotations**: 100% ground-truth tagged by construction (prompt generation logs).
- **Provenance of Real Images**: Paired with CelebA-HQ or independent Creative Commons photography.
- **Licensing & Access Requirements**: Permissive public research datasets on Hugging Face (e.g. `DiffusionFace`, `SynthWild`).
- **Storage Constraints**: $\sim 500$ MB to $2$ GB for 1,000 to 5,000 face crops.
- **Suitability for LX-DFD**:
  - **Strengths**: Unseen by training data; modern flow-matching and latent diffusion architectures exhibit fundamentally different frequency signatures than GANs.
  - **Limitations**: Public collections vary in facial alignment quality; face detection and cropping (`yunet.onnx`) is mandatory for standardized evaluation.

---

## 3. Comparative Dataset Synthesis Matrix

| Dataset Candidate | Primary Modality | Generators Documented | Generator Families | Open Access | Storage Footprint | Suitability for Unseen Generator Test |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ (FF++)** | Video Frames (C23/Raw) | Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter | Autoencoder, Reenactment, Computer Graphics | Gated (TUM Form) | $12$–$40$ GB | **High (Face Manipulation)** |
| **Celeb-DF (v2)** | Video Frames | Celeb-DF Deepfake | High-Quality Autoencoder Face-Swap | Gated (Albany Form) | $4$–$32$ GB | **High (Subtle Blending)** |
| **GenImage (Face Subset)** | Static Images | SD v1.4, SD v1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN | Latent Diffusion, Guided Diffusion, BigGAN | Open (Hugging Face) | $6$–$12$ GB (Subsets) | **Very High (Diffusion Generalization)** |
| **StyleGAN3 Official Synthetic** | Static Portraits | StyleGAN3-T, StyleGAN3-R | Alias-Free GAN | Open (NVIDIA) | $<1$ GB | **Critical (Anti-Aliasing Test)** |
| **DiffusionFace / FLUX.1** | Static Portraits | SDXL, Midjourney v6, FLUX.1 | Flow-Matching, Latent Diffusion | Open (Hugging Face) | $1$–$2$ GB | **Very High (Modern Synthesizers)** |
| **WildDeepfake** | Video Sequences | Unknown in-the-wild tools | Diverse unknown internet manipulations | Gated (Google Drive) | $\sim 10$ GB | **Moderate (No Generator Tags)** |

---

## 4. Prioritized Recommendations for LX-DFD Academic Benchmarking

Based on the research findings and physical operational constraints (network bandwidth, storage availability, and legal licensing terms), the following tiered external evaluation plan is recommended:

1. **Tier 1 (Immediate Open Benchmark: Modern Text-to-Image Generators)**:
   - Construct a standardized, curated external test split in `data/unseen_generator_test/` consisting of:
     - **Real**: 200 authentic faces from CelebA-HQ / Creative Commons portraits (verified zero hash overlap with FFHQ training split).
     - **StyleGAN3**: 200 synthetic faces from official StyleGAN3-T checkpoints.
     - **Midjourney v6**: 200 synthetic faces generated via Midjourney v6.
     - **FLUX.1 / SDXL**: 200 synthetic faces generated via FLUX.1-dev or SDXL.
   - **Rationale**: Completely open, immediately downloadable ($<2$ GB), zero institutional approval delay, and directly evaluates the exact research gap (generalizing from StyleGAN to alias-free GANs and modern diffusion models).

2. **Tier 2 (Academic Facial Manipulation Benchmark: FaceForensics++ & Celeb-DF)**:
   - Submit the formal academic application to Technical University of Munich for FaceForensics++ c23 and University at Albany for Celeb-DF v2.
   - Extract 10 aligned facial frames per test video using `yunet.onnx`.
   - **Rationale**: Provides standardized comparison against published literature on identity swapping and facial reenactment.
