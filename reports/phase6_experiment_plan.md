# Phase 6: Controlled Experiment Plan — Generator-Diverse Training Protocol

**Author**: Senior Deepfake Detection Researcher  
**Project**: LX-DFD (Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning)  
**Date**: October 2026  
**Status**: Formal Experiment Specification & Execution Protocol  

---

## 1. Scientific Motivation & Core Hypothesis

The failure of 2D DCT Spectral Dropout in Phase 5 demonstrated that architectural regularization applied to a single generative distribution cannot magically synthesize the missing physical characteristics of distinct generative paradigms. 

- **StyleGAN2**: Synthesizes images via convolutional upsampling, creating narrow, phase-locked harmonic spikes in the 2D DCT log-magnitude spectrum.
- **Latent Diffusion**: Synthesizes images via iterative reverse stochastic differential equations (Langevin dynamics) in a continuous latent manifold, creating smooth, broadband frequency distributions without transposed convolution spikes.
- **Face Swapping**: Warps and splices source facial features onto target heads, introducing spatial boundary blending seams, color mismatch, and Poisson smoothing gradients.

### Core Hypothesis:
> **Hypothesis**:  
> Training the dual-branch spatial-frequency architecture on a **balanced multi-generator cohort** (incorporating both GAN upsampling artifacts and Latent Diffusion synthesis patterns) will enable the network to learn shared, invariant representations of artificial synthesis. By preventing the frequency branch from treating StyleGAN grid spikes as the sole indicator of forgery, the model will achieve robust generalization to entirely unobserved generator families (e.g. Flow-Matching FLUX.1 / Face-Swapping) without degrading clean in-domain accuracy.

---

## 2. Controlled Multi-Generator Partition Design

To prevent identity and augmentation leakage while enabling rigorous Leave-One-Generator-Out (LOGO) cross-validation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 6 MULTI-GENERATOR PARTITION DESIGN                    │
├──────────────────────┬─────────────────────────────────┬────────────────────┤
│ Generative Family    │ Architectures                   │ Partition Role     │
├──────────────────────┼─────────────────────────────────┼────────────────────┤
│ Family A: GANs       │ StyleGAN2 + StyleGAN3           │ Training / Val     │
├──────────────────────┼─────────────────────────────────┼────────────────────┤
│ Family B: Diffusion  │ Stable Diffusion (v1.5 / SDXL)  │ Training / Val     │
├──────────────────────┼─────────────────────────────────┼────────────────────┤
│ Family C: Swap       │ InsightFace Swap / SimSwap      │ Training / Val     │
├──────────────────────┼─────────────────────────────────┼────────────────────┤
│ Family D: Held-Out   │ FLUX.1 (Flow-Matching)          │ Unseen LOGO Test   │
├──────────────────────┼─────────────────────────────────┼────────────────────┤
│ Authentic Real       │ FFHQ + CelebA-HQ + IMDB-WIKI    │ Balanced Controls  │
└──────────────────────┴─────────────────────────────────┴────────────────────┘
```

### 2.1 Partition Integrity Rules
1. **Source & Identity Grouping**: Images belonging to the same subject identity or derived from the same source photograph/seed must never be split across train, val, and test partitions.
2. **Strict Family Isolation (LOGO)**: Generative Family D (**FLUX.1 Flow-Matching**) is strictly quarantined. Zero FLUX.1 images or prompts will appear in training or validation splits. It serves as the primary unseen benchmark.
3. **Balanced Class & Generator Proportions**:
   - The training set $\mathcal{D}_{\text{train}}$ maintains a $1:1$ ratio of Real to Fake images.
   - The Fake training cohort is equally balanced across Family A (33.3%), Family B (33.3%), and Family C (33.3%).
   - Batch sampler uses stratified sampling to ensure each batch contains balanced representations across all three families.

---

## 3. Training Protocol & Hyperparameter Specifications

To strictly isolate the impact of generator diversity, the training configuration preserves the established Phase 2/3 parameters:

- **Model Architecture**: Proposed LX-DFD Dual-Branch Model with Attention Fusion ([`src/models/lxfd_model.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/src/models/lxfd_model.py)).
  - Spatial Branch: Pretrained EfficientNet-B0 (256-D embedding).
  - Frequency Branch: 2D DCT Log-Magnitude Transform + 4-Layer Spectral CNN (256-D embedding).
  - Fusion: Cross-modal learned attention weights $w_s, w_f$.
  - Dropout: $0.30$; Spectral Dropout: **Disabled** (based on Phase 5 rejection).
- **Augmentation Pipeline**:
  - Balanced JPEG compression ($q \in [60, 95]$, $p=0.35$).
  - Gaussian blur ($r \in [0.5, 1.5]$, $p=0.25$).
  - Random resized crop ($0.85 - 1.0$), horizontal flip ($p=0.5$), subtle color jitter ($0.2$).
  - Gaussian noise ($\sigma \in [0.01, 0.04]$, $p=0.30$).
- **Optimization**:
  - Optimizer: AdamW ($\text{lr} = 1\times 10^{-4}$, $\text{weight\_decay} = 1\times 10^{-4}$).
  - Scheduler: Cosine Annealing with 2-epoch linear warmup.
  - Loss Function: `BinaryCrossEntropyWithLogitsLoss()`.
  - Batch Size: 32; Epochs: 15; Early Stopping Patience: 5.
  - Precision: PyTorch Automatic Mixed Precision (AMP, FP16/FP32).
  - Target Hardware: NVIDIA GeForce RTX 3050 Laptop GPU.
- **Model Checkpoint Policy**:
  - Checkpoint Path: `checkpoints/lxfd_multigen_diverse_best.pt`.
  - Selection Metric: Strict validation ROC-AUC on multi-generator validation partition.
  - Immutability: Baseline checkpoints (`lxfd_attention_standard_best.pt`, `lxfd_attention_robustness_v2_best.pt`, `efficientnet_best.pt`) remain frozen and unmodified.

---

## 4. Benchmark Suite & Pre-Registered Go/No-Go Criteria

The trained multi-generator model will be benchmarked across the five evaluation tiers:

1. **Tier 2: In-Domain Clean Test** (`data/splits_v2/test.csv`, $N=1,956$).
2. **Tier 4A: Exploratory Benchmark** (`data/unseen_generator_test/manifest.csv`, $N=800$, SD1.5, InsightFace, SD Inpainting).
3. **Tier 4B: Unseen LOGO Benchmark** (FLUX.1 Flow-Matching, $N=400$).
4. **Tier 5: 14-Condition Corruption Suite** (Robustness score).

### 4.1 Go/No-Go Decision Thresholds

| Benchmark Metric | Baseline (LX-DFD Standard) | Target Success Threshold | Scientific Action |
|---|:---:|:---:|---|
| **In-Domain Clean ROC-AUC** | $0.9998$ | $\ge 0.9900$ | Verifies that multi-generator training does not impair native discrimination. |
| **In-Domain Clean Accuracy** | $99.39\%$ | $\ge 98.00\%$ | Preserves high classification utility on primary task. |
| **Stable Diffusion v1.5 AUC** | $0.5046$ | $\ge 0.8500$ | Primary validation of hypothesis: learning multi-family representations. |
| **Unseen LOGO (FLUX.1) AUC** | $\sim 0.5000$ (est.) | $\ge 0.7000$ | Definitive confirmation of cross-generator paradigm generalization. |
| **Composite Robustness Score** | $0.9178$ | $\ge 0.9200$ | Preserves resilience to transmission distortions. |

**Adoption Rule**:
- **Adopt Multi-Generator Model**: If the model achieves $\ge 0.7000$ ROC-AUC on the unobserved LOGO test while maintaining $\ge 0.9900$ in-domain AUC.
- **Negative Result Protocol**: If multi-generator training fails to transfer to unseen flow-matching architectures, the exact failure mode will be documented transparently without post-hoc threshold manipulation.
