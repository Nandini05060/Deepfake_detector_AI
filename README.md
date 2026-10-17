# LX-DFD: Generalizable Deepfake Face Detection Using Spatial-Frequency Feature Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Reproducible Research Codebase and Interactive Web Platform for Dual-Branch Spatial-Frequency Deepfake Detection with Cross-Domain Generalization and Explainable AI (XAI).**

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Base Paper & Research Gap](#-base-paper--research-gap)
- [Key AI & Deep Learning Concepts](#-key-ai--deep-learning-concepts)
- [Architecture Anatomy](#-architecture-anatomy)
- [Benchmark Results](#-benchmark-results)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Web Application & Live Dashboard](#-web-application--live-dashboard)
- [Execution Pipeline](#-execution-pipeline)
- [Explainability & Visualizations (XAI)](#-explainability--visualizations-xai)

---

## 🔍 Project Overview

Deepfake generation technologies driven by Generative Adversarial Networks (GANs) and diffusion models have reached near-photorealistic quality. Conventional deepfake detectors rely solely on spatial (pixel-level) RGB features, making them prone to failure when exposed to:
1. **Unseen image generators and novel deepfake synthesis algorithms** (Cross-Domain Shift).
2. **Real-world transmission corruptions** (JPEG compression, Gaussian noise, motion blur).

**LX-DFD** resolves these limitations by introducing a **Dual-Branch Spatial-Frequency Learning Architecture**:
- **Branch A (Spatial RGB Domain)**: Captures semantic facial cues, lighting variations, and blending boundaries.
- **Branch B (Frequency Domain)**: Transforms images via **2D Discrete Cosine Transform (2D DCT)** to expose invisible GAN upsampling grid artifacts and high-frequency spectral discrepancies.
- **Cross-Modal Attention Fusion**: Dynamically computes learned weights ($w_s, w_f$) to fuse evidence from both domains.

---

## 📄 Base Paper & Research Gap

- **Base Paper**: Patel et al., *"An Improved Dense CNN Architecture for Deepfake Image Detection"*, *IEEE Access*, Volume 11, 2023, pp. 22081–22095.  
  - DOI: [`10.1109/ACCESS.2023.3251417`](https://doi.org/10.1109/ACCESS.2023.3251417)
  - IEEE Xplore: [https://ieeexplore.ieee.org/document/10057390/](https://ieeexplore.ieee.org/document/10057390/)
- **Research Gap**:
  > *"Existing CNN-based deepfake detectors achieve high accuracy when trained and tested on known GAN-generated face datasets, but their ability to generalize to unseen generators, datasets, image transformations, and real-world conditions remains insufficiently investigated."*

---

## 🧠 Key AI & Deep Learning Concepts

| AI Concept | Description & Implementation |
|---|---|
| **Dual-Stream CNN** | Two parallel neural networks extracting spatial visual features and frequency spectrum features simultaneously. |
| **2D Discrete Cosine Transform (DCT)** | Converts spatial pixels into frequency domain components to capture GAN upsampling checkerboard artifacts. |
| **Cross-Modal Attention Fusion** | Self-attention mechanism that dynamically balances the influence of spatial vs. frequency embeddings. |
| **Transfer Learning** | Pretrained **EfficientNet** backbone fine-tuned for high-level facial artifact representations. |
| **Dense Connectivity** | Re-implementation of DenseNet-style feature reuse layers (Patel et al. 2023 baseline). |
| **Explainable AI (Grad-CAM)** | Gradient-weighted Class Activation Mapping to generate interpretable visual decision heatmaps. |
| **Domain Generalization** | Cross-domain and unseen generator testing to evaluate out-of-distribution robustness. |
| **Perturbation Robustness** | Stress-testing models against JPEG compression, Gaussian blur, additive noise, and brightness shifts. |

---

## 🏗️ Architecture Anatomy

```
                              ┌────────────────────────────────────────┐
                              │          Input Face Image (RGB)        │
                              └───────────────────┬────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │       Spatial Branch (RGB)       │             │     Frequency Branch (2D DCT)    │
        ├──────────────────────────────────┤             ├──────────────────────────────────┤
        │  EfficientNet-B0 Backbone        │             │  2D DCT Log-Magnitude Spectrum   │
        │  Adaptive Average Pooling        │             │  4-Stage Spectral ConvNet + Norm │
        │  Linear Projection (256-D)       │             │  Linear Projection (256-D)       │
        └────────────────┬─────────────────┘             └────────────────┬─────────────────┘
                         │ Embedding F_s                                  │ Embedding F_f
                         └────────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │  Cross-Modal Attention Fusion   │
                                 │     F_fused = w_s*F_s + w_f*F_f │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │       Binary MLP Classifier     │
                                 │    Linear(256->128) -> GELU     │
                                 │    Dropout(0.3) -> Linear(1)    │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                   REAL [0]  /  FAKE [1] Output
```

---

## 📊 Verified Benchmark Results (Phase 2, Phase 3 & Phase 4)

Evaluated on **12,890 face images** with leak-free train/val/test splits (`data/splits_v2`, 70% / 15% / 15%):

### 1. In-Domain Test Performance ($N=1,956$, Held-Out Test Set)

| Model Architecture | Test Accuracy | ROC-AUC | PR-AUC | F1-Score | Brier Score | ECE | Robustness Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Simple CNN Baseline** | 73.06% | 0.8138 | 0.8211 | 0.7702 | 0.1765 | 0.0546 | 0.680 |
| **Dense CNN (Patel et al., 2023)** | 96.47% | 0.9934 | 0.9941 | 0.9669 | 0.0262 | 0.0109 | 0.752 |
| **EfficientNet-B0 Baseline** | 99.54% | 0.9997 | 0.9997 | 0.9957 | 0.0034 | 0.0057 | 0.9170 |
| **LX-DFD (Standard Baseline)** | 99.39% | 0.9998 | 0.9998 | 0.9942 | 0.0052 | 0.0048 | 0.9185 |
| **LX-DFD (Robustness-Aware Aug v2)** | **95.91%** | **0.9943** | **0.9950** | **0.9609** | **0.0322** | **0.0222** | **0.9647** |

### 2. Perturbation & Corruption Benchmark (14 Stress Conditions)

| Corruption Type | Baseline LX-DFD AUC | LX-DFD Robustness v2 AUC | Relative Impact |
|---|:---:|:---:|---|
| **Clean / Original** | 0.9998 | 0.9943 | Clean performance preserved ($>99.4\%$) |
| **Gaussian Noise ($\sigma=0.05$)** | 0.6274 | **0.9725** | **$+0.3451$ gain (+55.0% boost, 89.8% acc)** |
| **JPEG Quality 30** | 0.9323 | **0.9822** | **$+0.0499$ gain (93.0% acc)** |
| **JPEG Quality 50** | 0.9637 | **0.9881** | **$+0.0244$ gain (93.9% acc)** |
| **Gaussian Blur ($r=1.5$)** | 0.9677 | **0.9826** | **$+0.0149$ gain (91.6% acc)** |
| **Gaussian Blur ($r=3.0$)** | 0.8601 | **0.9217** | **$+0.0616$ gain (75.9% acc)** |
| **Mean AUC Drop Across Corruptions** | 0.0815 | **0.0353** | **$56.7\%$ reduction in degradation** |

### 3. Phase 4: External Dataset & Unseen-Generator Benchmark

Evaluated on **800 physical samples** from the OpenRL/DeepFakeFace benchmark ($200$ IMDB-WIKI Real faces, $200$ Stable Diffusion v1.5, $200$ InsightFace Swap, $200$ SD Inpainting), completely independent of training:

| Architecture / Checkpoint | In-Domain ROC-AUC | In-Domain Accuracy | Unseen Macro ROC-AUC | Stable Diffusion v1.5 AUC | InsightFace Swap AUC | SD Inpainting AUC | Unseen Macro Accuracy | Unseen Macro F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Simple CNN Baseline** | 0.8138 | 81.39% | **0.5987** | **0.7074** | 0.5647 | 0.5241 | **55.92%** | **0.4452** |
| **Dense CNN (Patel et al., 2023)** | 0.9934 | 99.34% | 0.5519 | 0.6062 | 0.5499 | 0.4996 | 51.25% | 0.1484 |
| **EfficientNet-B0 Baseline** | 0.9997 | **99.54%** | 0.5648 | 0.5068 | 0.5938 | 0.5938 | 49.83% | 0.0593 |
| **Proposed LX-DFD (Standard)** | **0.9998** | 99.39% | 0.5771 | 0.5046 | **0.6292** | **0.5977** | 50.33% | 0.0418 |
| **Proposed LX-DFD (Robustness v2)** | 0.9943 | 95.91% | 0.5540 | 0.5431 | 0.5685 | 0.5503 | 50.92% | 0.0537 |

#### Key Insights & Forensic Takeaways:
- **Face Swap & Inpainting Generalization**: Proposed LX-DFD achieves the highest discrimination on facial manipulations: **0.6292 ROC-AUC** on InsightFace Swap and **0.5977** on SD Inpainting, detecting subtle facial blending boundary seam artifacts.
- **The Upsampling Grid Trap**: High-capacity models trained solely on StyleGAN overfit to narrow transposed convolution frequency spikes. When evaluated on Latent Diffusion (SD1.5), which uses iterative denoising in latent space without transposed convolutions, standard models drop to chance ($\sim 0.50$ AUC). `robustness_v2` recovers $+3.85\%$ AUC on SD1.5 with **85.71% Precision**.
- **Solution — Spectral Dropout**: Formulated in the [Phase 4 Next Experiment Plan](reports/phase4_next_experiment_plan.md) to stochastically drop 2D DCT log-magnitude blocks during training, breaking dependence on narrow upsampling spikes.

#### Phase 4 Academic Documentation & Research Artifacts
- **[Dataset Research Report](reports/phase4_dataset_research.md)**: Academic audit of FaceForensics++, Celeb-DF v2, GenImage, StyleGAN3, and Diffusion benchmarks.
- **[External Evaluation Protocol](reports/phase4_external_evaluation_protocol.md)**: Standardized zero-leakage cross-generator protocol with predeclared thresholds and calibration policies.
- **[Data Independence & Leakage Audit](reports/phase4_data_independence_audit.md)**: Forensic origin audit proving training images derive from FFHQ and StyleGAN1/2.
- **[External Results & Generalization Report](reports/phase4_external_results.md)**: Physical cross-generator empirical benchmarks with 95% bootstrap confidence intervals.
- **[Comprehensive Model Comparison Matrix](reports/phase4_model_comparison.csv)**: Complete machine-readable matrix across all 5 architectures.
- **[Spectral Dropout Experiment Plan](reports/phase4_next_experiment_plan.md)**: Formal research plan and compute budget for 2D DCT spectral dropout training.

---

## 📁 Project Directory Structure

```plaintext
DEEPFAKE DETECTION/
├── app/
│   ├── backend/
│   │   └── main.py              # FastAPI application & inference endpoints
│   └── frontend/
│       ├── index.html           # Modern AI laboratory research dashboard
│       ├── styles.css           # Premium dark-mode modern laboratory theme
│       └── app.js               # Interactive frontend controller & charts
├── checkpoints/                 # Trained model weights (.pt) and history logs (.json)
│   ├── dense_cnn_best.pt
│   ├── efficientnet_best.pt
│   ├── lxfd_attention_standard_best.pt
│   ├── lxfd_attention_robustness_v2_best.pt
│   ├── calibration_params.json
│   └── simple_cnn_best.pt
├── configs/
│   └── config.yaml              # Hyperparameters, augmentation & training configs
├── data/
│   ├── splits_v2/               # Leak-free train (70%), val (15%), test (15%) splits
│   └── unseen_generator_test/   # Harness for external generator evaluation
├── reports/                     # Audit, evaluation & robustness reports
│   ├── phase2_verification.md
│   ├── phase2_metrics.json
│   ├── phase2_comparison.csv
│   ├── phase3_baseline_verification.md
│   ├── phase3_experiment_plan.md
│   ├── phase3_metrics.json
│   ├── phase3_comparison.csv
│   ├── phase3_corruption_analysis.csv
│   ├── phase3_unseen_generator_results.md
│   └── phase3_final_recommendations.md
├── scripts/                     # Executable scripts for audit, training, testing
│   ├── audit_dataset.py
│   ├── create_leak_free_splits.py
│   ├── calibrate_model.py
│   ├── evaluate_unseen_generators.py
│   ├── robustness_test.py
│   ├── run_phase2_verification.py
│   ├── run_phase3_evaluation.py
│   ├── train_baseline.py
│   └── train_proposed.py
├── src/
│   ├── data/                    # Dataset loaders, augmentations, and splitters
│   ├── evaluation/              # Metrics, cross-domain, robustness & error analysis
│   ├── explainability/          # Grad-CAM heatmap generator
│   ├── frequency/               # 2D DCT transformations & spectral visualizations
│   ├── models/                  # PyTorch model definitions (LX-DFD, CNNs, Fusion)
│   ├── training/                # Losses, schedulers, and training loop engine
│   └── utils/                   # Device manager, logger, and reproducibility seed
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/Nandini05060/Deepfake-detector.git
cd "DEEPFAKE DETECTION"
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🌐 Web Application & Live Dashboard

Launch the FastAPI backend server and interactive web laboratory:

```bash
uvicorn app.backend.main:app --reload --port 8000
```

Once running, visit:
- **Interactive UI Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Dashboard Features:
- 📤 **Live Deepfake Scanner**: Drag-and-drop any face image for real-time inference.
- 🔬 **Grad-CAM Heatmap Viewer**: Visual explanation overlay highlighting facial artifacts.
- ⚡ **2D DCT Spectral Inspector**: Real-time frequency spectrum and high-pass residual filter.
- ⚖️ **Attention Weight Breakdown**: Visual display of $w_s$ (spatial) and $w_f$ (frequency) confidence contribution.
- 📈 **Benchmark Visualizer**: Interactive ROC curves, confusion matrices, and ablation comparisons.

---

## ⚙️ Execution Pipeline

### Step 1: Audit Dataset & Prevent Leakage
```bash
python scripts/audit_dataset.py
python scripts/create_splits.py
```

### Step 2: Train Baseline Models
```bash
# Simple CNN
python scripts/train_baseline.py --model simple_cnn

# Dense CNN (Patel et al., 2023)
python scripts/train_baseline.py --model dense_cnn

# EfficientNet Baseline
python scripts/train_baseline.py --model efficientnet
```

### Step 3: Train Proposed LX-DFD Architecture
```bash
# Proposed Attention-Fusion LX-DFD
python scripts/train_proposed.py --fusion attention --aug-mode standard
```

### Step 4: Run Robustness & Cross-Domain Benchmarking
```bash
python scripts/robustness_test.py --checkpoint checkpoints/lxfd_attention_standard_best.pt --model-type lxfd
```

### Step 5: Print Model Anatomy & Parameter Count
```bash
python scripts/print_model_summary.py
```

### Step 6: External Dataset & Unseen-Generator Benchmarking
```bash
# Dry-run integrity audit on external dataset
python scripts/evaluate_unseen_generators.py --dry-run --manifest data/unseen_generator_test/manifest.csv

# Evaluate trained models across external generators
python scripts/evaluate_unseen_generators.py \
    --checkpoint checkpoints/lxfd_attention_robustness_v2_best.pt \
    --manifest data/unseen_generator_test/manifest.csv \
    --model-type lxfd \
    --fusion attention \
    --output-json reports/phase4_unseen_lxfd_robustness_v2.json \
    --output-csv reports/phase4_unseen_lxfd_robustness_v2.csv

# Train Spectral Dropout Model (Phase 4 Next Experiment Plan)
python scripts/train_proposed.py \
    --fusion attention \
    --aug-mode standard \
    --spectral-dropout \
    --spectral-dropout-p 0.20
```

---

## 👁️ Explainability & Visualizations (XAI)

LX-DFD includes native Explainable AI tools:
1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Computes gradients from the final convolutional layer of the spatial branch to highlight the facial areas (e.g. eyes, mouth, blending seams) that influenced the prediction.
2. **Frequency Domain Visualizer**: Converts the input to a 2D DCT log-magnitude map and high-frequency residual colormap, revealing GAN lattice artifacts that are otherwise invisible to human observers.

---

## 📜 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
