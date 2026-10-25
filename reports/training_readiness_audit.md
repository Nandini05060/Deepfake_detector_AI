# LX-DFD — Pre-Training Readiness & System Audit

**Repository**: [https://github.com/Nandini05060/Deepfake_detector_AI](https://github.com/Nandini05060/Deepfake_detector_AI)  
**Execution Date**: 2026-10-25  
**Auditor**: Senior Machine Learning & Computer Vision Research Auditor  
**Primary Research Goal**: Retrain and calibrate the LX-DFD dual-branch model for cross-generator generalization, specifically mitigating Gemini/Imagen 3 false-real predictions without degrading genuine photography specificity ($\ge 88\%$).

---

## 1. System & Hardware Resource Profile

| Component | Specification | Operational Constraint / Setting |
|---|---|---|
| **Operating System** | Windows 11 Home / Pro (10.0.26300) | PowerShell / Windows sub-environment |
| **CPU Architecture** | 12 Logical Cores (8 Physical Cores) | Multi-processing DataLoader limit: `num_workers=0` or `2` |
| **System Memory (RAM)** | 15.71 GB Total (1.08 GB Currently Available) | High memory pressure; avoid caching full uncompressed tensors in RAM |
| **GPU Accelerator** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | CUDA 12.4 enabled; `torch.cuda.is_available() = True` |
| **GPU Dedicated VRAM** | 6.00 GB GDDR6 | Batch size constrained to $N \in [16, 32]$ with FP16 Automatic Mixed Precision (`torch.amp`) |
| **PyTorch Runtime** | PyTorch 2.6.0+cu124 | Full cuBLAS / cuDNN backend acceleration active |

---

## 2. Model Architecture Anatomy

The LX-DFD architecture ([`src/models/lxfd_model.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/src/models/lxfd_model.py)) is a synchronized multi-modal spatial-frequency forensic network:

```
                                  [ Input Facial Crop (224 x 224 x 3) ]
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
      [ Spatial Branch (B4) ]                                           [ Frequency Branch (2D DCT) ]
      EfficientNet-B4 Backbone                                          Block-wise 2D Discrete Cosine Transform
      Extracts spatial boundaries, blending                             Maps spatial energy to spectral frequency grid
      edges, and facial micro-texture cues                              4-layer Conv2D + BatchNorm + ReLU stream
      Embedding Dim: 1792                                               Embedding Dim: 256
                   │                                                                 │
                   └────────────────────────────────┬────────────────────────────────┘
                                                    ▼
                                   [ Dynamic Attention Fusion Head ]
                                   Linear projection -> Softmax weights (w_s, w_f)
                                   Enforced Attention Floor: min_floor = 0.25
                                   w_s in [0.25, 0.75], w_f in [0.25, 0.75]
                                   Fused Logit: z = w_s * z_s + w_f * z_f
                                                    │
                                                    ▼
                                       [ Single Logit Classification ]
                                       BCEWithLogitsLoss / Sigmoid Output
```

- **Spatial Stream**: `EfficientNet-B4` initialized with pretrained weights, pooled to 1792 dimensions.
- **Frequency Stream**: `FrequencyStream` converting RGB to 2D DCT spectral magnitudes followed by 4 convolutional blocks (16 $\to$ 32 $\to$ 64 $\to$ 128 channels) and adaptive average pooling to 256 dimensions.
- **Attention Fusion**: `AttentionFusion` (`src/models/fusion.py`) learning sample-adaptive weights $w_s$ and $w_f$ with a strict safety floor $w \ge 0.25$ to prevent single-branch dominance.
- **Parameters**: 20,248,367 total parameters (~77.2 MB in FP32).

---

## 3. Production Checkpoint Verification

- **Current Production Path**: `checkpoints/lxfd_phase10_modern_generators_best.pt`
- **Authenticated Hash (SHA-256)**: `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05`
- **Checkpoint Epoch**: 12 (Trained across 15 epochs in Phase 10)
- **Validation ROC-AUC**: 0.978689
- **Validation Check**: Confirmed that the running backend application (`app/backend/main.py`), explainability engine (`/explain`), and evaluation suites load this identical checkpoint file.

---

## 4. Current Training Configuration & Preprocessing Pipeline

### 4.1 Training Hyperparameters (Phase 10 Baseline)
- **Optimizer**: `AdamW(lr=5e-5, weight_decay=1e-4, betas=(0.9, 0.999))`
- **Learning Rate Schedule**: `CosineAnnealingLR(T_max=15, eta_min=1e-6)`
- **Loss Function**: Binary Cross Entropy with Logits (`nn.BCEWithLogitsLoss()`)
- **Precision**: FP16 Mixed Precision via `torch.amp.GradScaler('cuda')`
- **Early Stopping**: 5 epochs patience on validation ROC-AUC.

### 4.2 Preprocessing Pipeline Parity
- **Face Localization**: OpenCV YuNet ONNX (`yunet.onnx`), input resized dynamically to $(w, h)$, confidence threshold 0.50, NMS 0.30.
- **Context Margin**: Expansion margin $\text{pad} = 0.50 \times \text{bbox\_dim}$ on all 4 borders to preserve hair, jawline, and background blending boundaries.
- **Input Resolution**: Bicubic interpolation to $224 \times 224 \times 3$.
- **Tensor Normalization**: Standard ImageNet mean $\mu = [0.485, 0.456, 0.406]$ and std $\sigma = [0.229, 0.224, 0.225]$.
- **Inference vs. Training Consistency**: **Verified Identical**. The inference transform in `src/models/inference.py` uses the exact same bicubic interpolation and normalization parameters as evaluation datasets.

---

## 5. Dataset Inventory & Split Imbalance Audit

An audit of the active training manifest (`data/splits_phase10/train.csv`, `val.csv`, `test_*.csv`) revealed the exact mathematical reason why Gemini faces misclassified:

| Split | Total Samples | Real Faces | StyleGAN2 (Fake) | InsightFace (Fake) | SD 1.5 (Fake) | SD Inpaint (Fake) | Gemini Imagen 3 (Fake) | Gemini % in Split |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 1,165 | 580 | 300 | 140 | 140 | 0 (Held-out) | **5** | **0.43%** |
| **Validation** | 242 | 120 | 60 | 30 | 30 | 0 (Held-out) | **2** | **0.83%** |
| **Test (In-Domain)** | 240 | 120 | 60 | 30 | 30 | 0 | 0 | 0.00% |
| **Test (Inpaint)** | 200 | 0 | 0 | 0 | 0 | 200 | 0 | 0.00% |
| **Test (Gemini)** | 17 | 0 | 0 | 0 | 0 | 0 | **17** | **100.0%** |

### Verified Dataset Diagnosis
In the existing training dataset of 1,165 images, **only 5 images (0.43%) were from Google Gemini**, while 580 were Real and 580 were legacy GAN/diffusion fakes. Consequently:
1. The loss contribution from Gemini samples was virtually zero ($\approx \frac{5}{1165} \times \mathcal{L}$).
2. The spatial branch learned to associate the absence of StyleGAN checkerboards and GAN boundary seams with the "Real" class.
3. When evaluated on modern Gemini Imagen 3 portraits (which have clean photorealistic skin without GAN seams), the model leaned heavily towards Real.

---

## 6. Training Readiness Verdict

1. **Hardware Ready**: RTX 3050 (6GB VRAM) is fully verified with PyTorch CUDA.
2. **Architecture Verified**: Dual-branch `LXDFDModel` is fully intact and reproducible.
3. **Primary Action Required**: Expand and balance the training manifest by incorporating verified Gemini face samples and diverse genuine human controls, while strictly preserving dedicated, generator-isolated held-out evaluation sets.
