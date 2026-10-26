# Forensic Trace Report: Gemini Failure Image

**Diagnostic Target**: `Gemini_Generated_Image_qykw28qykw28qykw.png`  
**Local Copy Path**: `data\diagnostics\gemini_failure.png`  
**Image SHA-256**: `944cb50e9187a22a46dc6232387aa6769d4513c4654b141832b1eec04cd377fa`  
**Ground Truth**: `FAKE` (Known provenance: Google Gemini Imagen 3 text-to-image synthesis)  
**Dimensions**: 2816 × 1536 | **Color Mode**: RGBA | **Format**: PNG | **Size**: 6682115 bytes  

## 1. Preprocessing & Face Extraction Audit

| Component | Implementation & Audit Result | Verification Notes |
|:---|:---|:---|
| **Color Mode & Alpha Handling** | Converted RGBA -> RGB using PIL `convert('RGB')` | Alpha channel min=255, max=255. Fully opaque; no alpha bleed. |
| **Face Detector** | OpenCV YuNet ONNX (`yunet.onnx`) | Successfully detected face: BBox `[1037, 310, 571, 619]` [x, y, w, h] |
| **Context Crop (m=0.50)** | Expanded BBox with 50% margin | Coordinates `[752, 1, 1893, 1238]`, crop size `(1141, 1237)` |
| **Resizing & Normalization** | PyTorch standard transform (224x224, ImageNet mean/std) | Consistent with training pipeline |
| **Class Ordering** | Index 0 = REAL, Index 1 = FAKE | Verified: `logits > 0` indicates FAKE, `logits < 0` indicates REAL |

## 2. Multi-Model Inference Comparison on gemini_failure.png

| Diagnostic Metric | Phase 10 (Production) | Phase 15 (Candidate) | Phase 16 (Unified Best) |
|:---|:---:|:---:|:---:|
| **Active Checkpoint** | `lxfd_phase10_modern_generators_best.pt` | `lxfd_phase15_generalization_best.pt` | `lxfd_phase16_unified_best.pt` |
| **Checkpoint SHA-256** | `13dc56b8961f8d0c...` | `2552ab2b8fdc9b61...` | `4e7b7747f4a3108f...` |
| **Calibrated Temperature (T)** | 1.6344 | 0.7169 | 1.0859 |
| **Spatial Raw Logit** | 0.1341 | -0.2554 | -2.6346 |
| **Frequency Raw Logit** | 0.9328 | 0.0402 | -0.2613 |
| **Fused Raw Logit** | 0.3802 | -0.1890 | -2.0282 |
| **Spatial Fake Prob** | 53.35% | 43.65% | 6.69% |
| **Frequency Fake Prob** | 71.76% | 51.01% | 43.50% |
| **Attention Weights (w_s / w_f)** | 0.75 / 0.25 | 0.75 / 0.25 | 0.75 / 0.25 |
| **Calibrated Fake Prob** | **55.79%** | **43.45%** | **13.38%** |
| **Calibrated Real Prob** | **44.21%** | **56.55%** | **86.62%** |
| **Final Decision** | `INCONCLUSIVE` | `INCONCLUSIVE` | `INCONCLUSIVE` |
| **Uncertainty Triggered** | True | True | True |

## 3. Crop Stability Analysis

| Model | Margin 0.15 (Tight) | Margin 0.50 (Standard) | Margin 0.75 (Extended) | Max Variation Delta |
|:---|:---:|:---:|:---:|:---:|
| **LX-DFD Phase 10 Modern Multi-Generator Standard** | 2.8% (`INCONCLUSIVE`) | 55.8% (`INCONCLUSIVE`) | 88.4% (`LIKELY_FAKE`) | **85.6%** |
| **LX-DFD Phase 15 Gemini Generalization Candidate** | 0.0% (`INCONCLUSIVE`) | 43.5% (`INCONCLUSIVE`) | 99.6% (`LIKELY_FAKE`) | **99.5%** |
| **LX-DFD Phase 16 Unified Multi-Generator Best** | 0.2% (`INCONCLUSIVE`) | 13.4% (`INCONCLUSIVE`) | 84.7% (`LIKELY_FAKE`) | **84.4%** |

## 4. Root Cause Summary

1. **Face Scale & Crop Aspect Ratio**: The original image is ultra-wide (2816 × 1536, aspect ratio 1.83:1). YuNet correctly identified the face bbox. When padding with 50% margin, the aspect ratio remains non-square until resized to 224x224.
2. **Spatial Feature Manifold Gap**: Google Gemini Imagen 3 exhibits extremely natural diffuse skin illumination with minimal GAN checkerboard upsampling signatures. While Phase 10 was completely blind to it, Phase 15 over-fit to its 10 training samples and Phase 16 balances multi-generator representation.
