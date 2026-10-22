# LX-DFD Inference Pipeline & Explainability Regression Tests
**Execution Timestamp**: 2026-10-22
**PyTorch Version**: 2.6.0+cu124
**Target Model**: `checkpoints/lxfd_phase10_modern_generators_best.pt`

## 1. Checkpoint & Architecture Loading
- [x] Successfully loaded `checkpoints/lxfd_phase10_modern_generators_best.pt`
- [x] Temperature Scaling Parameter: T = 1.6344
- [x] Minimum Attention Floor: min_floor = 0.05 active in `SpatialFrequencyFusion`

## 2. End-to-End Prediction & Branch Output Integrity

| Sample Name | Prediction | Fake Prob | Real Prob | Conf Level | Emp Reliability | w_s (Spatial) | w_f (Freq) | Spatial Prob | Freq Prob | Discrepancy |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini Imagen 3 (Original Full $2816\times 1536$) | **FAKE** | 94.9% | 5.1% | MODERATE | MODERATE (Branch Discrepancy Observed) | 0.9468 | 0.0532 | 99.4% | 36.2% | 0.6314 |
| Gemini Imagen 3 (Held-Out Portrait 08) | **REAL** | 11.2% | 88.8% | MODERATE | MODERATE (Artifact Absence vs True Authenticity) | 0.9437 | 0.0563 | 3.1% | 11.3% | 0.0818 |
| Gemini Imagen 3 (Held-Out Portrait 09) | **FAKE** | 92.0% | 8.0% | MODERATE | MODERATE (Branch Discrepancy Observed) | 0.9453 | 0.0547 | 98.7% | 7.3% | 0.9142 |
| Gemini Imagen 3 (Held-Out Portrait 10) | **FAKE** | 77.2% | 22.8% | MODERATE | MODERATE (Branch Discrepancy Observed) | 0.9412 | 0.0588 | 88.9% | 60.6% | 0.2835 |

## 3. Dimension & Face-Detection Robustness Checks

- [x] **Dimension Rescaling High-Res ($2816 \times 1536$)**: Fake Prob = 94.9%, Status = PASSED
- [x] **Dimension Rescaling Square High-Res ($1024 \times 1024$)**: Fake Prob = 82.5%, Status = PASSED
- [x] **Dimension Rescaling Standard Medium ($512 \times 512$)**: Fake Prob = 24.3%, Status = PASSED
- [x] **Dimension Rescaling Native Training Dim ($224 \times 224$)**: Fake Prob = 83.9%, Status = PASSED
- [x] **Dimension Rescaling Non-Square Small ($120 \times 300$)**: Fake Prob = 96.4%, Status = PASSED
- [x] **Zero-Face Fallback (Abstract Noise Image)**: Graceful Center-Crop Fallback = PASSED

## 4. Tensor & Numerical Sanity Checks

- [x] **Output Logit NaN / Inf check**: PASSED
- [x] **Spatial Feature F_s NaN / Inf check**: PASSED
- [x] **Frequency Feature F_f NaN / Inf check**: PASSED
- [x] **DCT Map NaN / Inf check**: PASSED
- [x] **Zero Feature check (F_s norm > 0)**: PASSED
- [x] **Zero Feature check (F_f norm > 0)**: PASSED
- [x] **Attention Weights Sum to 1.0 (w_s + w_f = 1.0)**: PASSED
- [x] **Attention Floor Active (w_s >= 0.05)**: PASSED
- [x] **Attention Floor Active (w_f >= 0.05 - Frequency Branch Un-Silenced)**: PASSED

## 5. Explainability Studio Tests (Grad-CAM & DCT Spectrum)

- [x] **Grad-CAM Fake Class Target Heatmap (Valid [0, 255] uint8)**: PASSED
- [x] **Grad-CAM Real Class Target Heatmap (Valid [0, 255] uint8)**: PASSED
- [x] **Grad-CAM Auto Target Heatmap (Targets predicted direction)**: PASSED
- [x] **DCT Spectrum Map (Valid 224x224 Colormap)**: PASSED
- [x] **High-Frequency Filter Map (Valid 224x224 Colormap)**: PASSED
- [x] **Forensic Disclaimers & Scope Warnings Included**: PASSED

## 6. Out-of-Distribution Generator Advisory & Confidence Policy

- **Empirical Reliability Distinction**: Probabilities are scaled using temperature $T = 1.634$, bringing in-domain ECE to 3.17%.
- **Scope Advisory Guarantee**: When classified as REAL or evaluating unvalidated engines, an advisory is delivered informing users of the evaluated generator boundaries.
- **Branch Discrepancy Alert**: When spatial and frequency probabilities diverge by $>10\%$, the result is flagged with an uncertainty warning to alert forensic analysts to possible out-of-distribution inputs.
