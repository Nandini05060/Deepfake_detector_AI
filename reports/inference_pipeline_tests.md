# LX-DFD Inference Pipeline & Explainability Regression Tests
**Execution Timestamp**: 2026-10-22
**PyTorch Version**: 2.6.0+cu124
**Target Model**: `checkpoints/lxfd_multigen_diverse_best.pt`

## 1. Checkpoint & Architecture Loading
- [x] Successfully loaded `lxfd_multigen_diverse_best.pt`
- [x] Temperature Scaling Parameter: T = 1.0413
- [x] Minimum Attention Floor: min_floor = 0.05 active in `SpatialFrequencyFusion`

## 2. End-to-End Prediction & Branch Output Integrity

| Sample Name | Prediction | Fake Prob | Real Prob | Conf Level | Emp Reliability | w_s (Spatial) | w_f (Freq) | Spatial Prob | Freq Prob | Discrepancy |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini Imagen 3 (Full $2816\times 1536$) | **REAL** | 17.3% | 82.7% | MODERATE | MODERATE (Artifact Absence vs True Authenticity) | 0.9497 | 0.0503 | 16.8% | 10.0% | 0.0682 |
| Gemini Imagen 3 (YuNet Margin 0.45 Portrait Crop) | **REAL** | 0.1% | 99.9% | HIGH (In-Distribution) | MODERATE (Subject to Generator Scope) | 0.9497 | 0.0503 | 0.1% | 6.5% | 0.0639 |

## 3. Tensor & Numerical Sanity Checks

- [x] **Output Logit NaN / Inf check**: PASSED
- [x] **Spatial Feature F_s NaN / Inf check**: PASSED
- [x] **Frequency Feature F_f NaN / Inf check**: PASSED
- [x] **DCT Map NaN / Inf check**: PASSED
- [x] **Zero Feature check (F_s norm > 0)**: PASSED
- [x] **Zero Feature check (F_f norm > 0)**: PASSED
- [x] **Attention Weights Sum to 1.0 (w_s + w_f = 1.0)**: PASSED
- [x] **Attention Floor Active (w_s >= 0.05)**: PASSED
- [x] **Attention Floor Active (w_f >= 0.05 - Frequency Branch Un-Silenced)**: PASSED

## 4. Explainability Studio Tests (Grad-CAM & DCT Spectrum)

- [x] **Grad-CAM Fake Class Target Heatmap (Valid [0, 255] uint8)**: PASSED
- [x] **Grad-CAM Real Class Target Heatmap (Valid [0, 255] uint8)**: PASSED
- [x] **Grad-CAM Auto Target Heatmap (Targets predicted direction)**: PASSED
- [x] **DCT Spectrum Map (Valid 224x224 Colormap)**: PASSED
- [x] **High-Frequency Filter Map (Valid 224x224 Colormap)**: PASSED
- [x] **Forensic Disclaimers & Scope Warnings Included**: PASSED

## 5. Out-of-Distribution Generator Advisory & Confidence Policy

- **Empirical Reliability Distinction**: Softmax probabilities $\ge 0.90$ are no longer automatically designated as 'HIGH' confidence for external inputs. Instead, they are explicitly qualified as `MODERATE (Subject to Generator Scope)` or `MODERATE (Artifact Absence vs True Authenticity)`.
- **Scope Advisory Guarantee**: When classified as REAL, an advisory is delivered informing users that the detector has been validated across StyleGAN2, ProGAN, Stable Diffusion v1.5, and InsightFace, and that absence of known artifacts does not prove positive authenticity against unvalidated engines (Gemini Imagen 3, Midjourney v6, FLUX.1).
- **Branch Discrepancy Alert**: When spatial and frequency probabilities diverge by $>10\%$, the result is flagged with an uncertainty warning to alert forensic analysts to possible out-of-distribution inputs.
