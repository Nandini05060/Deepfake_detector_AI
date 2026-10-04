# Phase 3 Final Research Recommendations & Model Selection Report

**Date**: October 4, 2026  
**Author**: Machine Learning Engineer & Deepfake Detection Researcher  
**Recommended Production Checkpoint**: [`checkpoints/lxfd_attention_robustness_v2_best.pt`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/checkpoints/lxfd_attention_robustness_v2_best.pt)  
**Baseline Reference Checkpoint**: [`checkpoints/lxfd_attention_standard_best.pt`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/checkpoints/lxfd_attention_standard_best.pt)  

---

## 1. Executive Summary & Model Selection Verdict

Following the execution of Phase 3 experiments on the held-out, leak-free test split (`data/splits_v2/test.csv`, $N=1,956$), **LX-DFD with Robustness-Aware Augmentation (`robustness_v2`)** is selected as the superior, production-grade model.

### Key Decision Matrix

| Dimension | Baseline LX-DFD (`standard`) | LX-DFD (`robustness_v2`) | Impact / Verdict |
|---|:---:|:---:|---|
| **Gaussian Noise ($\sigma=0.05$) AUC** | $0.6274$ | **$0.9725$** | **$+0.3451$ (+55.0% relative boost)** |
| **Gaussian Noise ($\sigma=0.05$) Accuracy** | $48.26\%$ | **$89.83\%$** | **$+41.57$ percentage points** |
| **JPEG Quality 30 AUC** | $0.9323$ | **$0.9822$** | **$+0.0499$ gain (93.0% accuracy)** |
| **Gaussian Blur ($r=1.5$) AUC** | $0.9677$ | **$0.9826$** | **$+0.0149$ gain (91.56% accuracy)** |
| **Mean AUC Drop Across Corruptions** | $0.0815$ | **$0.0353$** | **$56.7\%$ reduction in degradation** |
| **Composite Robustness Score** | $0.9185$ | **$0.9647$** | **$+0.0462$ improvement** |
| **Clean In-Domain ROC-AUC** | $0.9998$ | **$0.9943$** | Preserved ($>0.994$) |
| **Inference Latency** | $4.58\text{ ms}$ | **$4.75\text{ ms}$** | Real-time capable ($>210\text{ fps}$) |

---

## 2. Why `robustness_v2` Solved the Failure Modes

1. **Frequency Invariance Learning**:
   - Baseline models trained only on clean imagery learned brittle high-frequency spectral spikes characteristic of uncompressed GAN upsampling. When mild noise was injected, the entire high-frequency spectrum shifted, completely blinding the frequency branch.
   - Stochastic injection of Gaussian noise ($\sigma \in [0.01, 0.04]$), JPEG compression ($q \in [60, 95]$), and blur during training forced the 4-layer Frequency CNN to learn robust mid-frequency harmonic patterns rather than noise-vulnerable outer frequencies.
2. **Balanced Cross-Modal Attention**:
   - Under heavy corruption, the cross-modal attention mechanism dynamically shifts feature reliance toward spatial semantics when the frequency spectrum exhibits high uncertainty, preventing classification collapse.

---

## 3. Unseen-Generator Status & Real-World Generalization

- **Honest Scientific Disclosure**: True cross-generator generalization on physical unseen models (e.g. Midjourney v6, Stable Diffusion XL, FLUX.1) remains **unverified** due to the absence of external generator datasets bundled in the repository.
- **Repeatable Protocol Established**: The evaluation harness in [`scripts/evaluate_unseen_generators.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/scripts/evaluate_unseen_generators.py) is ready to evaluate multi-generator directories without training leakage.

---

## 4. Next Actionable Steps

1. **Deploy `lxfd_attention_robustness_v2_best.pt`** as the default inference model in `app/backend/main.py`.
2. **Collect Multi-Generator External Test Data**: Add 200–500 images each of StyleGAN3, Midjourney, and Diffusion faces to `data/unseen_generator_test/` to obtain verified external metrics.
3. **Keep Architecture Lightweight**: Do not introduce heavy Vision Transformers or FFT rewrites, as the dual-branch EfficientNet + DCT CNN achieves 96.5% robustness score with sub-5ms latency.
