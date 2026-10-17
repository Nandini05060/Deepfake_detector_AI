# Phase 4: Standardized External Evaluation and Cross-Generator Generalization Protocol

**Author**: Senior Deepfake Detection Researcher  
**Project**: LX-DFD  
**Date**: October 2026  
**Status**: Formal Evaluation Protocol Specification  

---

## 1. Scientific Principles & Purpose

In deepfake detection research, models frequently suffer from "generator overfitting"—memorizing idiosyncratic synthesis artifacts (such as transposed convolution checkerboard frequencies or color balance shifts) specific to a single generative architecture. 

To scientifically evaluate genuine generalization without misleading aggregate statistics, this protocol defines a rigorous, repeatable evaluation methodology adhering to four fundamental pillars:
1. **Strict Partitioning**: Complete separation between in-domain test data and external cross-dataset benchmarks.
2. **Disaggregated Generator Reporting**: Disallowing indiscriminate pooling of diverse generators into a single undifferentiated average.
3. **Decoupled Robustness & Domain Transfer**: Separating clean domain shifts from image transmission corruptions.
4. **Predeclared Decision Thresholds**: Prohibiting post-hoc threshold searching on external evaluation splits.

---

## 2. Evaluation Protocols Specification

```
                                  EVALUATION SUITE
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
  [Protocol A]                     [Protocol B]                     [Protocol C]
In-Domain Benchmark           Cross-Dataset Transfer        Unseen-Generator Benchmark
Held-out StyleGAN/FFHQ        Entirely Novel Source          Disaggregated by Model:
data/splits_v2/test.csv       (e.g. Celeb-DF, FF++)          StyleGAN3, Midjourney, FLUX
(N=1,956, Zero Leakage)       (Independent Real/Fake)        (Macro & Weighted Averages)
```

### Protocol A: In-Domain Test Split Evaluation
- **Dataset**: `data/splits_v2/test.csv` ($N=1,956$, $1,041$ FAKE, $915$ REAL).
- **Purpose**: Establishes the performance ceiling and serves as the baseline reference point for in-distribution generalization.
- **Condition**: Clean, unmodified images, preprocessed to 224x224 RGB with standard normalization.

### Protocol B: Zero-Shot Cross-Dataset Transfer Evaluation
- **Dataset**: Independent external datasets (e.g. FaceForensics++, Celeb-DF v2).
- **Purpose**: Evaluates model performance across changes in camera sensors, demographics, subject poses, lighting conditions, and video compression pipelines.
- **Metric**: 
  $$\text{Generalization Gap} = \text{ROC-AUC}_{\text{in-domain}} - \text{ROC-AUC}_{\text{cross-dataset}}$$

### Protocol C: Disaggregated Unseen-Generator Benchmark
- **Dataset Structure**: `data/unseen_generator_test/` organized into:
  - `real/`: Authentic real photographs ($N_{\text{real}}$).
  - `<GeneratorName>/`: Synthetic portraits generated exclusively by that model ($N_{\text{fake}}$).
- **Execution Rule**: Each generator family $G_k$ is paired with the authentic real set to compute independent, per-generator discrimination metrics:
  $$\text{ROC-AUC}(G_k), \quad \text{PR-AUC}(G_k), \quad \text{F1}(G_k), \quad \text{Accuracy}(G_k)$$
- **Aggregate Metrics**:
  - **Macro-Average**: Unweighted mean across all $K$ evaluated generators:
    $$\text{Metric}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^K \text{Metric}(G_k)$$
    *Prevents generators with large sample counts from skewing overall findings.*
  - **Weighted-Average**: Sample-weighted mean across all evaluated generators:
    $$\text{Metric}_{\text{weighted}} = \frac{\sum_{k=1}^K N_k \cdot \text{Metric}(G_k)}{\sum_{k=1}^K N_k}$$

### Protocol D: Perturbation & Corruption Disentanglement
- When evaluating external datasets, clean external performance must be measured **before** evaluating corrupted variants.
- External images are then subjected to standard corruption benchmarks:
  - Moderate JPEG compression ($q = 75$)
  - Mild Gaussian noise ($\sigma = 0.02$)
  - Mild Gaussian blur ($r = 1.0$)
- This disentangles whether external failure is driven by generator domain shift or transmission sensitivity.

---

## 3. Decision Threshold & Calibration Policy

### 3.1 Predeclared Threshold Policy
- **Strict Rule**: The decision threshold $\tau$ **MUST NOT** be tuned on the external evaluation data. Finding a post-hoc threshold $\tau^*$ that maximizes F1 on the test set is a form of target leakage that invalidates real-world deployment claims.
- **Standard Threshold**: $\tau = 0.50$ (standard probabilistic threshold).
- **Validation-Tuned Threshold**: Optimal threshold $\tau_{\text{val}} = 0.50$ fit strictly on the validation partition (`data/splits_v2/val.csv`).

### 3.2 Probabilistic Calibration & Uncertainty Quantification
- Models output raw logit $z$. Raw predicted probability is $p = \sigma(z)$.
- **Temperature Scaling**: Uses temperature parameter $T = 1.0414$ derived on the validation split:
  $$p_{\text{cal}} = \sigma\left(\frac{z}{T}\right)$$
- Calibration performance is reported via:
  - **Brier Score**: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$
  - **Expected Calibration Error (ECE)**: Using 10 equal-width bins in $[0, 1]$.
  - **Uncertainty Rejection Band**: Samples with $p_{\text{cal}} \in [0.35, 0.65]$ are flagged as `UNCERTAIN` for human review.

---

## 4. Standardized Preprocessing Specification

All candidate models must be evaluated using the exact same preprocessing pipeline:
1. **Input Representation**: 3-channel RGB image.
2. **Face Crop Alignment**: If whole-image or video frames are provided, faces must be detected and cropped using YuNet (`yunet.onnx`) with a standard bounding-box expansion ratio of $1.25\times$ to capture chin and forehead boundary artifacts.
3. **Resizing**: Bilinear interpolation resizing to $224 \times 224$ pixels.
4. **Tensor Conversion & Normalization**:
   - Scaled to $[0.0, 1.0]$.
   - ImageNet mean normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
5. **Batch Size**: 32 (deterministic execution, shuffle=False).

---

## 5. Metrics Specification

For every evaluation run, the following metrics must be recorded and exported:

| Metric | Formula / Definition | Academic Purpose |
|---|---|---|
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve | Threshold-independent discrimination capability across all operating points |
| **PR-AUC** | Area under the Precision-Recall curve | Discriminative quality under class imbalance |
| **Accuracy** | $(TP + TN) / (TP + TN + FP + FN)$ | Overall classification correctness at $\tau=0.5$ |
| **Precision** | $TP / (TP + FP)$ | Reliability of positive (fake) detections |
| **Recall** | $TP / (TP + FN)$ | Sensitivity in catching deepfakes |
| **F1-Score** | $2 \cdot (P \cdot R) / (P + R)$ | Harmonic mean of precision and recall |
| **Brier Score** | Mean squared difference between probabilities and binary labels | Calibration and probabilistic accuracy |
| **ECE** | $\sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ | Confidence reliability in deployment |
| **Generalization Gap** | $\text{AUC}_{\text{in-domain}} - \text{AUC}_{\text{external}}$ | Quantitative degradation when transitioning to novel domains |
