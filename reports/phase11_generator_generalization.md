# Phase 11 Cross-Generator Generalization & Statistical Benchmark
## Empirical Performance Profile Across Modern AI Generators and Authentic Negative Controls

**Document Version:** 1.0.0  
**Audit Execution Date:** October 2026  
**Document Identifier:** `reports/phase11_generator_generalization.md`  
**Associated Data:** `reports/phase11_test_metrics.json` and `data/manifests/phase11_generator_manifest.csv` ($N=457$ locked test images)  

---

## 1. Executive Summary & Scientific Scope

Claims of deepfake detection generalization often fail when evaluated on generators absent from the training distribution. In Phase 11, we evaluated the production system (`checkpoints/lxfd_phase10_modern_generators_best.pt`, SHA-256 `13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05`) across $N=457$ strictly locked test samples encompassing five generative architectures and authentic photographic negative controls.

### Headline Findings:
1. **Familiar vs. Unseen Generalization Gap:** While the system achieves near-ceiling detection on familiar GAN architectures (**StyleGAN2: 95.0% recall, ROC-AUC 0.9846**), detection drops sharply on unseen modern latent diffusion architectures (**Stable Diffusion Inpainting: 28.0% recall, Gemini Imagen 3: 11.8% recall** under default $0.50$ crop).
2. **Macro-Average Performance:** Across all five synthetic generator families, the macro-average ROC-AUC is **0.8694**, macro-average recall is **50.28%**, and macro-average F1-score is **0.5147**.
3. **Authentic Photographic Specificity:** The system maintains high specificity on authentic human controls (**89.17% Specificity, FPR = 10.83%**, 95% CI: $[82.34\%, 93.56\%]$).
4. **Resolution Downsampling Restores Gemini Recall:** As demonstrated in the diagnostic suite, downsampling Gemini portraits to $112 \times 112$ restores detection recall to **88.2%** ($\bar{P}=0.8061$).

---

## 2. Generator-Specific Benchmark Table

All metrics below are computed on locked test splits against authentic photographic human negative controls ($N=120$). 95% Confidence Intervals are calculated via the Wilson score interval method.

| Generative Cohort | Role / Category | Sample Count ($N$) | ROC-AUC | PR-AUC | Recall (Sens.) | 95% CI (Recall) | Specificity | Precision | F1-Score | Mean $P(\text{Fake})$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **StyleGAN2** | GAN Synthetic | 60 | **0.9846** | **0.9759** | **95.0%** (57/60) | $[86.30\%, 98.29\%]$ | 89.17% | 81.43% | **0.8769** | 0.9152 |
| **Stable Diffusion v1.5** | Latent Diffusion | 30 | **0.9069** | **0.7755** | **63.3%** (19/30) | $[45.51\%, 78.13\%]$ | 89.17% | 59.38% | **0.6129** | 0.6709 |
| **InsightFace Swap** | Identity Swapping | 30 | **0.8950** | **0.5804** | **53.3%** (16/30) | $[36.14\%, 69.77\%]$ | 89.17% | 55.17% | **0.5424** | 0.5252 |
| **SD Inpainting** | Unseen Inpainting | 200 | **0.8102** | **0.8295** | **28.0%** (56/200) | $[22.24\%, 34.59\%]$ | 89.17% | 81.16% | **0.4164** | 0.3556 |
| **Google Gemini / Imagen 3** | Unseen Diffusion | 17 | **0.7505** | **0.3161** | **11.8%** (2/17) | $[3.29\%, 34.34\%]$ | 89.17% | 13.33% | **0.1250** | 0.2621 |
| **Authentic Real Controls** | Real Baseline | 120 | — | — | — | — | **89.17%** | — | — | 0.1344 |
| **MACRO-AVERAGE** | All 5 Synthetic Cohorts | **337** | **0.8694** | **0.6955** | **50.28%** | — | **89.17%** | **58.09%** | **0.5147** | — |

---

## 3. Confusion Matrices Against Authentic Controls ($N=120$)

### 3.1 StyleGAN2 ($N=60$ Fake, $N=120$ Real)
```
                Predicted REAL    Predicted FAKE
Actual REAL           107               13          (Specificity = 89.17%)
Actual FAKE             3               57          (Recall = 95.00%)
```

### 3.2 Stable Diffusion v1.5 ($N=30$ Fake, $N=120$ Real)
```
                Predicted REAL    Predicted FAKE
Actual REAL           107               13          (Specificity = 89.17%)
Actual FAKE            11               19          (Recall = 63.33%)
```

### 3.3 InsightFace Swap ($N=30$ Fake, $N=120$ Real)
```
                Predicted REAL    Predicted FAKE
Actual REAL           107               13          (Specificity = 89.17%)
Actual FAKE            14               16          (Recall = 53.33%)
```

### 3.4 Stable Diffusion Inpainting ($N=200$ Fake, $N=120$ Real)
```
                Predicted REAL    Predicted FAKE
Actual REAL           107               13          (Specificity = 89.17%)
Actual FAKE           144               56          (Recall = 28.00%)
```

### 3.5 Google Gemini / Imagen 3 ($N=17$ Fake, $N=120$ Real)
```
                Predicted REAL    Predicted FAKE
Actual REAL           107               13          (Specificity = 89.17%)
Actual FAKE            15                2          (Recall = 11.76%)
```

---

## 4. Scientific Discussion & Architectural Limitations

### 1. The Generalization Discrepancy
The system exhibits a pronounced hierarchy of detection efficacy:
$$\text{StyleGAN2 (95.0\%)} > \text{SD 1.5 (63.3\%)} > \text{InsightFace (53.3\%)} > \text{SD Inpainting (28.0\%)} > \text{Gemini Imagen 3 (11.8\%)}$$
* Models that produce direct synthesis with transposed-convolution checkerboard artifacts (StyleGAN2) are detected reliably.
* Modern latent diffusion models operating at native $1024 \times 1024$ resolution with learned variational autoencoder decoders (Imagen 3) synthesize naturalistic high-frequency noise distributions that closely mimic authentic camera sensors.

### 2. Equal-Weight Averaging vs. Attention Over-Suppression
As discovered in the ablation study, switching from the learned attention weighting ($w_s \approx 0.94, w_f \approx 0.06$) to an Equal-Weight average ($0.5/0.5$) dramatically elevates recall on boundary-manipulated generators:
* SD Inpainting recall increases from **28.0% to 62.5%** (+34.5%).
* InsightFace Swap accuracy increases from **53.3% to 73.3%** (+20.0%).
* Overall test set accuracy rises from **56.24% to 63.46%**, and F1 rises from **0.6000 to 0.6890**.

### 3. Uncertainty Gating Prevents Deceptive False Negatives
Crucially, **100%** of Gemini test portraits trigger the automated uncertainty rule ($|w_s - w_f| > 0.15$ or $0.40 \le P \le 0.60$). In production, the system does not output a false "CERTIFIED REAL" verdict; it outputs `INCONCLUSIVE — High Branch Discrepancy / Out-of-Distribution Warning`.
