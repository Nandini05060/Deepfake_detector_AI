# Phase 3 Unseen-Generator Generalization Benchmark Report

**Evaluation Status**: **NOT YET VERIFIED (Awaiting External Evaluation Data)**  
**Benchmark Script**: [`scripts/evaluate_unseen_generators.py`](file:///c:/Users/nandi/Desktop/DEEPFAKE%20DETECTION/scripts/evaluate_unseen_generators.py)  
**Standardized Test Directory**: `data/unseen_generator_test/`  

---

## 1. Audit Finding & Statement of Integrity

In accordance with strict scientific integrity guidelines:
- **No Ground-Truth Generator Metadata in Current Dataset**: The primary Kaggle deepfake dataset (`Final Dataset`) contains binary `Real/` and `Fake/` image directories without ground-truth generator annotations (e.g., distinguishing StyleGAN2, ProGAN, StarGAN, or Diffusion synthesizers).
- **Zero External Test Datasets Bundled**: No physical external generator images (e.g. CelebA-HQ, FFHQ, StyleGAN3, Midjourney, Stable Diffusion) were packaged inside the local repository.
- **Explicit Classification**: Cross-generator generalization is **NOT YET EMPIRICALLY VERIFIED** on physical external datasets. Any historical 0.895 citation in early template summaries was a preliminary placeholder.

---

## 2. Standardized Unseen-Generator Evaluation Harness

A fully automated, leak-free evaluation interface is established in `scripts/evaluate_unseen_generators.py`.

### Required Directory Structure
To execute an unseen-generator benchmark, test images should be organized into:
```
data/unseen_generator_test/
├── real/                      # Ground-truth authentic real photographs
│   ├── real_0001.jpg
│   └── ...
├── StyleGAN2/                 # Synthesized by StyleGAN2 (absent from training)
│   ├── fake_0001.jpg
│   └── ...
├── StyleGAN3/                 # Synthesized by StyleGAN3
│   ├── fake_0001.jpg
│   └── ...
├── Midjourney_v6/             # Synthesized by Midjourney
│   ├── fake_0001.jpg
│   └── ...
└── FLUX_1/                    # Synthesized by FLUX Diffusion
    ├── fake_0001.jpg
    └── ...
```

---

## 3. Automated Benchmark Execution Command

When images are placed into the respective subdirectories, execute:
```bash
python scripts/evaluate_unseen_generators.py \
    --checkpoint checkpoints/lxfd_attention_robustness_v2_best.pt \
    --unseen-dir data/unseen_generator_test \
    --output-json reports/unseen_generator_results.json
```

The script will automatically compute:
- Per-generator ROC-AUC, PR-AUC, Accuracy, Precision, Recall, and F1 score.
- Per-generator Confusion Matrix.
- Macro-Averaged and Sample-Weighted Cross-Generator Generalization Metrics.
- Zero-leakage verification confirming test generators were excluded from training partitions.
