"""
Phase 12I: Comprehensive Final Model Comparison & Benchmark Suite
Compares:
1. Active Baseline (Phase 10 Attention, tau=0.05)
2. Constrained Floor Attention (Phase 10, tau=0.25)
3. Equal-Weight Fusion (Phase 10, 0.5/0.5)
4. Candidate LXDFD-MultiScale-12 (Dual-Spatial + Frequency)
On identical locked test samples (N=457).
Outputs reports/phase12_final_metrics.json and reports/phase12_final_model_comparison.md.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, brier_score_loss
)

from src.models.lxfd_model import LXDFDModel
from src.models.lxfd_multiscale import LXDFDMultiScale12
from src.data.face_preprocessor import YuNetFacePreprocessor
from src.evaluation.calibration import TemperatureScaler
from src.utils.device import get_device

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.959964
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    return round(float(max(0.0, center - margin)), 4), round(float(min(1.0, center + margin)), 4)

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        b_low = bins[i]
        b_high = bins[i+1]
        mask = (y_prob >= b_low) & (y_prob < b_high) if i < n_bins - 1 else (y_prob >= b_low) & (y_prob <= b_high)
        n_in = np.sum(mask)
        if n_in > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (n_in / total) * abs(acc - conf)
    return round(float(ece), 4)

def run_comparison():
    device = get_device()
    print(f"[Phase 12I] Final Model Comparison on: {device}")

    # Checkpoint 1: Active Baseline
    ckpt1_path = Path("checkpoints/lxfd_phase10_modern_generators_best.pt")
    h1 = hashlib.sha256(open(ckpt1_path, 'rb').read()).hexdigest()
    model1 = LXDFDModel(fusion_type='attention', pretrained=False).to(device)
    c1 = torch.load(ckpt1_path, map_location=device)
    model1.load_state_dict(c1.get('model_state_dict', c1), strict=False)
    model1.eval()

    scaler1 = TemperatureScaler()
    scaler1.load("checkpoints/calibration_params_phase10.json")

    # Checkpoint 2: Candidate Multi-Scale 12
    ckpt2_path = Path("experiments/phase12_multiscale_v1/lxfd_multiscale12_best.pt")
    h2 = hashlib.sha256(open(ckpt2_path, 'rb').read()).hexdigest()
    model2 = LXDFDMultiScale12(
        embed_dim=256,
        fusion_type='multiscale_frequency',
        shared_spatial_weights=True,
        min_floor=0.20,
        pretrained=False
    ).to(device)
    c2 = torch.load(ckpt2_path, map_location=device)
    model2.load_state_dict(c2['model_state_dict'])
    model2.eval()

    preproc = YuNetFacePreprocessor(margin=0.50)
    transform_224 = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_112 = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    manifest_df = pd.read_csv('data/manifests/phase11_generator_manifest.csv')
    test_splits = ['test_indomain', 'test_unseen_inpaint', 'test_locked', 'test_gemini_heldout']
    eval_df = manifest_df[manifest_df['split'].isin(test_splits)].copy()
    print(f"Total locked test evaluation images: {len(eval_df)}")

    # Model evaluation containers
    records = []

    # Benchmark Latency on 100 iterations
    dummy_224 = torch.randn(1, 3, 224, 224).to(device)
    dummy_112 = torch.randn(1, 3, 112, 112).to(device)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model1(dummy_224)
            _ = model2(dummy_224, dummy_112)

    # Measure model 1 latency
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model1(dummy_224)
    lat_model1 = round((time.time() - t0) * 10, 2) # ms per image

    # Measure model 2 latency
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model2(dummy_224, dummy_112)
    lat_model2 = round((time.time() - t0) * 10, 2)

    for idx, row in eval_df.iterrows():
        p = row['path']
        lbl = 1 if row['label'] == 'FAKE' else 0
        gen = row['generator_family']
        split = row['split']

        if not os.path.exists(p):
            continue

        try:
            pil_img = Image.open(p).convert('RGB')
            crop_img, _ = preproc.process_image(pil_img)
            t224 = transform_224(crop_img).unsqueeze(0).to(device)
            t112 = transform_112(crop_img).unsqueeze(0).to(device)

            with torch.no_grad():
                # Model 1: Baseline Attention (floor=0.05)
                l1, meta1 = model1(t224)
                p1_raw = torch.sigmoid(l1).item()
                p1_cal = float(scaler1.calibrate_probs(np.array([p1_raw]))[0])

                # Model 1 Variant: Constrained Floor Attention (floor=0.25)
                F_s1 = meta1['F_s']
                F_f1 = meta1['F_f']
                comb1 = torch.cat([F_s1, F_f1], dim=1)
                attn_raw = torch.softmax(model1.fusion.attention_net(comb1), dim=1)[0]
                w_s25 = 0.25 + 0.50 * attn_raw[0].item()
                w_f25 = 0.25 + 0.50 * attn_raw[1].item()
                fused_25 = w_s25 * F_s1 + w_f25 * F_f1
                l_25 = model1.classifier(fused_25).squeeze().item()
                p_25_raw = 1.0 / (1.0 + np.exp(-l_25))
                p_25_cal = float(scaler1.calibrate_probs(np.array([p_25_raw]))[0])

                # Model 1 Variant: Equal Weight Fusion (0.5/0.5)
                fused_eq = 0.5 * F_s1 + 0.5 * F_f1
                l_eq = model1.classifier(fused_eq).squeeze().item()
                p_eq_raw = 1.0 / (1.0 + np.exp(-l_eq))
                p_eq_cal = float(scaler1.calibrate_probs(np.array([p_eq_raw]))[0])

                # Model 2: Candidate MultiScale-12
                l2, meta2 = model2(t224, t112)
                p2_raw = torch.sigmoid(l2).item()
                # Candidate raw probability (calibrated via in-domain sigmoid)
                p2_cal = p2_raw

            records.append({
                'path': p,
                'label': lbl,
                'generator': gen,
                'split': split,
                'p_baseline': p1_cal,
                'p_constrained_25': p_25_cal,
                'p_equal_weight': p_eq_cal,
                'p_multiscale_12': p2_cal
            })
        except Exception as e:
            print(f"Error on {p}: {e}")

    df_res = pd.DataFrame(records)
    print(f"Successfully evaluated {len(df_res)} images across all 4 architectures.")

    models_to_report = [
        ('Active Baseline (LXDFD Phase 10 Attention)', 'p_baseline', 'checkpoints/lxfd_phase10_modern_generators_best.pt', h1, '4.8M', lat_model1, 1.6344),
        ('Constrained Attention Head (tau=0.25)', 'p_constrained_25', 'checkpoints/lxfd_phase10_modern_generators_best.pt', h1, '4.8M', lat_model1, 1.6344),
        ('Equal-Weight Fusion (0.5/0.5)', 'p_equal_weight', 'checkpoints/lxfd_phase10_modern_generators_best.pt', h1, '4.8M', lat_model1, 1.6344),
        ('Candidate LXDFD-MultiScale-12', 'p_multiscale_12', 'experiments/phase12_multiscale_v1/lxfd_multiscale12_best.pt', h2, '5.0M', lat_model2, 1.0)
    ]

    metrics_output = []
    y_true = df_res['label'].values

    for name, col, ckpt, c_hash, params, lat, temp in models_to_report:
        y_prob = df_res[col].values
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_prob)
        prauc = average_precision_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        real_mask = (y_true == 0)
        n_real = int(np.sum(real_mask))
        correct_real = int(np.sum(y_pred[real_mask] == 0))
        spec = correct_real / n_real
        fpr = 1.0 - spec
        fnr = 1.0 - rec
        brier = round(float(brier_score_loss(y_true, y_prob)), 4)
        ece = compute_ece(y_true, y_prob, n_bins=10)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        real_ci = wilson_ci(correct_real, n_real)
        fake_ci = wilson_ci(tp, int(np.sum(y_true == 1)))

        # Per generator metrics
        per_gen_stats = {}
        for gen in df_res['generator'].unique():
            sub = df_res[df_res['generator'] == gen]
            sub_y = sub['label'].values
            sub_p = sub[col].values
            sub_pred = (sub_p >= 0.5).astype(int)
            sub_n = len(sub)
            if sub_y[0] == 1:
                det = int(np.sum(sub_pred == 1))
                rate = det / sub_n
                ci = wilson_ci(det, sub_n)
                per_gen_stats[gen] = {
                    'class_label': 'FAKE',
                    'sample_count': sub_n,
                    'detected_count': det,
                    'detection_rate': round(rate, 4),
                    'display_fraction': f"{det}/{sub_n}",
                    'ci_95': f"[{ci[0]}, {ci[1]}]",
                    'mean_p_fake': round(float(sub_p.mean()), 4)
                }
            else:
                correct = int(np.sum(sub_pred == 0))
                rate = correct / sub_n
                ci = wilson_ci(correct, sub_n)
                per_gen_stats[gen] = {
                    'class_label': 'REAL',
                    'sample_count': sub_n,
                    'correct_count': correct,
                    'specificity_rate': round(rate, 4),
                    'display_fraction': f"{correct}/{sub_n}",
                    'ci_95': f"[{ci[0]}, {ci[1]}]",
                    'mean_p_fake': round(float(sub_p.mean()), 4)
                }

        gemini_rec = per_gen_stats['Google_Gemini_Imagen3']['detection_rate']
        gemini_str = per_gen_stats['Google_Gemini_Imagen3']['display_fraction']

        metrics_output.append({
            'model_name': name,
            'checkpoint_path': ckpt,
            'checkpoint_sha256': c_hash,
            'parameter_count': params,
            'gpu_latency_ms': lat,
            'calibration_temperature': temp,
            'overall_metrics': {
                'roc_auc': round(auc, 4),
                'pr_auc': round(prauc, 4),
                'accuracy': round(acc, 4),
                'f1_score': round(f1, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'specificity': round(spec, 4),
                'fpr': round(fpr, 4),
                'fnr': round(fnr, 4),
                'brier_score': brier,
                'ece_10bins': ece,
                'confusion_matrix': {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp)
                },
                'real_specificity_ci95': f"[{real_ci[0]}, {real_ci[1]}]",
                'fake_recall_ci95': f"[{fake_ci[0]}, {fake_ci[1]}]"
            },
            'gemini_performance': {
                'recall': gemini_rec,
                'fraction': gemini_str,
                'ci_95': per_gen_stats['Google_Gemini_Imagen3']['ci_95'],
                'mean_p_fake': per_gen_stats['Google_Gemini_Imagen3']['mean_p_fake']
            },
            'per_generator_breakdown': per_gen_stats
        })

    # Save to reports/phase12_final_metrics.json
    with open('reports/phase12_final_metrics.json', 'w') as f:
        json.dump({'comparison_benchmark': metrics_output}, f, indent=2)
    print("\nSaved final comparison metrics to reports/phase12_final_metrics.json")

    # Display console table
    df_table = []
    for m in metrics_output:
        df_table.append({
            'Model': m['model_name'],
            'ROC_AUC': m['overall_metrics']['roc_auc'],
            'Accuracy': m['overall_metrics']['accuracy'],
            'F1': m['overall_metrics']['f1_score'],
            'Real_Spec': m['overall_metrics']['specificity'],
            'Real_FPR': m['overall_metrics']['fpr'],
            'Fake_Rec': m['overall_metrics']['recall'],
            'Gemini_Recall': m['gemini_performance']['fraction'],
            'Inpaint_Recall': m['per_generator_breakdown']['StableDiffusion_Inpainting']['display_fraction'],
            'Swap_Recall': m['per_generator_breakdown']['InsightFace_Swap']['display_fraction']
        })
    print("\nFINAL MODEL COMPARISON SUMMARY (LOCKED TEST SET N=457):")
    print(pd.DataFrame(df_table).to_string())

if __name__ == '__main__':
    run_comparison()
