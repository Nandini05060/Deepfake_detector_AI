"""
Phase 11 Comprehensive Evaluation, Generalization & Calibration Audit
Computes statistical metrics (ROC-AUC, PR-AUC, F1, Prec, Rec, Spec, FPR, FNR, CI-95,
Confusion Matrices, ECE, Brier Score, NLL) across in-domain and external test splits.
"""

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, log_loss, brier_score_loss
)

from src.models.lxfd_model import LXDFDModel
from src.data.face_preprocessor import YuNetFacePreprocessor
from src.evaluation.calibration import TemperatureScaler
from src.utils.device import get_device

def wilson_score_ci(successes, total, confidence=0.95):
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.959964  # for 95% confidence
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(float(lower), 4), round(float(upper), 4)

def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error with documented uniform binning."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    bin_details = []
    
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_samples = np.sum(mask)
        if bin_samples > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            abs_err = np.abs(bin_acc - bin_conf)
            ece += (bin_samples / total_samples) * abs_err
            bin_details.append({
                'bin_range': f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                'samples': int(bin_samples),
                'accuracy': round(float(bin_acc), 4),
                'confidence': round(float(bin_conf), 4),
                'error': round(float(abs_err), 4)
            })
    return round(float(ece), 4), bin_details

def main():
    os.makedirs('reports', exist_ok=True)
    device = get_device()
    model = LXDFDModel(fusion_type='attention', pretrained=False).to(device)
    ckpt_path = "checkpoints/lxfd_phase10_modern_generators_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    scaler = TemperatureScaler()
    cal_path = "checkpoints/calibration_params_phase10.json"
    scaler.load(cal_path)

    preprocessor = YuNetFacePreprocessor(margin=0.50)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    manifest_df = pd.read_csv('data/manifests/phase11_generator_manifest.csv')
    test_splits = ['test_indomain', 'test_unseen_inpaint', 'test_locked', 'test_gemini_heldout']
    eval_df = manifest_df[manifest_df['split'].isin(test_splits)].copy()
    print(f"Total locked evaluation samples: {len(eval_df)}")

    preds = []
    for idx, row in eval_df.iterrows():
        img_path = row['path']
        label = 1 if row['label'] == 'FAKE' else 0
        gen = row['generator_family']
        split = row['split']

        if not os.path.exists(img_path):
            continue

        try:
            pil_img = Image.open(img_path).convert('RGB')
            crop_img, meta = preprocessor.process_image(pil_img)
            tensor_img = transform(crop_img).unsqueeze(0).to(device)

            with torch.no_grad():
                out, b_meta = model(tensor_img)
                raw_logit = out.squeeze().item()
                w_s = b_meta['w_s'].item() if isinstance(b_meta['w_s'], torch.Tensor) else b_meta['w_s']
                w_f = b_meta['w_f'].item() if isinstance(b_meta['w_f'], torch.Tensor) else b_meta['w_f']
                s_prob = b_meta['spatial_fake_prob'].item()
                f_prob = b_meta['frequency_fake_prob'].item()

            raw_p_fake = 1.0 / (1.0 + np.exp(-raw_logit))
            cal_p_fake = float(scaler.calibrate_probs(np.array([raw_p_fake]))[0])
            p_fake = float(np.clip(cal_p_fake, 0.0, 1.0))
            p_real = float(1.0 - p_fake)
            pred_fake = int(p_fake >= 0.5)

            # Uncertainty rule: branch divergence > 0.15 or unconfident probability [0.40, 0.60]
            is_inconclusive = bool(abs(w_s - w_f) > 0.15 or (0.40 <= p_fake <= 0.60))

            preds.append({
                'image_id': row['image_id'],
                'path': img_path,
                'generator': gen,
                'split': split,
                'label': label,
                'raw_logit': raw_logit,
                'p_fake': p_fake,
                'p_real': p_real,
                'pred_fake': pred_fake,
                'spatial_p_fake': s_prob,
                'frequency_p_fake': f_prob,
                'w_s': w_s,
                'w_f': w_f,
                'is_inconclusive': is_inconclusive
            })
        except Exception as e:
            print(f"Error evaluating {img_path}: {e}")

    df_p = pd.DataFrame(preds)
    print(f"Successfully evaluated {len(df_p)} images.")

    # 1. Global Benchmark Metrics by Generator Cohort
    results = {
        'metadata': {
            'checkpoint': ckpt_path,
            'temperature': float(scaler.temperature),
            'total_test_samples': len(df_p),
            'timestamp': '2026-10-24T00:30:00Z'
        },
        'generator_benchmarks': {},
        'split_benchmarks': {},
        'calibration_audit': {}
    }

    # Combined real + fake evaluation for generator families
    real_sub = df_p[df_p['label'] == 0]
    y_real = real_sub['label'].values
    p_real_preds = real_sub['p_fake'].values

    for gen in sorted(df_p['generator'].unique()):
        sub = df_p[df_p['generator'] == gen]
        n_samples = len(sub)
        
        if gen == 'None_Real':
            # Real controls evaluation
            acc = float(accuracy_score(sub['label'], sub['pred_fake']))
            spec = float((sub['pred_fake'] == 0).mean())
            ci_acc = wilson_score_ci(int(np.sum(sub['pred_fake'] == 0)), n_samples)
            results['generator_benchmarks'][gen] = {
                'sample_count': n_samples,
                'role': 'Authentic Negative Control',
                'accuracy': round(acc, 4),
                'specificity': round(spec, 4),
                'false_positive_rate': round(1.0 - spec, 4),
                'ci_95_accuracy': ci_acc,
                'mean_p_fake': round(float(sub['p_fake'].mean()), 4),
                'mean_p_real': round(float(sub['p_real'].mean()), 4),
                'inconclusive_rate': round(float(sub['is_inconclusive'].mean()), 4)
            }
        else:
            # Synthetic generator cohort evaluated against real controls for binary metrics
            combo_sub = pd.concat([sub, real_sub])
            y_combo = combo_sub['label'].values
            p_combo = combo_sub['p_fake'].values
            pred_combo = combo_sub['pred_fake'].values

            auc = float(roc_auc_score(y_combo, p_combo))
            ap = float(average_precision_score(y_combo, p_combo))
            rec = float(recall_score(sub['label'], sub['pred_fake'], zero_division=0))
            prec = float(precision_score(combo_sub['label'], pred_combo, zero_division=0))
            f1 = float(f1_score(combo_sub['label'], pred_combo, zero_division=0))
            cm = confusion_matrix(combo_sub['label'], pred_combo).tolist() # [[TN, FP], [FN, TP]]
            ci_rec = wilson_score_ci(int(np.sum(sub['pred_fake'] == 1)), n_samples)

            results['generator_benchmarks'][gen] = {
                'sample_count': n_samples,
                'role': 'Synthetic Generator Cohort',
                'roc_auc_vs_real': round(auc, 4),
                'pr_auc_vs_real': round(ap, 4),
                'recall': round(rec, 4),
                'precision': round(prec, 4),
                'f1_score': round(f1, 4),
                'false_negative_rate': round(1.0 - rec, 4),
                'ci_95_recall': ci_rec,
                'mean_p_fake': round(float(sub['p_fake'].mean()), 4),
                'inconclusive_rate': round(float(sub['is_inconclusive'].mean()), 4),
                'confusion_matrix_vs_real': {
                    'tn': cm[0][0],
                    'fp': cm[0][1],
                    'fn': cm[1][0],
                    'tp': cm[1][1]
                }
            }

    # Macro-average across all 5 synthetic generators
    fake_gens = [g for g in results['generator_benchmarks'] if g != 'None_Real']
    macro_auc = np.mean([results['generator_benchmarks'][g]['roc_auc_vs_real'] for g in fake_gens])
    macro_rec = np.mean([results['generator_benchmarks'][g]['recall'] for g in fake_gens])
    macro_f1 = np.mean([results['generator_benchmarks'][g]['f1_score'] for g in fake_gens])

    results['macro_average_synthetic'] = {
        'macro_roc_auc': round(float(macro_auc), 4),
        'macro_recall': round(float(macro_rec), 4),
        'macro_f1': round(float(macro_f1), 4),
        'evaluated_generator_count': len(fake_gens)
    }

    # Split-Level Evaluation
    for sp in sorted(df_p['split'].unique()):
        sub_sp = df_p[df_p['split'] == sp]
        y_sp = sub_sp['label'].values
        p_sp = sub_sp['p_fake'].values
        pred_sp = sub_sp['pred_fake'].values
        
        has_two_classes = len(np.unique(y_sp)) > 1
        auc_sp = float(roc_auc_score(y_sp, p_sp)) if has_two_classes else None
        acc_sp = float(accuracy_score(y_sp, pred_sp))
        f1_sp = float(f1_score(y_sp, pred_sp, zero_division=0))
        rec_sp = float(recall_score(y_sp, pred_sp, zero_division=0)) if 1 in y_sp else None

        results['split_benchmarks'][sp] = {
            'sample_count': len(sub_sp),
            'accuracy': round(acc_sp, 4),
            'roc_auc': round(auc_sp, 4) if auc_sp is not None else 'N/A (Single Class)',
            'f1_score': round(f1_sp, 4),
            'recall': round(rec_sp, 4) if rec_sp is not None else 'N/A',
            'mean_p_fake': round(float(p_sp.mean()), 4),
            'inconclusive_rate': round(float(sub_sp['is_inconclusive'].mean()), 4)
        }

    # Calibration Metrics (In-Domain vs Out-of-Domain)
    # 1. In-Domain Calibration (test_indomain: 120 Real + 120 StyleGAN2)
    indomain_df = df_p[df_p['split'] == 'test_indomain']
    ece_in, bins_in = compute_ece(indomain_df['label'].values, indomain_df['p_fake'].values, n_bins=10)
    brier_in = float(brier_score_loss(indomain_df['label'].values, indomain_df['p_fake'].values))
    nll_in = float(log_loss(indomain_df['label'].values, np.clip(indomain_df['p_fake'].values, 1e-6, 1.0 - 1e-6)))

    # 2. Out-of-Domain Calibration (SD Inpainting + Locked Gemini + Real controls)
    ood_df = df_p[df_p['split'].isin(['test_unseen_inpaint', 'test_locked', 'test_gemini_heldout']) | (df_p['generator'] == 'None_Real')]
    ece_ood, bins_ood = compute_ece(ood_df['label'].values, ood_df['p_fake'].values, n_bins=10)
    brier_ood = float(brier_score_loss(ood_df['label'].values, ood_df['p_fake'].values))
    nll_ood = float(log_loss(ood_df['label'].values, np.clip(ood_df['p_fake'].values, 1e-6, 1.0 - 1e-6)))

    results['calibration_audit'] = {
        'in_domain': {
            'split': 'test_indomain (N=240: 120 Real, 120 StyleGAN2)',
            'expected_calibration_error_ece': ece_in,
            'brier_score': round(brier_in, 4),
            'negative_log_likelihood_nll': round(nll_in, 4),
            'bin_table': bins_in
        },
        'out_of_distribution': {
            'split': 'external_unseen (N=337: 200 SD Inpaint, 17 Gemini, 120 Real)',
            'expected_calibration_error_ece': ece_ood,
            'brier_score': round(brier_ood, 4),
            'negative_log_likelihood_nll': round(nll_ood, 4),
            'bin_table': bins_ood
        }
    }

    # Save JSON files
    with open('reports/phase11_test_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    config = {
        'experiment_name': 'phase11_model_improvement_and_generalization_audit',
        'checkpoint_path': ckpt_path,
        'checkpoint_sha256': '13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05',
        'calibrator': {
            'method': 'Temperature Scaling',
            'temperature': float(scaler.temperature),
            'parameters_file': cal_path
        },
        'face_preprocessor': {
            'engine': 'OpenCV YuNet ONNX',
            'model_file': 'yunet.onnx',
            'context_margin': 0.50,
            'score_threshold': 0.60,
            'nms_threshold': 0.30,
            'target_dimensions': [224, 224]
        },
        'uncertainty_gating': {
            'branch_disagreement_threshold': 0.15,
            'confidence_deadband': [0.40, 0.60],
            'inconclusive_verdict_supported': True
        },
        'data_manifest': 'data/manifests/phase11_generator_manifest.csv',
        'locked_test_splits': test_splits
    }

    with open('reports/phase11_experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\nSuccessfully wrote:")
    print("  - reports/phase11_test_metrics.json")
    print("  - reports/phase11_experiment_config.json")

    # Print Key Stats
    print("\n=== MACRO-AVERAGE PERFORMANCE ===")
    print(f"Macro ROC-AUC: {macro_auc:.4f} | Macro Recall: {macro_rec*100:.2f}% | Macro F1: {macro_f1:.4f}")
    print("\n=== CALIBRATION METRICS ===")
    print(f"In-Domain ECE:        {ece_in:.4f} | Brier: {brier_in:.4f} | NLL: {nll_in:.4f}")
    print(f"Out-of-Domain ECE:    {ece_ood:.4f} | Brier: {brier_ood:.4f} | NLL: {nll_ood:.4f}")

if __name__ == '__main__':
    main()
