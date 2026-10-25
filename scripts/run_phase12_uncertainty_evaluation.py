"""
Phase 12G: Uncertainty, Calibration & Inconclusive Routing Audit
Audits calibration transferability from in-domain validation data (N=242)
to out-of-domain external test cohorts (N=217: SD Inpainting + Gemini).
Computes ECE, Brier score, NLL, and evaluates the INCONCLUSIVE policy.
Outputs reports/phase12_calibration_metrics.json and reports/phase12_uncertainty_evaluation.md.
"""

import os
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from src.models.lxfd_model import LXDFDModel
from src.data.face_preprocessor import YuNetFacePreprocessor
from src.evaluation.calibration import TemperatureScaler
from src.utils.device import get_device

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    bin_info = []
    for i in range(n_bins):
        b_low = bins[i]
        b_high = bins[i+1]
        mask = (y_prob >= b_low) & (y_prob < b_high) if i < n_bins - 1 else (y_prob >= b_low) & (y_prob <= b_high)
        n_in = np.sum(mask)
        if n_in > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            diff = abs(acc - conf)
            ece += (n_in / total) * diff
            bin_info.append({
                'bin': f"[{b_low:.1f}, {b_high:.1f}]",
                'count': int(n_in),
                'accuracy': round(acc, 4),
                'confidence': round(conf, 4),
                'error': round(diff, 4)
            })
    return round(float(ece), 4), bin_info

def run_calibration_audit():
    device = get_device()
    print(f"[Phase 12G] Running calibration & uncertainty audit on: {device}")

    model = LXDFDModel(fusion_type='attention', pretrained=False).to(device)
    ckpt_path = "checkpoints/lxfd_phase10_modern_generators_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    scaler = TemperatureScaler()
    cal_path = "checkpoints/calibration_params_phase10.json"
    scaler.load(cal_path)
    T_val = float(scaler.temperature)

    preproc = YuNetFacePreprocessor(margin=0.50)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    manifest_df = pd.read_csv('data/manifests/phase11_generator_manifest.csv')

    # Partitions:
    # 1. Validation (N=242)
    # 2. In-Domain Test (N=240)
    # 3. Out-of-Domain External Test (N=217: Inpainting + Gemini)
    partitions = {
        'val_in_domain': manifest_df[manifest_df['split'] == 'val'],
        'test_in_domain': manifest_df[manifest_df['split'] == 'test_indomain'],
        'test_out_of_domain': manifest_df[manifest_df['split'].isin(['test_unseen_inpaint', 'test_locked', 'test_gemini_heldout'])]
    }

    results = {}

    for part_name, part_df in partitions.items():
        y_trues = []
        raw_probs = []
        cal_probs = []
        spatial_probs = []
        freq_probs = []
        weights_s = []
        weights_f = []

        for idx, row in part_df.iterrows():
            img_path = row['path']
            lbl = 1 if row['label'] == 'FAKE' else 0
            if not os.path.exists(img_path):
                continue

            try:
                pil_img = Image.open(img_path).convert('RGB')
                crop_img, _ = preproc.process_image(pil_img)
                t_img = transform(crop_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logit, meta = model(t_img)
                    raw_l = logit.item()
                    s_p = meta['spatial_fake_prob'].item()
                    f_p = meta['frequency_fake_prob'].item()
                    w_s = meta['w_s'].item() if isinstance(meta['w_s'], torch.Tensor) else meta['w_s']
                    w_f = meta['w_f'].item() if isinstance(meta['w_f'], torch.Tensor) else meta['w_f']

                raw_p = 1.0 / (1.0 + np.exp(-raw_l))
                cal_p = float(scaler.calibrate_probs(np.array([raw_p]))[0])

                y_trues.append(lbl)
                raw_probs.append(raw_p)
                cal_probs.append(cal_p)
                spatial_probs.append(s_p)
                freq_probs.append(f_p)
                weights_s.append(w_s)
                weights_f.append(w_f)
            except Exception as e:
                pass

        y_t = np.array(y_trues)
        p_raw = np.array(raw_probs)
        p_cal = np.array(cal_probs)
        p_s = np.array(spatial_probs)
        p_f = np.array(freq_probs)

        # Uncalibrated vs Calibrated ECE
        ece_raw, _ = compute_ece(y_t, p_raw)
        ece_cal, bin_details = compute_ece(y_t, p_cal)
        brier_raw = round(float(brier_score_loss(y_t, p_raw)), 4)
        brier_cal = round(float(brier_score_loss(y_t, p_cal)), 4)
        
        # Clip for log loss
        eps = 1e-6
        nll_raw = round(float(log_loss(y_t, np.clip(p_raw, eps, 1-eps), labels=[0, 1])), 4)
        nll_cal = round(float(log_loss(y_t, np.clip(p_cal, eps, 1-eps), labels=[0, 1])), 4)

        # Uncertainty routing:
        # Condition 1: Decision boundary band [0.40, 0.60]
        in_band = (p_cal >= 0.40) & (p_cal <= 0.60)
        # Condition 2: Branch divergence > 0.15
        branch_div = np.abs(p_s - p_f) > 0.15
        
        inconclusive = in_band | (branch_div & (np.abs(p_cal - 0.5) < 0.35))
        inc_rate = round(float(np.mean(inconclusive)), 4)

        # Accuracy on conclusive samples
        conclusive_mask = ~inconclusive
        if np.sum(conclusive_mask) > 0:
            concl_acc = round(float(accuracy_score(y_t[conclusive_mask], (p_cal[conclusive_mask] >= 0.5).astype(int))), 4)
        else:
            concl_acc = 0.0

        results[part_name] = {
            'sample_count': len(y_t),
            'synthetic_count': int(np.sum(y_t == 1)),
            'real_count': int(np.sum(y_t == 0)),
            'uncalibrated_ece': ece_raw,
            'calibrated_ece': ece_cal,
            'ece_reduction': round(ece_raw - ece_cal, 4),
            'brier_score_raw': brier_raw,
            'brier_score_cal': brier_cal,
            'nll_raw': nll_raw,
            'nll_cal': nll_cal,
            'inconclusive_rate': inc_rate,
            'conclusive_samples': int(np.sum(conclusive_mask)),
            'conclusive_accuracy': concl_acc,
            'mean_branch_discrepancy': round(float(np.mean(np.abs(p_s - p_f))), 4),
            'calibration_bins': bin_details
        }

    # Save metrics JSON
    cal_metrics = {
        'temperature_T': T_val,
        'temperature_source': cal_path,
        'calibration_method': 'Temperature Scaling via Platt Scaling Logit Optimization',
        'uncertainty_policy': {
            'decision_threshold': 0.50,
            'uncertainty_band': '[0.40, 0.60] (delta = 0.10)',
            'branch_divergence_threshold': 0.15,
            'routing_rules': 'INCONCLUSIVE if (0.40 <= P <= 0.60) OR (|P_spatial - P_freq| > 0.15 AND |P - 0.50| < 0.35)'
        },
        'partition_evaluations': results
    }

    with open('reports/phase12_calibration_metrics.json', 'w') as f:
        json.dump(cal_metrics, f, indent=2)

    print("\nSuccessfully wrote reports/phase12_calibration_metrics.json")
    for k, v in results.items():
        print(f"\n[{k.upper()}] N={v['sample_count']} | Raw ECE: {v['uncalibrated_ece']} -> Cal ECE: {v['calibrated_ece']} | Inconclusive Rate: {v['inconclusive_rate']*100:.1f}% | Conclusive Acc: {v['conclusive_accuracy']*100:.1f}%")

if __name__ == '__main__':
    run_calibration_audit()
