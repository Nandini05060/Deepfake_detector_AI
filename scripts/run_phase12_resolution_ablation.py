"""
Phase 12C: Controlled Resolution Ablation Experiment
Evaluates the locked test dataset (N=457) across 112x112, 160x160, 224x224 (baseline),
and 320x320 resolutions to rigorously test the downsampling hypothesis, measure
the trade-off with authentic photograph false positives, and quantify prediction flips.
"""

import os
import sys
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
    f1_score, accuracy_score, confusion_matrix
)

from src.models.lxfd_model import LXDFDModel
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

def run_experiment():
    device = get_device()
    print(f"[Phase 12C] Initializing model on device: {device}")
    
    model = LXDFDModel(fusion_type='attention', pretrained=False).to(device)
    ckpt_path = "checkpoints/lxfd_phase10_modern_generators_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    scaler = TemperatureScaler()
    cal_path = "checkpoints/calibration_params_phase10.json"
    scaler.load(cal_path)
    cal_T = float(scaler.temperature)

    preprocessor = YuNetFacePreprocessor(margin=0.50)
    
    # Base normalization
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    manifest_df = pd.read_csv('data/manifests/phase11_generator_manifest.csv')
    test_splits = ['test_indomain', 'test_unseen_inpaint', 'test_locked', 'test_gemini_heldout']
    eval_df = manifest_df[manifest_df['split'].isin(test_splits)].copy()
    print(f"Total locked evaluation samples: {len(eval_df)}")

    resolutions = [112, 160, 224, 320]
    
    # Store predictions for each resolution
    # key: resolution, value: list of records
    all_res_preds = {res: [] for res in resolutions}

    for idx, row in eval_df.iterrows():
        img_path = row['path']
        label = 1 if row['label'] == 'FAKE' else 0
        gen = row['generator_family']
        split = row['split']

        if not os.path.exists(img_path):
            continue

        try:
            pil_img = Image.open(img_path).convert('RGB')
            # Extract standard face crop (margin=0.50)
            crop_img, meta = preprocessor.process_image(pil_img)
            
            for res in resolutions:
                # Resize to resolution, then resize to 224 for consistent tensor input
                # This models bandwidth/resolution scaling:
                if res != 224:
                    res_img = crop_img.resize((res, res), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
                else:
                    res_img = crop_img.resize((224, 224), Image.Resampling.BILINEAR)
                
                t_img = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=norm_mean, std=norm_std)
                ])(res_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logit, out_meta = model(t_img)
                    raw_l = logit.item()
                    s_prob = out_meta['spatial_fake_prob'].item()
                    f_prob = out_meta['frequency_fake_prob'].item()
                    w_s = out_meta['w_s'].item() if isinstance(out_meta['w_s'], torch.Tensor) else out_meta['w_s']
                    w_f = out_meta['w_f'].item() if isinstance(out_meta['w_f'], torch.Tensor) else out_meta['w_f']

                # Temperature scaling
                raw_p = 1.0 / (1.0 + np.exp(-raw_l))
                cal_p = float(scaler.calibrate_probs(np.array([raw_p]))[0])
                pred = 1 if cal_p >= 0.5 else 0

                all_res_preds[res].append({
                    'path': img_path,
                    'label': label,
                    'generator': gen,
                    'split': split,
                    'raw_logit': raw_l,
                    'raw_prob': raw_p,
                    'cal_p_fake': cal_p,
                    'pred': pred,
                    'spatial_prob': s_prob,
                    'frequency_prob': f_prob,
                    'w_s': w_s,
                    'w_f': w_f
                })

        except Exception as e:
            print(f"Error on {img_path}: {e}")

    # Build comparative metrics table
    metrics_records = []
    
    baseline_224_preds = {r['path']: r['pred'] for r in all_res_preds[224]}

    for res in resolutions:
        df_res = pd.DataFrame(all_res_preds[res])
        y_true = df_res['label'].values
        y_prob = df_res['cal_p_fake'].values
        y_pred = df_res['pred'].values

        # Overall metrics
        auc = roc_auc_score(y_true, y_prob)
        prauc = average_precision_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Real controls (label == 0)
        real_mask = (y_true == 0)
        n_real = int(np.sum(real_mask))
        correct_real = int(np.sum(y_pred[real_mask] == 0))
        specificity = correct_real / n_real
        fpr = 1.0 - specificity
        real_ci = wilson_ci(correct_real, n_real)

        # Synthetic samples (label == 1)
        fake_mask = (y_true == 1)
        n_fake = int(np.sum(fake_mask))
        detected_fake = int(np.sum(y_pred[fake_mask] == 1))
        overall_recall = detected_fake / n_fake
        fnr = 1.0 - overall_recall
        fake_ci = wilson_ci(detected_fake, n_fake)

        # Gemini specific cohort
        gem_df = df_res[df_res['generator'] == 'Google_Gemini_Imagen3']
        n_gem = len(gem_df)
        gem_det = int(np.sum(gem_df['pred'] == 1))
        gem_rec = gem_det / n_gem if n_gem > 0 else 0.0
        gem_ci = wilson_ci(gem_det, n_gem)
        gem_mean_prob = float(gem_df['cal_p_fake'].mean())
        gem_mean_spat = float(gem_df['spatial_prob'].mean())
        gem_mean_freq = float(gem_df['frequency_prob'].mean())

        # Other generators
        inpaint_df = df_res[df_res['generator'] == 'StableDiffusion_Inpainting']
        inpaint_rec = float(np.mean(inpaint_df['pred'] == 1))
        
        sd_df = df_res[df_res['generator'] == 'StableDiffusion_v1.5']
        sd_rec = float(np.mean(sd_df['pred'] == 1))

        swap_df = df_res[df_res['generator'] == 'InsightFace_Swap']
        swap_rec = float(np.mean(swap_df['pred'] == 1))

        style_df = df_res[df_res['generator'] == 'StyleGAN2']
        style_rec = float(np.mean(style_df['pred'] == 1))

        # Prediction changes relative to 224
        if res == 224:
            flip_count = 0
            flip_rate = 0.0
        else:
            flips = sum(1 for r in all_res_preds[res] if r['pred'] != baseline_224_preds.get(r['path'], r['pred']))
            flip_count = flips
            flip_rate = round(flips / len(df_res), 4)

        metrics_records.append({
            'Resolution': f"{res}x{res}",
            'Total_N': len(df_res),
            'ROC_AUC': round(auc, 4),
            'PR_AUC': round(prauc, 4),
            'Accuracy': round(acc, 4),
            'F1_Score': round(f1, 4),
            'Real_Specificity': round(specificity, 4),
            'Real_FPR': round(fpr, 4),
            'Real_CI_95': f"[{real_ci[0]}, {real_ci[1]}]",
            'Overall_Fake_Recall': round(overall_recall, 4),
            'Overall_FNR': round(fnr, 4),
            'Gemini_Recall_Pct': round(gem_rec * 100, 2),
            'Gemini_Numerator_Denominator': f"{gem_det}/{n_gem}",
            'Gemini_CI_95': f"[{gem_ci[0]}, {gem_ci[1]}]",
            'Gemini_Mean_P_Fake': round(gem_mean_prob, 4),
            'Gemini_Mean_Spatial_P': round(gem_mean_spat, 4),
            'Gemini_Mean_Freq_P': round(gem_mean_freq, 4),
            'SD_Inpainting_Recall': round(inpaint_rec, 4),
            'SD_v15_Recall': round(sd_rec, 4),
            'InsightFace_Recall': round(swap_rec, 4),
            'StyleGAN2_Recall': round(style_rec, 4),
            'Prediction_Flips_vs_224': flip_count,
            'Flip_Rate_vs_224': flip_rate
        })

    out_df = pd.DataFrame(metrics_records)
    out_csv = 'reports/phase12_resolution_metrics.csv'
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved resolution metrics to {out_csv}")
    print("\nResolution Comparison Summary Table:")
    print(out_df[['Resolution', 'ROC_AUC', 'Real_Specificity', 'Real_FPR', 'Gemini_Numerator_Denominator', 'Gemini_Recall_Pct', 'Gemini_Mean_P_Fake', 'Flip_Rate_vs_224']].to_string())

if __name__ == '__main__':
    run_experiment()
