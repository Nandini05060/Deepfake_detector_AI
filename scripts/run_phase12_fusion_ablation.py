"""
Phase 12E: Systematic Frequency Branch and Fusion Strategy Ablation
Evaluates:
A. Spatial-Only
B. Frequency-Only
C. Existing Attention Fusion (min floor tau = 0.05)
D. Equal-Weight Average Fusion (0.5 / 0.5)
E. Constrained Floor Attention (tau = 0.25)
F. Constrained Floor Attention (tau = 0.35)
Across the locked test set (N=457) and all generator cohorts.
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

def run_fusion_ablation():
    device = get_device()
    print(f"[Phase 12E] Evaluating fusion strategies on: {device}")
    
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

    records = []

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
            t_img = transform(crop_img).unsqueeze(0).to(device)

            with torch.no_grad():
                F_s = model.spatial_branch(t_img)
                F_f, dct_map = model.frequency_branch(t_img)

                # Raw logits through classifier head
                logit_s = model.classifier(F_s).squeeze().item()
                logit_f = model.classifier(F_f).squeeze().item()

                # Attention raw weights
                combined = torch.cat([F_s, F_f], dim=1)
                attn_logits = model.fusion.attention_net(combined)
                raw_w = torch.softmax(attn_logits, dim=1)[0]
                alpha_s, alpha_f = raw_w[0].item(), raw_w[1].item()

                # Strategy C: Active Baseline Attention (floor = 0.05)
                w_s_base = 0.05 + 0.90 * alpha_s
                w_f_base = 0.05 + 0.90 * alpha_f
                fused_base = w_s_base * F_s + w_f_base * F_f
                logit_base = model.classifier(fused_base).squeeze().item()

                # Strategy D: Equal-Weight Fusion (0.5 / 0.5)
                fused_eq = 0.5 * F_s + 0.5 * F_f
                logit_eq = model.classifier(fused_eq).squeeze().item()

                # Strategy E: Constrained Attention (floor = 0.25)
                w_s_25 = 0.25 + 0.50 * alpha_s
                w_f_25 = 0.25 + 0.50 * alpha_f
                fused_25 = w_s_25 * F_s + w_f_25 * F_f
                logit_25 = model.classifier(fused_25).squeeze().item()

                # Strategy F: Constrained Attention (floor = 0.35)
                w_s_35 = 0.35 + 0.30 * alpha_s
                w_f_35 = 0.35 + 0.30 * alpha_f
                fused_35 = w_s_35 * F_s + w_f_35 * F_f
                logit_35 = model.classifier(fused_35).squeeze().item()

            p_s = 1.0 / (1.0 + np.exp(-logit_s))
            p_f = 1.0 / (1.0 + np.exp(-logit_f))
            p_base = 1.0 / (1.0 + np.exp(-logit_base))
            p_eq = 1.0 / (1.0 + np.exp(-logit_eq))
            p_25 = 1.0 / (1.0 + np.exp(-logit_25))
            p_35 = 1.0 / (1.0 + np.exp(-logit_35))

            records.append({
                'path': img_path,
                'label': label,
                'generator': gen,
                'split': split,
                'p_spatial': p_s,
                'p_freq': p_f,
                'p_attention_base': p_base,
                'p_equal_weight': p_eq,
                'p_constrained_25': p_25,
                'p_constrained_35': p_35,
                'w_s_base': w_s_base,
                'w_f_base': w_f_base,
                'alpha_s': alpha_s,
                'alpha_f': alpha_f
            })

        except Exception as e:
            print(f"Error on {img_path}: {e}")

    df_all = pd.DataFrame(records)
    print(f"Successfully evaluated {len(df_all)} images for fusion ablation.")

    strategies = [
        ('Spatial-Only', 'p_spatial'),
        ('Frequency-Only', 'p_freq'),
        ('Attention Fusion (tau=0.05)', 'p_attention_base'),
        ('Equal-Weight (0.5/0.5)', 'p_equal_weight'),
        ('Constrained Attention (tau=0.25)', 'p_constrained_25'),
        ('Constrained Attention (tau=0.35)', 'p_constrained_35')
    ]

    summary_rows = []
    
    # 1. Overall Metrics across All Test Splits
    for strat_name, col in strategies:
        y_true = df_all['label'].values
        y_prob = df_all[col].values
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_prob)
        prauc = average_precision_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Real controls
        real_mask = (y_true == 0)
        spec = accuracy_score(y_true[real_mask], y_pred[real_mask])
        fpr = 1.0 - spec

        # Fake recall
        fake_mask = (y_true == 1)
        rec = recall_score(y_true[fake_mask], y_pred[fake_mask], zero_division=0)

        # Per generator recalls
        gem_mask = (df_all['generator'] == 'Google_Gemini_Imagen3')
        gem_rec = recall_score(y_true[gem_mask], y_pred[gem_mask], zero_division=0)
        gem_mean_p = float(y_prob[gem_mask].mean())

        inpaint_mask = (df_all['generator'] == 'StableDiffusion_Inpainting')
        inpaint_rec = recall_score(y_true[inpaint_mask], y_pred[inpaint_mask], zero_division=0)

        swap_mask = (df_all['generator'] == 'InsightFace_Swap')
        swap_rec = recall_score(y_true[swap_mask], y_pred[swap_mask], zero_division=0)

        sd_mask = (df_all['generator'] == 'StableDiffusion_v1.5')
        sd_rec = recall_score(y_true[sd_mask], y_pred[sd_mask], zero_division=0)

        style_mask = (df_all['generator'] == 'StyleGAN2')
        style_rec = recall_score(y_true[style_mask], y_pred[style_mask], zero_division=0)

        summary_rows.append({
            'Fusion_Strategy': strat_name,
            'ROC_AUC': round(auc, 4),
            'PR_AUC': round(prauc, 4),
            'Accuracy': round(acc, 4),
            'F1_Score': round(f1, 4),
            'Real_Specificity': round(spec, 4),
            'Real_FPR': round(fpr, 4),
            'Overall_Fake_Recall': round(rec, 4),
            'Gemini_Recall': round(gem_rec, 4),
            'Gemini_Mean_P': round(gem_mean_p, 4),
            'SD_Inpainting_Recall': round(inpaint_rec, 4),
            'InsightFace_Recall': round(swap_rec, 4),
            'SD_v15_Recall': round(sd_rec, 4),
            'StyleGAN2_Recall': round(style_rec, 4)
        })

    out_df = pd.DataFrame(summary_rows)
    out_csv = 'reports/phase12_fusion_ablation.csv'
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved fusion ablation to {out_csv}")
    print(out_df.to_string())

if __name__ == '__main__':
    run_fusion_ablation()
