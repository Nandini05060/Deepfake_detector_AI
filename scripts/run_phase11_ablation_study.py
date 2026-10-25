"""
Phase 11 Scientific Ablation Study & Frequency Utility Evaluation
Compares Spatial-Only, Frequency-Only, Attention-Fused, Average-Fused,
and Branch Weightings across all generator families and Real controls.
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from src.models.lxfd_model import LXDFDModel
from src.data.face_preprocessor import YuNetFacePreprocessor
from src.utils.device import get_device

def evaluate_ablation():
    device = get_device()
    model = LXDFDModel(fusion_type='attention', pretrained=False).to(device)
    ckpt_path = "checkpoints/lxfd_phase10_modern_generators_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    preprocessor = YuNetFacePreprocessor(margin=0.50)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    manifest_df = pd.read_csv('data/manifests/phase11_generator_manifest.csv')
    
    # We evaluate on test splits: test_indomain, test_unseen_inpaint, test_locked, test_gemini_heldout
    test_splits = ['test_indomain', 'test_unseen_inpaint', 'test_locked', 'test_gemini_heldout']
    eval_df = manifest_df[manifest_df['split'].isin(test_splits)].copy()
    print(f"Total test evaluation samples: {len(eval_df)}")

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
            tensor_img = transform(crop_img).unsqueeze(0).to(device)

            with torch.no_grad():
                F_s = model.spatial_branch(tensor_img)
                F_f, dct_map = model.frequency_branch(tensor_img)

                # 1. Dual-Branch Attention Fused (Active Model)
                fused_att, w_dict = model.fusion(F_s, F_f)
                logit_att = model.classifier(fused_att).squeeze().item()
                p_att = 1.0 / (1.0 + np.exp(-logit_att))

                # 2. Spatial-Only Model (Feature F_s directly through classifier)
                logit_spatial = model.classifier(F_s).squeeze().item()
                p_spatial = 1.0 / (1.0 + np.exp(-logit_spatial))

                # 3. Frequency-Only Model (Feature F_f directly through classifier)
                logit_freq = model.classifier(F_f).squeeze().item()
                p_freq = 1.0 / (1.0 + np.exp(-logit_freq))

                # 4. Equal-Weight Average Fusion (0.5 * F_s + 0.5 * F_f)
                fused_avg = 0.5 * F_s + 0.5 * F_f
                logit_avg = model.classifier(fused_avg).squeeze().item()
                p_avg = 1.0 / (1.0 + np.exp(-logit_avg))

                w_s = w_dict['w_s'].item() if isinstance(w_dict['w_s'], torch.Tensor) else w_dict['w_s']
                w_f = w_dict['w_f'].item() if isinstance(w_dict['w_f'], torch.Tensor) else w_dict['w_f']

            records.append({
                'path': img_path,
                'generator': gen,
                'split': split,
                'label': label,
                'p_spatial': p_spatial,
                'p_freq': p_freq,
                'p_att': p_att,
                'p_avg': p_avg,
                'w_s': w_s,
                'w_f': w_f
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    df_out = pd.DataFrame(records)
    print(f"\nSuccessfully evaluated {len(df_out)} test samples.")

    # Calculate Metrics by Model Variant and Generator Family
    summary = []
    
    # 1. Overall Test Metrics
    for name, col in [('Spatial-Only', 'p_spatial'), ('Frequency-Only', 'p_freq'), ('Equal-Avg Fusion', 'p_avg'), ('Attention Fusion (Active)', 'p_att')]:
        y_true = df_out['label']
        y_prob = df_out[col]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)

        summary.append({
            'Model Configuration': name,
            'Subset': 'All Test Splits (N=457)',
            'ROC-AUC': round(auc, 4) if not np.isnan(auc) else 'N/A',
            'Accuracy': round(acc, 4),
            'F1-Score': round(f1, 4),
            'Precision': round(prec, 4),
            'Recall (Fake)': round(rec, 4),
            'Mean P(Fake)': round(float(y_prob.mean()), 4)
        })

    # 2. Generator-Specific Breakdowns
    print("\n=== GENERATOR-SPECIFIC COMPARISONS ===")
    for gen in df_out['generator'].unique():
        sub = df_out[df_out['generator'] == gen]
        y_true = sub['label']
        for name, col in [('Spatial-Only', 'p_spatial'), ('Frequency-Only', 'p_freq'), ('Attention Fusion', 'p_att')]:
            y_prob = sub[col]
            y_pred = (y_prob >= 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred)
            mean_prob = float(y_prob.mean())
            summary.append({
                'Model Configuration': name,
                'Subset': f"Generator: {gen} (N={len(sub)})",
                'ROC-AUC': 'N/A',
                'Accuracy': round(acc, 4),
                'F1-Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
                'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
                'Recall (Fake)': round(recall_score(y_true, y_pred, zero_division=0), 4) if y_true.iloc[0] == 1 else 'N/A (Real)',
                'Mean P(Fake)': round(mean_prob, 4)
            })
            print(f"[{gen[:20]:20s}] {name:18s} | Acc={acc*100:5.1f}% | Mean P(Fake)={mean_prob:.4f}")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv('reports/phase11_ablation_summary.csv', index=False)
    df_out.to_csv('reports/phase11_ablation_predictions.csv', index=False)
    print("\nSaved ablation summary to reports/phase11_ablation_summary.csv")

if __name__ == '__main__':
    evaluate_ablation()
