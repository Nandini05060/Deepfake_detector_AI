"""
Phase 12F: Robustness and Shortcut Analysis Suite
Evaluates 17 controlled transformations across balanced evaluation cohorts:
- Standard face crop (margin 0.50, baseline)
- Full image uncropped
- Wider face crop (margin 0.75)
- Tighter face crop (margin 0.15)
- Face interior only (background masked)
- Background only (face occluded)
- Resizing (112x112 downsample)
- JPEG compression (Q=95, Q=75, Q=50, Q=25)
- Brightness (+20%, -20%)
- Contrast (+20%, -20%)
- Mild Gaussian blur (radius 1.5)
- Central facial occlusion (40% central box masked)
Computes ROC-AUC, Recall, Specificity, FPR, FNR, Probability Shifts, and Prediction Flip Rates.
Outputs reports/phase12_robustness_metrics.csv and reports/phase12_robustness_audit.md.
"""

import os
import sys
import io
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from src.models.lxfd_model import LXDFDModel
from src.data.face_preprocessor import YuNetFacePreprocessor
from src.evaluation.calibration import TemperatureScaler
from src.utils.device import get_device

def evaluate_robustness():
    device = get_device()
    print(f"[Phase 12F] Robustness evaluation on: {device}")

    model = LXDFDModel(fusion_type='attention', pretrained=False).to(device)
    ckpt_path = "checkpoints/lxfd_phase10_modern_generators_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    scaler = TemperatureScaler()
    cal_path = "checkpoints/calibration_params_phase10.json"
    scaler.load(cal_path)

    preproc_base = YuNetFacePreprocessor(margin=0.50)
    preproc_wide = YuNetFacePreprocessor(margin=0.75)
    preproc_tight = YuNetFacePreprocessor(margin=0.15)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    manifest_df = pd.read_csv('data/manifests/phase11_generator_manifest.csv')
    
    # Balanced evaluation cohort: 25 StyleGAN2, 25 SD 1.5, 17 Gemini, 25 Inpainting, 25 InsightFace, 35 Real controls
    cohort_list = [
        manifest_df[(manifest_df['generator_family'] == 'StyleGAN2') & (manifest_df['split'] == 'test_indomain')].head(25),
        manifest_df[(manifest_df['generator_family'] == 'StableDiffusion_v1.5') & (manifest_df['split'] == 'test_indomain')].head(25),
        manifest_df[manifest_df['generator_family'] == 'Google_Gemini_Imagen3'], # all 17
        manifest_df[(manifest_df['generator_family'] == 'StableDiffusion_Inpainting') & (manifest_df['split'] == 'test_unseen_inpaint')].head(25),
        manifest_df[(manifest_df['generator_family'] == 'InsightFace_Swap') & (manifest_df['split'] == 'test_indomain')].head(25),
        manifest_df[(manifest_df['generator_family'] == 'None_Real') & (manifest_df['split'] == 'test_indomain')].head(35)
    ]
    eval_set = pd.concat(cohort_list).reset_index(drop=True)
    print(f"Total balanced robustness test samples: {len(eval_set)} ({sum(eval_set['label']=='FAKE')} Fake, {sum(eval_set['label']=='REAL')} Real)")

    def predict_pil(img_pil):
        t = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logit, _ = model(t)
            raw_l = logit.item()
        raw_p = 1.0 / (1.0 + np.exp(-raw_l))
        cal_p = float(scaler.calibrate_probs(np.array([raw_p]))[0])
        return cal_p

    eval_records = []

    for idx, row in eval_set.iterrows():
        img_path = row['path']
        lbl = 1 if row['label'] == 'FAKE' else 0
        gen = row['generator_family']

        if not os.path.exists(img_path):
            continue

        try:
            pil_orig = Image.open(img_path).convert('RGB')
            base_crop, meta_b = preproc_base.process_image(pil_orig)
            cw, ch = base_crop.size

            # Transformations:
            trans_dict = {}

            # 1. Baseline
            trans_dict['Standard Face Crop (m=0.50)'] = base_crop

            # 2. Full Image Uncropped
            trans_dict['Full Image Uncropped'] = pil_orig

            # 3. Wider Context (m=0.75)
            crop_w, _ = preproc_wide.process_image(pil_orig)
            trans_dict['Wider Face Crop (m=0.75)'] = crop_w

            # 4. Tighter Context (m=0.15)
            crop_t, _ = preproc_tight.process_image(pil_orig)
            trans_dict['Tighter Face Crop (m=0.15)'] = crop_t

            # 5. Face Interior Only (Background Masked)
            mask_int = Image.new('L', (cw, ch), 0)
            draw_i = ImageDraw.Draw(mask_int)
            draw_i.ellipse([int(cw*0.12), int(ch*0.08), int(cw*0.88), int(ch*0.92)], fill=255)
            gray_img = Image.new('RGB', (cw, ch), (128, 128, 128))
            trans_dict['Face Interior (Bg Masked)'] = Image.composite(base_crop, gray_img, mask_int)

            # 6. Background Context Only (Face Occluded)
            trans_dict['Background Only (Face Occluded)'] = Image.composite(gray_img, base_crop, mask_int)

            # 7. Resizing 112x112
            trans_dict['Resized 112x112'] = base_crop.resize((112, 112), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)

            # 8-11. JPEG Compression
            for q in [95, 75, 50, 25]:
                buf = io.BytesIO()
                base_crop.save(buf, format='JPEG', quality=q)
                buf.seek(0)
                trans_dict[f'JPEG Compression (Q={q})'] = Image.open(buf).convert('RGB')

            # 12-13. Brightness
            trans_dict['Brightness +20%'] = ImageEnhance.Brightness(base_crop).enhance(1.20)
            trans_dict['Brightness -20%'] = ImageEnhance.Brightness(base_crop).enhance(0.80)

            # 14-15. Contrast
            trans_dict['Contrast +20%'] = ImageEnhance.Contrast(base_crop).enhance(1.20)
            trans_dict['Contrast -20%'] = ImageEnhance.Contrast(base_crop).enhance(0.80)

            # 16. Mild Gaussian Blur
            trans_dict['Mild Gaussian Blur (sigma=1.5)'] = base_crop.filter(ImageFilter.GaussianBlur(radius=1.5))

            # 17. Central Facial Occlusion
            mask_occ = Image.new('L', (cw, ch), 255)
            draw_o = ImageDraw.Draw(mask_occ)
            draw_o.rectangle([int(cw*0.30), int(ch*0.30), int(cw*0.70), int(ch*0.70)], fill=0)
            trans_dict['Central Face Occlusion (40%)'] = Image.composite(base_crop, gray_img, mask_occ)

            # Evaluate each transformation for this image
            base_p = predict_pil(base_crop)
            base_pred = 1 if base_p >= 0.5 else 0

            for t_name, t_pil in trans_dict.items():
                p_val = predict_pil(t_pil)
                pred_val = 1 if p_val >= 0.5 else 0
                eval_records.append({
                    'path': img_path,
                    'label': lbl,
                    'generator': gen,
                    'transformation': t_name,
                    'p_fake': p_val,
                    'pred': pred_val,
                    'baseline_p': base_p,
                    'baseline_pred': base_pred,
                    'flipped': (pred_val != base_pred),
                    'shift': (p_val - base_p)
                })

        except Exception as e:
            print(f"Error on {img_path}: {e}")

    df_res = pd.DataFrame(eval_records)
    print(f"\nEvaluated {len(df_res)} total transformation instances.")

    summary = []
    base_sub = df_res[df_res['transformation'] == 'Standard Face Crop (m=0.50)']
    base_auc = roc_auc_score(base_sub['label'], base_sub['p_fake'])

    for t_name in df_res['transformation'].unique():
        sub = df_res[df_res['transformation'] == t_name]
        y_true = sub['label'].values
        y_prob = sub['p_fake'].values
        y_pred = sub['pred'].values

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        
        real_mask = (y_true == 0)
        spec = accuracy_score(y_true[real_mask], y_pred[real_mask])
        fpr = 1.0 - spec

        fake_mask = (y_true == 1)
        rec = recall_score(y_true[fake_mask], y_pred[fake_mask], zero_division=0)
        fnr = 1.0 - rec

        flip_rate = float(sub['flipped'].mean())
        mean_shift = float(sub['shift'].mean())
        mean_p = float(y_prob.mean())

        # Gemini specific recall
        gem_sub = sub[sub['generator'] == 'Google_Gemini_Imagen3']
        gem_rec = float((gem_sub['pred'] == 1).mean()) if len(gem_sub) > 0 else 0.0

        summary.append({
            'Transformation': t_name,
            'Sample_Count': len(sub),
            'ROC_AUC': round(auc, 4) if not np.isnan(auc) else 'N/A',
            'ROC_AUC_Delta': round(auc - base_auc, 4) if not np.isnan(auc) else 'N/A',
            'Fake_Recall': round(rec, 4),
            'Real_Specificity': round(spec, 4),
            'Real_FPR': round(fpr, 4),
            'Synthetic_FNR': round(fnr, 4),
            'Gemini_Recall': round(gem_rec, 4),
            'Mean_P_Fake': round(mean_p, 4),
            'Mean_Prob_Shift': round(mean_shift, 4),
            'Prediction_Flip_Rate': round(flip_rate, 4)
        })

    sum_df = pd.DataFrame(summary)
    sum_csv = 'reports/phase12_robustness_metrics.csv'
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nSaved robustness metrics to {sum_csv}")
    print(sum_df[['Transformation', 'ROC_AUC', 'Fake_Recall', 'Real_Specificity', 'Real_FPR', 'Prediction_Flip_Rate']].to_string())

if __name__ == '__main__':
    evaluate_robustness()
