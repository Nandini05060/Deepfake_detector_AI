"""
Phase 11 Shortcut Learning Diagnostic Suite
Evaluates Face-Only, Background-Only, Border-Only, Face-Occluded,
and Photometric shifts across balanced generator cohorts.
"""

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import io
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.lxfd_model import LXDFDModel
from src.data.face_preprocessor import YuNetFacePreprocessor
from src.utils.device import get_device

def evaluate_shortcuts():
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
    
    # Select 20 samples from each of 4 cohorts: StyleGAN2, SD 1.5, Gemini, Real
    cohorts = {
        'StyleGAN2': manifest_df[(manifest_df['generator_family'] == 'StyleGAN2') & (manifest_df['split'] == 'test_indomain')].head(25),
        'StableDiffusion_v1.5': manifest_df[(manifest_df['generator_family'] == 'StableDiffusion_v1.5') & (manifest_df['split'] == 'train')].head(25),
        'Google_Gemini_Imagen3': manifest_df[manifest_df['generator_family'] == 'Google_Gemini_Imagen3'].head(20),
        'None_Real': manifest_df[(manifest_df['generator_family'] == 'None_Real') & (manifest_df['split'] == 'test_indomain')].head(25)
    }

    test_samples = pd.concat(cohorts.values()).reset_index(drop=True)
    print(f"Total balanced shortcut diagnostic samples: {len(test_samples)}")

    records = []

    for idx, row in test_samples.iterrows():
        img_path = row['path']
        label = 1 if row['label'] == 'FAKE' else 0
        gen = row['generator_family']

        if not os.path.exists(img_path):
            continue

        try:
            pil_orig = Image.open(img_path).convert('RGB')
            base_crop, meta = preprocessor.process_image(pil_orig)
            cw, ch = base_crop.size

            # 1. Base Face Crop (Normal)
            # 2. Face Interior Only (Bg Masked)
            mask_interior = Image.new('L', (cw, ch), 0)
            draw_int = ImageDraw.Draw(mask_interior)
            draw_int.ellipse([int(cw*0.15), int(ch*0.10), int(cw*0.85), int(ch*0.90)], fill=255)
            gray_bg = Image.new('RGB', (cw, ch), (128, 128, 128))
            crop_interior = Image.composite(base_crop, gray_bg, mask_interior)

            # 3. Background Only (Face Masked)
            crop_background = Image.composite(gray_bg, base_crop, mask_interior)

            # 4. Border Only (Outer 15% ring of image)
            mask_border = Image.new('L', (cw, ch), 255)
            draw_bdr = ImageDraw.Draw(mask_border)
            draw_bdr.rectangle([int(cw*0.15), int(ch*0.15), int(cw*0.85), int(ch*0.85)], fill=0)
            crop_border = Image.composite(base_crop, gray_bg, mask_border)

            # 5. Face Center Occluded (Central 50% box blocked)
            mask_occl = Image.new('L', (cw, ch), 255)
            draw_occ = ImageDraw.Draw(mask_occl)
            draw_occ.rectangle([int(cw*0.25), int(ch*0.25), int(cw*0.75), int(ch*0.75)], fill=0)
            crop_occluded = Image.composite(base_crop, gray_bg, mask_occl)

            # 6. Brightness +30%
            crop_bright = ImageEnhance.Brightness(base_crop).enhance(1.3)

            # 7. Brightness -30%
            crop_dark = ImageEnhance.Brightness(base_crop).enhance(0.7)

            # 8. JPEG Compression Q=50
            buf = io.BytesIO()
            base_crop.save(buf, format='JPEG', quality=50)
            buf.seek(0)
            crop_jpeg50 = Image.open(buf).convert('RGB')

            variants = {
                'Baseline_Crop': base_crop,
                'Face_Interior_Only': crop_interior,
                'Background_Only': crop_background,
                'Border_Only': crop_border,
                'Face_Center_Occluded': crop_occluded,
                'Brightness_Plus30': crop_bright,
                'Brightness_Minus30': crop_dark,
                'JPEG_Quality_50': crop_jpeg50
            }

            for v_name, v_img in variants.items():
                tensor_v = transform(v_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logit, meta = model(tensor_v)
                    raw_l = logit.squeeze().item()
                    p_fake = 1.0 / (1.0 + np.exp(-raw_l))
                records.append({
                    'image_id': row['image_id'],
                    'generator': gen,
                    'label': label,
                    'variant': v_name,
                    'p_fake': p_fake,
                    'pred_fake': int(p_fake >= 0.5)
                })

        except Exception as e:
            print(f"Error on {img_path}: {e}")

    df_out = pd.DataFrame(records)
    df_out.to_csv('reports/phase11_shortcut_evaluations.csv', index=False)

    print("\n=== SHORTCUT VARIANT EVALUATION SUMMARY ===")
    summary = []
    for v_name in df_out['variant'].unique():
        sub = df_out[df_out['variant'] == v_name]
        y_true = sub['label']
        y_pred = sub['pred_fake']
        y_prob = sub['p_fake']
        
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        acc = accuracy_score(y_true, y_pred)
        mean_p_real = sub[sub['label'] == 0]['p_fake'].mean()
        mean_p_fake = sub[sub['label'] == 1]['p_fake'].mean()
        
        summary.append({
            'Perturbation / Shortcut Condition': v_name,
            'Overall Accuracy': round(acc, 4),
            'ROC-AUC': round(auc, 4),
            'Mean P(Fake) for Real': round(mean_p_real, 4),
            'Mean P(Fake) for Synthetic': round(mean_p_fake, 4),
            'Discrimination Gap': round(mean_p_fake - mean_p_real, 4)
        })
        print(f"Variant {v_name:22s} | Acc={acc*100:5.1f}% | AUC={auc:.4f} | Real P={mean_p_real:.4f} | Fake P={mean_p_fake:.4f} | Gap={mean_p_fake - mean_p_real:+.4f}")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv('reports/phase11_shortcut_metrics.csv', index=False)
    print("\nSaved summary to reports/phase11_shortcut_metrics.csv")

if __name__ == '__main__':
    evaluate_shortcuts()
