"""
Phase 11 Gemini Failure Diagnostic & Sensitivity Analysis Suite
Evaluates Conditions A through L across verified Gemini/Imagen 3 samples
and Real control samples to isolate root causes of false negatives.
"""

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import io
import glob
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

from src.models.lxfd_model import LXDFDModel
from src.data.face_preprocessor import YuNetFacePreprocessor
from src.evaluation.calibration import TemperatureScaler
from src.utils.device import get_device

def load_model_and_scaler():
    device = get_device()
    model = LXDFDModel(fusion_type='attention', pretrained=False).to(device)
    ckpt_path = "checkpoints/lxfd_phase10_modern_generators_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    scaler = TemperatureScaler()
    cal_path = "checkpoints/calibration_params_phase10.json"
    if os.path.exists(cal_path):
        scaler.load(cal_path)
        calibrated = True
    else:
        calibrated = False
        
    return model, scaler, device, calibrated

def evaluate_tensor(model, scaler, tensor, device, calibrated):
    with torch.no_grad():
        tensor = tensor.unsqueeze(0).to(device)
        logit, meta = model(tensor)
        raw_logit = logit.item()
        
        w_s = meta['w_s'].item() if isinstance(meta['w_s'], torch.Tensor) else meta['w_s']
        w_f = meta['w_f'].item() if isinstance(meta['w_f'], torch.Tensor) else meta['w_f']
        s_logit = meta['spatial_logit'].item()
        f_logit = meta['frequency_logit'].item()
        s_prob = meta['spatial_fake_prob'].item()
        f_prob = meta['frequency_fake_prob'].item()
        
    raw_p_fake = 1.0 / (1.0 + np.exp(-raw_logit))
    if calibrated:
        cal_p_fake = float(scaler.calibrate_probs(np.array([raw_p_fake]))[0])
    else:
        cal_p_fake = float(raw_p_fake)
        
    p_fake = float(np.clip(cal_p_fake, 0.0, 1.0))
    p_real = float(1.0 - p_fake)
    pred_label = "FAKE" if p_fake >= 0.5 else "REAL"
    margin = float(abs(p_fake - 0.5))
    
    # Inconclusive check: branch disagreement > 0.15 or confidence within [0.40, 0.60]
    is_inconclusive = bool(abs(w_s - w_f) > 0.15 or (0.40 <= p_fake <= 0.60))
    
    return {
        'raw_logit': raw_logit,
        'p_fake': p_fake,
        'p_real': p_real,
        'pred_label': pred_label,
        'margin': margin,
        'spatial_logit': s_logit,
        'spatial_p_fake': s_prob,
        'frequency_logit': f_logit,
        'frequency_p_fake': f_prob,
        'w_s': w_s,
        'w_f': w_f,
        'is_inconclusive': is_inconclusive
    }

def main():
    os.makedirs('reports', exist_ok=True)
    model, scaler, device, calibrated = load_model_and_scaler()
    
    # Transform pipeline
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test images collection
    gemini_p11 = sorted(glob.glob('data/gemini_dataset/phase11/*.jpg'))
    gemini_heldout = sorted(glob.glob('data/gemini_dataset/test/*.jpg') + glob.glob('data/gemini_dataset/test/*.png'))
    real_controls = sorted(glob.glob('data/external/IMDB_WIKI_real/*.jpg'))[:5]
    
    test_samples = []
    for p in gemini_p11:
        test_samples.append({'path': p, 'label': 'FAKE', 'gen': 'Google_Gemini_Imagen3', 'group': 'p11_locked'})
    for p in gemini_heldout:
        test_samples.append({'path': p, 'label': 'FAKE', 'gen': 'Google_Gemini_Imagen3', 'group': 'p10_heldout'})
    for p in real_controls:
        test_samples.append({'path': p, 'label': 'REAL', 'gen': 'Authentic_Control', 'group': 'imdb_wiki_real'})
        
    print(f"Total diagnostic samples to evaluate: {len(test_samples)} ({len(gemini_p11) + len(gemini_heldout)} Gemini, {len(real_controls)} Real)")
    
    records = []
    
    for sample in test_samples:
        img_path = sample['path']
        label = sample['label']
        gen = sample['gen']
        sample_name = os.path.basename(img_path)
        
        pil_orig = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_orig.size
        
        # Base face detection with YuNet default (margin 0.50)
        preproc_default = YuNetFacePreprocessor(margin=0.50)
        pil_crop_default, meta_default = preproc_default.process_image(pil_orig)
        det_conf = meta_default.get('detector_confidence', 0.0) if meta_default.get('face_detected') else 0.0
        crop_box = meta_default.get('bounding_box_square', [0, 0, orig_w, orig_h])
        
        # Evaluate baseline Condition B (Default face crop)
        t_b = img_transform(pil_crop_default)
        res_b = evaluate_tensor(model, scaler, t_b, device, calibrated)
        baseline_verdict = res_b['pred_label']
        
        def add_rec(cond_id, cond_name, pil_img, res, custom_conf=None):
            verdict_changed = (res['pred_label'] != baseline_verdict)
            records.append({
                'sample_name': sample_name,
                'ground_truth': label,
                'generator_family': gen,
                'condition_id': cond_id,
                'condition_name': cond_name,
                'crop_width': pil_img.width,
                'crop_height': pil_img.height,
                'detector_confidence': custom_conf if custom_conf is not None else det_conf,
                'raw_logit': round(res['raw_logit'], 4),
                'p_real': round(res['p_real'], 4),
                'p_fake': round(res['p_fake'], 4),
                'pred_label': res['pred_label'],
                'decision_margin': round(res['margin'], 4),
                'verdict_changed': verdict_changed,
                'spatial_logit': round(res['spatial_logit'], 4),
                'spatial_p_fake': round(res['spatial_p_fake'], 4),
                'frequency_logit': round(res['frequency_logit'], 4),
                'frequency_p_fake': round(res['frequency_p_fake'], 4),
                'fusion_w_s': round(res['w_s'], 4),
                'fusion_w_f': round(res['w_f'], 4),
                'is_inconclusive': res['is_inconclusive']
            })

        # Condition B: Default face crop (margin 0.50)
        add_rec('B', 'Default Face Crop (margin=0.50)', pil_crop_default, res_b)
        
        # Condition A: Full image (direct resize)
        t_a = img_transform(pil_orig)
        res_a = evaluate_tensor(model, scaler, t_a, device, calibrated)
        add_rec('A', 'Full Image Uncropped', pil_orig, res_a, custom_conf=1.0)
        
        # Condition C: Generous context margin 0.75
        prep_c = YuNetFacePreprocessor(margin=0.75)
        crop_c, meta_c = prep_c.process_image(pil_orig)
        res_c = evaluate_tensor(model, scaler, img_transform(crop_c), device, calibrated)
        add_rec('C', 'Face Crop Context Margin 0.75', crop_c, res_c)
        
        # Condition D: Tight face crop margin 0.15
        prep_d = YuNetFacePreprocessor(margin=0.15)
        crop_d, meta_d = prep_d.process_image(pil_orig)
        res_d = evaluate_tensor(model, scaler, img_transform(crop_d), device, calibrated)
        add_rec('D', 'Tight Face Crop Margin 0.15', crop_d, res_d)
        
        # Condition E: Multiple controlled margins (0.05, 0.30, 0.45, 0.60)
        for m in [0.05, 0.30, 0.45, 0.60]:
            prep_e = YuNetFacePreprocessor(margin=m)
            crop_e, meta_e = prep_e.process_image(pil_orig)
            res_e = evaluate_tensor(model, scaler, img_transform(crop_e), device, calibrated)
            add_rec(f'E_m{int(m*100):02d}', f'Controlled Margin {m:.2f}', crop_e, res_e)
            
        # Condition F: Controlled Resizing (112x112 and 64x64 downsampling)
        for r in [112, 64]:
            down_crop = pil_crop_default.resize((r, r), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
            res_f = evaluate_tensor(model, scaler, img_transform(down_crop), device, calibrated)
            add_rec(f'F_res_{r}', f'Downsampled {r}x{r}', down_crop, res_f)
            
        # Condition G: Controlled JPEG Compression (Q=95, 75, 50, 25)
        for q in [95, 75, 50, 25]:
            buf = io.BytesIO()
            pil_crop_default.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            jpeg_crop = Image.open(buf).convert('RGB')
            res_g = evaluate_tensor(model, scaler, img_transform(jpeg_crop), device, calibrated)
            add_rec(f'G_jpeg_q{q}', f'JPEG Quality Q={q}', jpeg_crop, res_g)
            
        # Condition H: Face Interior Mask (Background replaced with neutral gray)
        # Create an elliptical mask over the central 70% of the face crop
        mask_h = Image.new('L', pil_crop_default.size, 0)
        draw_h = ImageDraw.Draw(mask_h)
        cw, ch = pil_crop_default.size
        draw_h.ellipse([int(cw*0.15), int(ch*0.10), int(cw*0.85), int(ch*0.90)], fill=255)
        gray_bg = Image.new('RGB', pil_crop_default.size, (128, 128, 128))
        interior_crop = Image.composite(pil_crop_default, gray_bg, mask_h)
        res_h = evaluate_tensor(model, scaler, img_transform(interior_crop), device, calibrated)
        add_rec('H', 'Face Interior (Background Masked)', interior_crop, res_h)
        
        # Condition I: Background Context Only (Face Occluded with neutral gray)
        mask_i = Image.new('L', pil_crop_default.size, 255)
        draw_i = ImageDraw.Draw(mask_i)
        draw_i.ellipse([int(cw*0.15), int(ch*0.10), int(cw*0.85), int(ch*0.90)], fill=0)
        context_crop = Image.composite(pil_crop_default, gray_bg, mask_i)
        res_i = evaluate_tensor(model, scaler, img_transform(context_crop), device, calibrated)
        add_rec('I', 'Background Only (Face Occluded)', context_crop, res_i)

    df_res = pd.DataFrame(records)
    csv_out = 'reports/phase11_gemini_failure_analysis.csv'
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved detailed failure analysis to {csv_out} ({len(df_res)} evaluated transformation conditions).")
    
    # Generate summary metrics
    print("\n=== SUMMARY ACROSS CONDITIONS FOR GEMINI SAMPLES ===")
    gemini_df = df_res[df_res['generator_family'] == 'Google_Gemini_Imagen3']
    for cond in df_res['condition_id'].unique():
        sub = gemini_df[gemini_df['condition_id'] == cond]
        mean_p_fake = sub['p_fake'].mean()
        fake_recall = (sub['pred_label'] == 'FAKE').mean()
        inconcl_rate = sub['is_inconclusive'].mean()
        cond_name = sub['condition_name'].iloc[0]
        print(f"Condition {cond:12s} ({cond_name[:30]:30s}): Recall={fake_recall*100:5.1f}% | Mean P(Fake)={mean_p_fake:.4f} | Inconclusive={inconcl_rate*100:4.1f}%")

if __name__ == '__main__':
    main()
