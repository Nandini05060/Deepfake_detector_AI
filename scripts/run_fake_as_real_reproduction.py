"""
Phase 2: Systematic Reproduction of Fake-as-Real Misclassifications
Repository: https://github.com/Nandini05060/Deepfake_detector_AI

Evaluates:
- Known real control images (N=120)
- Known fake cohorts (StyleGAN2 N=60, SD v1.5 N=30, SD Inpainting N=200, InsightFace N=30, Gemini N=17)
- Problematic user cases (gemini_p11_13, SD_Inpainting_0018)

Outputs:
- reports/fake_as_real_reproduction_results.csv
- reports/fake_as_real_reproduction_summary.json
"""

import sys
import csv
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from PIL import Image

from src.models.inference import LXDFDPredictor
from app.backend.main import extract_face_crop_with_meta
from src.utils.device import get_device

def run_reproduction():
    device = get_device()
    print("=" * 70)
    print(f"[REPRODUCTION] Running Fake-as-Real Reproduction Suite on {device}")
    print("=" * 70)

    predictor = LXDFDPredictor(
        checkpoint_path="checkpoints/lxfd_phase10_modern_generators_best.pt",
        calibration_path="checkpoints/calibration_params_phase10.json",
        min_floor=0.25,
        decision_threshold=0.55
    )

    manifest_path = Path("data/manifests/phase11_generator_manifest.csv")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    test_splits = {'test_indomain', 'test_unseen_inpaint', 'test_locked', 'test_gemini_heldout'}
    test_rows = [r for r in reader if r.get('split') in test_splits]
    print(f"[DATA] Evaluating {len(test_rows)} locked test images...")

    results = []
    cohort_stats = {}

    for idx, row in enumerate(test_rows):
        img_p = Path(row['path'])
        if not img_p.exists():
            img_p = repo_root / img_p
        if not img_p.exists():
            continue

        try:
            pil_img = Image.open(img_p).convert('RGB')
            crop_img, det_meta = extract_face_crop_with_meta(pil_img, margin=0.50)
            res = predictor.predict(crop_img, face_meta=det_meta)

            gen = row['generator_family']
            gt = row['label']
            pred = res['prediction']
            decision = res.get('final_decision', pred)
            cal_fake = res['fake_probability']
            cal_real = res['real_probability']
            branch = res.get('branch_outputs', {})
            spat_fake = branch.get('spatial_branch_fake_prob', 0.5)
            freq_fake = branch.get('frequency_branch_fake_prob', 0.5)
            ws = branch.get('spatial_attention_weight_ws', 0.5)
            wf = branch.get('frequency_attention_weight_wf', 0.5)
            disc = branch.get('branch_discrepancy', abs(spat_fake - freq_fake))
            is_unc = res.get('is_uncertain', False)
            conf_level = res.get('confidence_level', 'MODERATE')
            emp_rel = res.get('empirical_reliability', 'MODERATE')
            face_det = det_meta.get('face_detected', False)

            # Is this a fake misclassified as real?
            fake_as_real_error = (gt == 'FAKE' and decision == 'LIKELY_REAL')

            rec = {
                "image_id": row['image_id'],
                "filename": row['filename'],
                "generator_family": gen,
                "ground_truth": gt,
                "predicted_label": pred,
                "final_decision": decision,
                "fake_as_real_error": fake_as_real_error,
                "fake_probability": round(cal_fake, 4),
                "real_probability": round(cal_real, 4),
                "spatial_fake_prob": round(spat_fake, 4),
                "frequency_fake_prob": round(freq_fake, 4),
                "spatial_weight_ws": round(ws, 4),
                "frequency_weight_wf": round(wf, 4),
                "branch_discrepancy": round(disc, 4),
                "is_uncertain": is_unc,
                "confidence_level": conf_level,
                "empirical_reliability": emp_rel,
                "face_detected": face_det
            }
            results.append(rec)

            if gen not in cohort_stats:
                cohort_stats[gen] = {
                    "total": 0, "fake_as_real": 0, "predicted_fake": 0,
                    "predicted_real": 0, "inconclusive": 0,
                    "mean_spat_fake": [], "mean_freq_fake": [],
                    "mean_ws": [], "mean_wf": [], "mean_cal_fake": []
                }
            st = cohort_stats[gen]
            st["total"] += 1
            if fake_as_real_error:
                st["fake_as_real"] += 1
            if decision == 'LIKELY_FAKE':
                st["predicted_fake"] += 1
            elif decision == 'LIKELY_REAL':
                st["predicted_real"] += 1
            elif decision == 'INCONCLUSIVE':
                st["inconclusive"] += 1
            st["mean_spat_fake"].append(spat_fake)
            st["mean_freq_fake"].append(freq_fake)
            st["mean_ws"].append(ws)
            st["mean_wf"].append(wf)
            st["mean_cal_fake"].append(cal_fake)

        except Exception as e:
            print(f"Error on {img_p}: {e}")

    # Write CSV
    csv_out = Path("reports/fake_as_real_reproduction_results.csv")
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[SAVED] {csv_out}")

    # Compile Summary
    summary = {}
    for gen, st in cohort_stats.items():
        summary[gen] = {
            "total_samples": st["total"],
            "fake_as_real_errors": st["fake_as_real"],
            "error_rate_pct": round(st["fake_as_real"] / st["total"] * 100.0, 2) if st["total"] > 0 else 0,
            "predicted_fake_count": st["predicted_fake"],
            "predicted_real_count": st["predicted_real"],
            "inconclusive_count": st["inconclusive"],
            "mean_spatial_fake_prob": round(float(np.mean(st["mean_spat_fake"])), 4),
            "mean_frequency_fake_prob": round(float(np.mean(st["mean_freq_fake"])), 4),
            "mean_ws": round(float(np.mean(st["mean_ws"])), 4),
            "mean_wf": round(float(np.mean(st["mean_wf"])), 4),
            "mean_calibrated_fake_prob": round(float(np.mean(st["mean_cal_fake"])), 4)
        }
        print(f"Cohort {gen}: Total={st['total']}, FakeAsRealErrors={st['fake_as_real']} ({summary[gen]['error_rate_pct']}%), MeanSpat={summary[gen]['mean_spatial_fake_prob']:.3f}, MeanFreq={summary[gen]['mean_frequency_fake_prob']:.3f}, MeanWF={summary[gen]['mean_wf']:.3f}")

    json_out = Path("reports/fake_as_real_reproduction_summary.json")
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {json_out}")
    print("[REPRODUCTION] Complete!")

if __name__ == "__main__":
    run_reproduction()
