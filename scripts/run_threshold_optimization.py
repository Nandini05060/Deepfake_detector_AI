"""
scripts/run_threshold_optimization.py
Sweeps decision thresholds on the VALIDATION SET (N=248) to find the optimal
operating threshold that achieves the >= 88% Real Specificity constraint
while maximizing Gemini and overall fake recall.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.train_phase15_gemini_generalization import (
    VerifiedFacesDataset, evaluate_model, LXDFDModel, get_device
)

def run_threshold_sweep():
    device = get_device()
    manifest_path = repo_root / "data" / "verified_faces" / "verified_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    val_ds = VerifiedFacesDataset(manifest_df, split="validation")
    test_ds = VerifiedFacesDataset(manifest_df, split="test")

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # 1. Load Baseline
    base_ckpt_path = repo_root / "checkpoints" / "lxfd_phase10_modern_generators_best.pt"
    base_model = LXDFDModel(fusion_type="attention", min_floor=0.25, pretrained=False).to(device)
    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    base_model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt))
    base_model.eval()

    # 2. Load Phase 15 Candidate
    cand_ckpt_path = repo_root / "experiments" / "phase15_gemini_generalization" / "lxfd_phase15_generalization_best.pt"
    cand_model = LXDFDModel(fusion_type="attention", min_floor=0.25, pretrained=False).to(device)
    cand_ckpt = torch.load(cand_ckpt_path, map_location=device, weights_only=False)
    cand_model.load_state_dict(cand_ckpt.get('model_state_dict', cand_ckpt))
    cand_model.eval()

    # Load temperatures
    base_temp = 1.6344
    cand_calib = repo_root / "experiments" / "phase15_gemini_generalization" / "calibration_params_phase15.json"
    with open(cand_calib, "r") as f:
        cand_temp = float(json.load(f)["temperature"])

    print("=" * 80)
    print(f"THRESHOLD OPTIMIZATION SWEEP ON VALIDATION SET (N={len(val_ds)})")
    print(f"Baseline Temp: {base_temp:.4f} | Candidate Temp: {cand_temp:.4f}")
    print("=" * 80)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    val_sweep_results = []

    for th in thresholds:
        b_val = evaluate_model(base_model, val_loader, device, temp=base_temp, threshold=th)
        c_val = evaluate_model(cand_model, val_loader, device, temp=cand_temp, threshold=th)
        
        val_sweep_results.append({
            "threshold": th,
            "baseline_val_spec": b_val["real_specificity"],
            "baseline_val_recall": b_val["fake_recall"],
            "baseline_val_acc": b_val["accuracy"],
            "baseline_inconc": b_val["inconclusive_rate"],
            "cand_val_spec": c_val["real_specificity"],
            "cand_val_recall": c_val["fake_recall"],
            "cand_val_acc": c_val["accuracy"],
            "cand_inconc": c_val["inconclusive_rate"]
        })
        print(f"Threshold: {th:.2f} | Base Spec: {b_val['real_specificity']*100:.1f}%, Rec: {b_val['fake_recall']*100:.1f}% | Cand Spec: {c_val['real_specificity']*100:.1f}%, Rec: {c_val['fake_recall']*100:.1f}%")

    sweep_df = pd.DataFrame(val_sweep_results)
    sweep_csv = repo_root / "reports" / "validation_threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nSaved validation sweep to {sweep_csv}")

    # Select optimal threshold for Candidate where Val Spec >= 88.0%
    valid_cands = [r for r in val_sweep_results if r["cand_val_spec"] >= 0.88]
    if valid_cands:
        best_cand_thresh = max(valid_cands, key=lambda x: x["cand_val_recall"])["threshold"]
        print(f"\n[OPTIMAL CANDIDATE THRESHOLD SELECTED FROM VALIDATION]: {best_cand_thresh:.2f}")
    else:
        # Fallback to threshold that yields highest specificity
        best_cand_thresh = max(val_sweep_results, key=lambda x: x["cand_val_spec"])["threshold"]
        print(f"\n[CANDIDATE THRESHOLD HIGHEST SPEC]: {best_cand_thresh:.2f} (Val Spec: {sweep_df.loc[sweep_df['threshold']==best_cand_thresh, 'cand_val_spec'].values[0]*100:.1f}%)")

    # Select optimal threshold for Baseline where Val Spec >= 88.0%
    valid_bases = [r for r in val_sweep_results if r["baseline_val_spec"] >= 0.88]
    best_base_thresh = max(valid_bases, key=lambda x: x["baseline_val_recall"])["threshold"] if valid_bases else 0.55
    print(f"[OPTIMAL BASELINE THRESHOLD SELECTED FROM VALIDATION]: {best_base_thresh:.2f}")

    # Evaluate BOTH models on the HELD-OUT TEST SET (N=450) using their validation-selected thresholds
    print("\n" + "=" * 80)
    print("EVALUATING SELECTED OPERATING THRESHOLDS ON HELD-OUT TEST SET (N=450)")
    print("=" * 80)
    b_test_opt = evaluate_model(base_model, test_loader, device, temp=base_temp, threshold=best_base_thresh)
    c_test_opt = evaluate_model(cand_model, test_loader, device, temp=cand_temp, threshold=best_cand_thresh)

    print(f"Metric                        Baseline (tau={best_base_thresh:.2f})    Candidate (tau={best_cand_thresh:.2f})")
    print(f"-------------------------------------------------------------------------")
    print(f"Test Decisive Accuracy        {b_test_opt['accuracy']*100:.2f}%                 {c_test_opt['accuracy']*100:.2f}%")
    print(f"Test Fake Recall              {b_test_opt['fake_recall']*100:.2f}%                 {c_test_opt['fake_recall']*100:.2f}%")
    print(f"Test Real Specificity         {b_test_opt['real_specificity']*100:.2f}%                 {c_test_opt['real_specificity']*100:.2f}%")
    print(f"Test Inconclusive Rate        {b_test_opt['inconclusive_rate']*100:.2f}%                 {c_test_opt['inconclusive_rate']*100:.2f}%")
    
    b_g = b_test_opt['per_generator'].get('Google_Gemini_Imagen3', {})
    c_g = c_test_opt['per_generator'].get('Google_Gemini_Imagen3', {})
    print(f"Gemini Fake Recall            {b_g.get('recall', 0)*100:.1f}%                  {c_g.get('recall', 0)*100:.1f}%")
    print(f"Gemini False-Real Certs       {b_g.get('false_real_errors', 0)} / 10                   {c_g.get('false_real_errors', 0)} / 10")

    b_inp = b_test_opt['per_generator'].get('StableDiffusion_Inpainting', {})
    c_inp = c_test_opt['per_generator'].get('StableDiffusion_Inpainting', {})
    print(f"SD Inpainting Recall          {b_inp.get('recall', 0)*100:.1f}%                  {c_inp.get('recall', 0)*100:.1f}%")
    print(f"SD Inpainting False-Reals     {b_inp.get('false_real_errors', 0)} / 200                {c_inp.get('false_real_errors', 0)} / 200")

    return {
        "validation_sweep": val_sweep_results,
        "selected_thresholds": {
            "baseline": best_base_thresh,
            "candidate": best_cand_thresh
        },
        "test_results": {
            "baseline": b_test_opt,
            "candidate": c_test_opt
        }
    }

if __name__ == "__main__":
    run_threshold_sweep()
