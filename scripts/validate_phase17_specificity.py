"""
Script: scripts/validate_phase17_specificity.py
Evaluates Phase 17 checkpoint using official LXDFDPredictor on:
1. Gemini failure diagnostic image across margins (0.15, 0.50, 0.75)
2. Full test set (N=543) across all 10 generator cohorts
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import cv2
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score
)

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.inference import LXDFDPredictor
from src.utils.device import get_device

def main():
    device = get_device()
    print(f"Device: {device}")

    ckpt_path = repo_root / "experiments/phase17_multigenerator_retraining/lxfd_phase17_best.pt"
    cal_path = repo_root / "experiments/phase17_multigenerator_retraining/calibration_params_phase17.json"

    predictor = LXDFDPredictor(
        checkpoint_path=str(ckpt_path),
        calibration_path=str(cal_path),
        device=device,
        decision_threshold=0.54,
        uncertainty_band=0.15,
        min_floor=0.25
    )

    print("=" * 80)
    print("PHASE 17 PREDICTOR EVALUATION (OFFICIAL LXDFD INFERENCE PIPELINE)")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Calibration Temp: {predictor.scaler.temperature:.4f} (Threshold: {predictor.decision_threshold})")
    print("=" * 80)

    # 1. Test gemini_failure.png across margins 0.15, 0.50, 0.75
    diag_path = repo_root / "data/diagnostics/gemini_failure.png"
    yunet_path = repo_root / "yunet.onnx"
    detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), score_threshold=0.5)

    pil_img = Image.open(diag_path).convert("RGB")
    w, h = pil_img.size
    detector.setInputSize((w, h))
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _, faces = detector.detect(bgr)

    print("\n[DIAGNOSTIC TEST] gemini_failure.png:")
    for margin in [0.15, 0.50, 0.75]:
        if faces is not None and len(faces) > 0:
            f = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            cx = fx + fw // 2
            cy = fy + fh // 2
            side = int(max(fw, fh) * (1.0 + 2.0 * margin))
            half_side = side // 2
            x1 = max(0, cx - half_side)
            y1 = max(0, cy - half_side)
            x2 = min(w, cx + half_side)
            y2 = min(h, cy + half_side)
            crop_img = pil_img.crop((x1, y1, x2, y2))
            face_meta = {"face_detected": True, "bbox": [x1, y1, x2, y2], "crop_margin": margin}
        else:
            min_dim = min(w, h)
            crop_img = pil_img.crop(((w-min_dim)//2, (h-min_dim)//2, (w-min_dim)//2+min_dim, (h-min_dim)//2+min_dim))
            face_meta = {"face_detected": False, "bbox": [0, 0, w, h], "crop_margin": margin}

        res = predictor.predict(crop_img, face_meta=face_meta)
        bo = res["branch_outputs"]
        print(f"  Margin m={margin:.2f}:")
        print(f"    Spatial Fake Prob: {bo['spatial_branch_fake_prob']:.4f}")
        print(f"    Freq Fake Prob:    {bo['frequency_branch_fake_prob']:.4f}")
        print(f"    Fused Fake Prob:   {res['fake_probability']:.4f}")
        print(f"    Discrepancy:       {bo['branch_discrepancy']:.4f}")
        print(f"    Final Decision:    {res['final_decision']} ({res['confidence_level']})")
        print(f"    Reliability:       {res['empirical_reliability']}")

    # 2. Test Full Test Split (N=543)
    manifest_path = repo_root / "experiments/phase17_multigenerator_retraining/data_manifest.csv"
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"].str.lower() == "test"].reset_index(drop=True)
    print(f"\n[TEST SPLIT EVALUATION] Evaluating {len(test_df)} held-out samples...")

    results = []
    for idx, row in test_df.iterrows():
        img_p = repo_root / row["relative_path"]
        lbl = 1 if str(row["label"]).upper() == "FAKE" else 0
        gen = row["generator_family"]

        try:
            pimg = Image.open(img_p).convert("RGB")
            pw, ph = pimg.size
            detector.setInputSize((pw, ph))
            pbgr = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
            _, pfaces = detector.detect(pbgr)

            margin = 0.50
            if pfaces is not None and len(pfaces) > 0:
                pf = max(pfaces, key=lambda x: x[2] * x[3])
                pfx, pfy, pfw, pfh = int(pf[0]), int(pf[1]), int(pf[2]), int(pf[3])
                pcx = pfx + pfw // 2
                pcy = pfy + pfh // 2
                pside = int(max(pfw, pfh) * (1.0 + 2.0 * margin))
                phalf_side = pside // 2
                px1 = max(0, pcx - phalf_side)
                py1 = max(0, pcy - phalf_side)
                px2 = min(pw, pcx + phalf_side)
                py2 = min(ph, pcy + phalf_side)
                crop_face = pimg.crop((px1, py1, px2, py2))
                face_meta = {"face_detected": True, "bbox": [px1, py1, px2, py2], "crop_margin": margin}
            else:
                pmin = min(pw, ph)
                crop_face = pimg.crop(((pw-pmin)//2, (ph-pmin)//2, (pw-pmin)//2+pmin, (ph-pmin)//2+pmin))
                face_meta = {"face_detected": False, "bbox": [0, 0, pw, ph], "crop_margin": margin}

            pred_res = predictor.predict(crop_face, face_meta=face_meta)
            p_bo = pred_res["branch_outputs"]

            results.append({
                "relative_path": row["relative_path"],
                "ground_truth": "FAKE" if lbl == 1 else "REAL",
                "ground_truth_int": lbl,
                "generator_family": gen,
                "spatial_prob": p_bo["spatial_branch_fake_prob"],
                "freq_prob": p_bo["frequency_branch_fake_prob"],
                "fused_prob": pred_res["fake_probability"],
                "discrepancy": p_bo["branch_discrepancy"],
                "verdict": pred_res["final_decision"],
                "is_inconclusive": (pred_res["final_decision"] == "INCONCLUSIVE"),
                "is_fake_pred": (pred_res["fake_probability"] >= predictor.decision_threshold)
            })
        except Exception as e:
            print(f"Error on {img_p}: {e}")

    res_df = pd.DataFrame(results)

    # Compute overall metrics
    all_y = res_df["ground_truth_int"].values
    all_p = res_df["fused_prob"].values
    bin_preds = res_df["is_fake_pred"].astype(int).values

    auc = float(roc_auc_score(all_y, all_p))
    pr_auc = float(average_precision_score(all_y, all_p))
    brier = float(brier_score_loss(all_y, all_p))

    # Confusion matrix binary
    cm_bin = confusion_matrix(all_y, bin_preds, labels=[0, 1])
    bin_tn, bin_fp, bin_fn, bin_tp = [int(v) for v in cm_bin.ravel()]

    bin_acc = (bin_tp + bin_tn) / len(all_y)
    bin_rec = bin_tp / (bin_tp + bin_fn)
    bin_spec = bin_tn / (bin_tn + bin_fp)
    bin_f1 = float(f1_score(all_y, bin_preds))

    # Decisive 3-way
    decisive_df = res_df[~res_df["is_inconclusive"]].copy()
    inconc_count = int(res_df["is_inconclusive"].sum())
    inconc_rate = inconc_count / len(res_df)
    coverage = len(decisive_df) / len(res_df)

    dec_tp = int(((decisive_df["ground_truth"] == "FAKE") & (decisive_df["verdict"] == "LIKELY_FAKE")).sum())
    dec_tn = int(((decisive_df["ground_truth"] == "REAL") & (decisive_df["verdict"] == "LIKELY_REAL")).sum())
    dec_fp = int(((decisive_df["ground_truth"] == "REAL") & (decisive_df["verdict"] == "LIKELY_FAKE")).sum())
    dec_fn = int(((decisive_df["ground_truth"] == "FAKE") & (decisive_df["verdict"] == "LIKELY_REAL")).sum())
    dec_total = len(decisive_df)

    dec_acc = (dec_tp + dec_tn) / max(1, dec_total)
    dec_rec = dec_tp / max(1, dec_tp + dec_fn)
    dec_spec = dec_tn / max(1, dec_tn + dec_fp)

    print("\n" + "=" * 80)
    print("PHASE 17 TEST SET RESULTS SUMMARY (OFFICIAL PREDICTOR)")
    print("=" * 80)
    print(f"Total Test Samples:        {len(res_df)} (Real: {(all_y==0).sum()}, Fake: {(all_y==1).sum()})")
    print(f"ROC-AUC:                   {auc:.4f}")
    print(f"PR-AUC:                    {pr_auc:.4f}")
    print(f"Brier Score:               {brier:.4f}")
    print(f"Binary Accuracy:           {bin_acc*100:.2f}% (Spec: {bin_spec*100:.2f}%, Recall: {bin_rec*100:.2f}%)")
    print(f"Decisive Total:            {dec_total} / {len(res_df)} (Coverage: {coverage*100:.2f}%)")
    print(f"Inconclusive Count:        {inconc_count} ({inconc_rate*100:.2f}%)")
    print(f"Decisive Accuracy:         {dec_acc*100:.2f}%")
    print(f"Decisive Fake Recall:      {dec_rec*100:.2f}% ({dec_tp}/{dec_tp+dec_fn})")
    print(f"Decisive Genuine Spec:     {dec_spec*100:.2f}% ({dec_tn}/{dec_tn+dec_fp})")
    print(f"Decisive Confusion Matrix: TN={dec_tn}, FP={dec_fp}, FN={dec_fn}, TP={dec_tp}")

    # Generator cohort breakdown
    print("\n" + "-" * 80)
    print("GENERATOR COHORT BREAKDOWN:")
    print("-" * 80)
    cohort_stats = {}
    for gen, gdf in res_df.groupby("generator_family"):
        gt = gdf["ground_truth"].iloc[0]
        n = len(gdf)
        mean_p = float(gdf["fused_prob"].mean())
        n_inconc = int(gdf["is_inconclusive"].sum())
        n_fake = int((gdf["verdict"] == "LIKELY_FAKE").sum())
        n_real = int((gdf["verdict"] == "LIKELY_REAL").sum())

        g_dec = gdf[~gdf["is_inconclusive"]]
        if gt == "REAL":
            corr = int((g_dec["verdict"] == "LIKELY_REAL").sum())
            g_acc = corr / max(1, len(g_dec))
            bin_corr = int((gdf["is_fake_pred"] == False).sum())
        else:
            corr = int((g_dec["verdict"] == "LIKELY_FAKE").sum())
            g_acc = corr / max(1, len(g_dec))
            bin_corr = int((gdf["is_fake_pred"] == True).sum())

        cohort_stats[gen] = {
            "ground_truth": gt,
            "total": n,
            "mean_fake_prob": round(mean_p, 4),
            "likely_fake_count": n_fake,
            "likely_real_count": n_real,
            "inconclusive_count": n_inconc,
            "inconclusive_rate": round(n_inconc / n, 4),
            "decisive_count": len(g_dec),
            "decisive_accuracy": round(g_acc, 4),
            "binary_accuracy": round(bin_corr / n, 4)
        }
        print(f"  * {gen:28s} [{gt:4s}] (N={n:3d}): Mean P={mean_p:.4f} | Decisive Acc={g_acc*100:5.1f}% | Fake={n_fake:2d} Real={n_real:2d} Inconc={n_inconc:2d}")

    # Save evaluation artifacts
    exp_dir = repo_root / "experiments/phase17_multigenerator_retraining"
    
    # Save per-image predictions CSV
    per_image_csv = exp_dir / "phase17_test_predictions.csv"
    res_df.to_csv(per_image_csv, index=False)
    print(f"\nSaved per-image predictions to {per_image_csv}")

    # Save summary json
    summary = {
        "metadata": {
            "model_version": "phase17",
            "model_name": "LX-DFD Phase 17 Multi-Generator Retrained",
            "checkpoint_path": "experiments/phase17_multigenerator_retraining/lxfd_phase17_best.pt",
            "calibrated_temperature": float(predictor.scaler.temperature),
            "calibrated_threshold": float(predictor.decision_threshold),
            "uncertainty_band": float(predictor.uncertainty_band),
            "min_floor": float(predictor.min_floor)
        },
        "overall_metrics": {
            "total_test_samples": len(res_df),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "binary_accuracy": round(bin_acc, 4),
            "binary_specificity": round(bin_spec, 4),
            "binary_recall": round(bin_rec, 4),
            "binary_f1": round(bin_f1, 4),
            "coverage": round(coverage, 4),
            "inconclusive_rate": round(inconc_rate, 4),
            "decisive_total": dec_total,
            "decisive_accuracy": round(dec_acc, 4),
            "decisive_recall": round(dec_rec, 4),
            "decisive_specificity": round(dec_spec, 4),
            "cm_binary": {"tn": bin_tn, "fp": bin_fp, "fn": bin_fn, "tp": bin_tp},
            "cm_decisive": {"tn": dec_tn, "fp": dec_fp, "fn": dec_fn, "tp": dec_tp}
        },
        "cohort_breakdown": cohort_stats
    }

    eval_json = exp_dir / "phase17_test_evaluation.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {eval_json}")

if __name__ == "__main__":
    main()
