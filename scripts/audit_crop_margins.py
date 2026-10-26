import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import LXDFDPredictor
from app.backend.main import extract_face_crop_with_meta

def audit_single_image_diagnostic(image_path: Path):
    """
    Audits the specific failure case image (gemini_failure.png) across
    margins 0.15, 0.50, and 0.75 for Phase 10, Phase 15, and Phase 16.
    """
    print(f"\n=======================================================")
    print(f"--- 1. SINGLE-IMAGE DIAGNOSTIC CASE STUDY ---")
    print(f"Image: {image_path}")
    print(f"=======================================================")
    
    if not image_path.exists():
        print(f"ERROR: Diagnostic image not found at {image_path}")
        return None

    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    print(f"Original Dimensions: {w}x{h}")

    models = {
        "phase10": {
            "name": "Phase 10 Modern Multi-Generator",
            "ckpt": "checkpoints/lxfd_phase10_modern_generators_best.pt",
            "cal": "checkpoints/calibration_params_phase10.json"
        },
        "phase15": {
            "name": "Phase 15 Gemini Generalization Candidate",
            "ckpt": "experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt",
            "cal": "experiments/phase15_gemini_generalization/calibration_params_phase15.json"
        },
        "phase16": {
            "name": "Phase 16 Unified Generalization Best",
            "ckpt": "experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt",
            "cal": "experiments/phase16_unified_generalization/calibration_params_phase16.json"
        }
    }

    margins = [0.15, 0.50, 0.75]
    single_image_results = {}

    for model_id, m_info in models.items():
        predictor = LXDFDPredictor(
            checkpoint_path=m_info["ckpt"],
            calibration_path=m_info["cal"] if Path(m_info["cal"]).exists() else None,
            device=torch.device("cpu"),
            min_floor=0.25
        )
        single_image_results[model_id] = {}

        for m in margins:
            face_img, meta = extract_face_crop_with_meta(pil_img, margin=m)
            res = predictor.predict(face_img, face_meta=meta)
            single_image_results[model_id][f"margin_{m:.2f}"] = {
                "margin": m,
                "face_detected": meta.get("face_detected", False),
                "crop_coords": meta.get("crop_coords", []),
                "prediction": res["prediction"],
                "final_decision": res.get("final_decision", res["prediction"]),
                "calibrated_fake_prob": round(res["fake_probability"], 4),
                "calibrated_real_prob": round(res["real_probability"], 4),
                "raw_fake_prob": round(res.get("raw_fake_prob", res["fake_probability"]), 4),
                "spatial_prob": round(res.get("branch_outputs", {}).get("spatial_fake_prob", 0.5), 4),
                "freq_prob": round(res.get("branch_outputs", {}).get("frequency_fake_prob", 0.5), 4),
                "w_s": round(res.get("branch_outputs", {}).get("w_s", 0.5), 4),
                "w_f": round(res.get("branch_outputs", {}).get("w_f", 0.5), 4),
            }
            print(f"[{m_info['name']}] Margin {m:.2f} -> Decision: {res['prediction']}, P(Fake): {res['fake_probability']:.4f}, Spatial: {single_image_results[model_id][f'margin_{m:.2f}']['spatial_prob']:.4f}, Freq: {single_image_results[model_id][f'margin_{m:.2f}']['freq_prob']:.4f}")

    return single_image_results

def audit_aggregate_crop_sensitivity(manifest_path: Path):
    """
    Evaluates aggregate crop-margin sensitivity (m=0.15, 0.50, 0.75) across
    all 448 test images using the active Phase 16 unified checkpoint.
    Separates fake detection recall from genuine real-image false positive rate (FPR).
    """
    print(f"\n=======================================================")
    print(f"--- 2. AGGREGATE CROP-MARGIN EVALUATION ON TEST SET (N=448) ---")
    print(f"=======================================================")

    import csv
    test_rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"].upper() == "TEST":
                test_rows.append(row)

    print(f"Loaded {len(test_rows)} test images from {manifest_path}")

    # Active Phase 16 model
    ckpt_path = "experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt"
    cal_path = "experiments/phase16_unified_generalization/calibration_params_phase16.json"
    predictor = LXDFDPredictor(
        checkpoint_path=ckpt_path,
        calibration_path=cal_path,
        device=torch.device("cpu"),
        min_floor=0.25
    )

    margins = [0.15, 0.50, 0.75]
    margin_eval = {}

    for m in margins:
        print(f"\n--- Evaluating Margin m = {m:.2f} ---")
        start_t = time.time()
        
        # Track counts
        total_real = 0
        total_fake = 0
        
        # Binary evaluation counts (tau = 0.57)
        bin_tn = 0
        bin_fp = 0
        bin_fn = 0
        bin_tp = 0
        
        # 3-way evaluation counts
        tri_tn = 0
        tri_fp = 0
        tri_fn = 0
        tri_tp = 0
        tri_inconc_real = 0
        tri_inconc_fake = 0
        
        # Per-generator tracking
        gen_stats = {}

        face_detected_count = 0

        for row in test_rows:
            img_path = PROJECT_ROOT / row["relative_path"]
            gt = row["label"].upper() # REAL or FAKE
            gen = row["generator_family"]

            if gen not in gen_stats:
                gen_stats[gen] = {
                    "total": 0, "gt": gt,
                    "fake_probs": [],
                    "bin_tp": 0, "bin_tn": 0, "bin_fp": 0, "bin_fn": 0,
                    "decisive_tp": 0, "decisive_tn": 0, "decisive_fp": 0, "decisive_fn": 0,
                    "inconclusive": 0
                }
            gen_stats[gen]["total"] += 1

            if gt == "REAL":
                total_real += 1
            else:
                total_fake += 1

            pil_img = Image.open(img_path).convert("RGB")
            face_img, meta = extract_face_crop_with_meta(pil_img, margin=m)
            if meta.get("face_detected", False):
                face_detected_count += 1

            res = predictor.predict(face_img, face_meta=meta)
            p_fake = res["fake_probability"]
            decision = res["prediction"]
            gen_stats[gen]["fake_probs"].append(p_fake)

            # Binary threshold 0.57
            bin_pred = "FAKE" if p_fake >= 0.57 else "REAL"
            if gt == "REAL":
                if bin_pred == "REAL":
                    bin_tn += 1
                    gen_stats[gen]["bin_tn"] += 1
                else:
                    bin_fp += 1
                    gen_stats[gen]["bin_fp"] += 1
            else: # FAKE
                if bin_pred == "FAKE":
                    bin_tp += 1
                    gen_stats[gen]["bin_tp"] += 1
                else:
                    bin_fn += 1
                    gen_stats[gen]["bin_fn"] += 1

            # 3-Way policy
            if decision == "INCONCLUSIVE":
                gen_stats[gen]["inconclusive"] += 1
                if gt == "REAL":
                    tri_inconc_real += 1
                else:
                    tri_inconc_fake += 1
            elif decision == "REAL":
                if gt == "REAL":
                    tri_tn += 1
                    gen_stats[gen]["decisive_tn"] += 1
                else:
                    tri_fn += 1
                    gen_stats[gen]["decisive_fn"] += 1
            elif decision == "FAKE":
                if gt == "FAKE":
                    tri_tp += 1
                    gen_stats[gen]["decisive_tp"] += 1
                else:
                    tri_fp += 1
                    gen_stats[gen]["decisive_fp"] += 1

        elapsed = time.time() - start_t
        decisive_real = tri_tn + tri_fp
        decisive_fake = tri_tp + tri_fn
        total_decisive = decisive_real + decisive_fake
        total_inconc = tri_inconc_real + tri_inconc_fake

        # Binary metrics (Denominator: 448)
        bin_acc = (bin_tn + bin_tp) / (total_real + total_fake)
        bin_spec = bin_tn / total_real if total_real > 0 else 0.0
        bin_recall = bin_tp / total_fake if total_fake > 0 else 0.0
        bin_fpr = bin_fp / total_real if total_real > 0 else 0.0

        # Decisive metrics
        dec_acc = (tri_tn + tri_tp) / total_decisive if total_decisive > 0 else 0.0
        dec_spec = tri_tn / decisive_real if decisive_real > 0 else 0.0
        dec_recall = tri_tp / decisive_fake if decisive_fake > 0 else 0.0
        dec_fpr = tri_fp / decisive_real if decisive_real > 0 else 0.0

        # Per generator summary
        per_gen_summary = {}
        for gen, s in gen_stats.items():
            mean_p = float(np.mean(s["fake_probs"]))
            dec_total = s["decisive_tp"] + s["decisive_tn"] + s["decisive_fp"] + s["decisive_fn"]
            if s["gt"] == "REAL":
                dec_acc_g = s["decisive_tn"] / dec_total if dec_total > 0 else 0.0
                bin_acc_g = s["bin_tn"] / s["total"] if s["total"] > 0 else 0.0
            else:
                dec_acc_g = s["decisive_tp"] / dec_total if dec_total > 0 else 0.0
                bin_acc_g = s["bin_tp"] / s["total"] if s["total"] > 0 else 0.0

            per_gen_summary[gen] = {
                "total": s["total"],
                "ground_truth": s["gt"],
                "mean_fake_prob": round(mean_p, 4),
                "inconclusive_count": s["inconclusive"],
                "inconclusive_rate": round(s["inconclusive"] / s["total"], 4),
                "decisive_count": dec_total,
                "decisive_accuracy": round(dec_acc_g, 4),
                "binary_accuracy": round(bin_acc_g, 4)
            }

        margin_eval[f"margin_{m:.2f}"] = {
            "margin": m,
            "face_detection_rate": round(face_detected_count / len(test_rows), 4),
            "binary_all_samples": {
                "total_samples": len(test_rows),
                "tn": bin_tn,
                "fp": bin_fp,
                "fn": bin_fn,
                "tp": bin_tp,
                "accuracy": round(bin_acc, 4),
                "fake_recall": round(bin_recall, 4),
                "real_specificity": round(bin_spec, 4),
                "false_positive_rate": round(bin_fpr, 4)
            },
            "decisive_3way_policy": {
                "coverage": round(total_decisive / len(test_rows), 4),
                "inconclusive_rate": round(total_inconc / len(test_rows), 4),
                "decisive_samples": total_decisive,
                "inconclusive_samples": total_inconc,
                "tn": tri_tn,
                "fp": tri_fp,
                "fn": tri_fn,
                "tp": tri_tp,
                "inconclusive_real": tri_inconc_real,
                "inconclusive_fake": tri_inconc_fake,
                "accuracy": round(dec_acc, 4),
                "fake_recall": round(dec_recall, 4),
                "real_specificity": round(dec_spec, 4),
                "false_positive_rate": round(dec_fpr, 4)
            },
            "per_generator": per_gen_summary,
            "elapsed_seconds": round(elapsed, 2)
        }

        print(f"Margin {m:.2f}: Binary Acc={bin_acc:.4f}, Fake Recall={bin_recall:.4f}, Real Spec={bin_spec:.4f}, FPR={bin_fpr:.4f}")
        print(f"Margin {m:.2f}: Decisive Acc={dec_acc:.4f}, Fake Recall={dec_recall:.4f}, Real Spec={dec_spec:.4f}, Coverage={total_decisive/len(test_rows):.4f}")

    return margin_eval

def main():
    diagnostic_img = PROJECT_ROOT / "data/diagnostics/gemini_failure.png"
    manifest_csv = PROJECT_ROOT / "reports/phase16_generator_split_manifest.csv"

    single_results = audit_single_image_diagnostic(diagnostic_img)
    aggregate_results = audit_aggregate_crop_sensitivity(manifest_csv)

    combined_crop_audit = {
        "audit_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_evaluated": "LX-DFD Phase 16 Unified Best (experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt)",
            "detector": "OpenCV YuNet ONNX (yunet.onnx)",
            "test_set_size": 448
        },
        "single_image_case_study": single_results,
        "aggregate_crop_sensitivity": aggregate_results
    }

    out_json = PROJECT_ROOT / "reports/crop_margin_audit_recomputed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_crop_audit, f, indent=2)
    print(f"\nSaved combined crop audit to {out_json}")

if __name__ == "__main__":
    main()
