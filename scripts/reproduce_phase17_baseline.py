import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lxfd_model import LXDFDModel
from src.models.inference import LXDFDPredictor
from app.backend.main import extract_face_crop_with_meta

def get_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def evaluate_checkpoint_on_diagnostic(
    ckpt_path: Path,
    cal_path: Path,
    img_path: Path,
    margins: list = [0.15, 0.50, 0.75],
    device: torch.device = torch.device("cpu")
):
    ckpt_hash = get_file_sha256(ckpt_path)
    cal_temp = 1.0
    cal_thresh = 0.5
    if cal_path and cal_path.exists():
        with open(cal_path, "r") as f:
            cdata = json.load(f)
            cal_temp = float(cdata.get("temperature", 1.0))
            cal_thresh = float(cdata.get("optimal_threshold", 0.5))

    predictor = LXDFDPredictor(
        checkpoint_path=str(ckpt_path),
        calibration_path=str(cal_path) if cal_path and cal_path.exists() else None,
        device=device,
        min_floor=0.25
    )

    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size

    margin_results = {}

    for m in margins:
        face_img, meta = extract_face_crop_with_meta(pil_img, margin=m)
        tensor_img = predictor.transform(face_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out, model_meta = predictor.model(tensor_img)
            fused_logit = out[0].item() if out.dim() > 0 else out.item()
            raw_fake_p = 1.0 / (1.0 + np.exp(-fused_logit))

            spatial_p = model_meta.get("spatial_fake_prob", torch.tensor(0.5)).item()
            freq_p = model_meta.get("frequency_fake_prob", torch.tensor(0.5)).item()
            w_s = model_meta.get("w_s", 0.5)
            w_f = model_meta.get("w_f", 0.5)

            # Invert logits from probabilities (p = sigmoid(logit) -> logit = ln(p/(1-p)))
            spatial_logit = float(np.log(spatial_p / max(1e-7, (1.0 - spatial_p))))
            freq_logit = float(np.log(freq_p / max(1e-7, (1.0 - freq_p))))

        # Calibrate
        if predictor.calibration_enabled:
            cal_fake_p = float(predictor.scaler.calibrate_probs(np.array([raw_fake_p]))[0])
        else:
            cal_fake_p = float(raw_fake_p)

        pred_dict = predictor.predict(face_img, face_meta=meta)

        margin_results[f"margin_{m:.2f}"] = {
            "margin": m,
            "face_detected": meta.get("face_detected", False),
            "face_bbox": meta.get("face_bbox", []),
            "crop_coords": meta.get("crop_coords", []),
            "spatial_branch": {
                "logit": round(spatial_logit, 4),
                "probability": round(spatial_p, 4),
                "attention_weight_ws": round(float(w_s), 4)
            },
            "frequency_branch": {
                "logit": round(freq_logit, 4),
                "probability": round(freq_p, 4),
                "attention_weight_wf": round(float(w_f), 4)
            },
            "fused": {
                "logit": round(fused_logit, 4),
                "raw_probability": round(raw_fake_p, 4),
                "calibrated_fake_probability": round(cal_fake_p, 4),
                "calibrated_real_probability": round(1.0 - cal_fake_p, 4)
            },
            "branch_discrepancy": round(abs(spatial_p - freq_p), 4),
            "high_discrepancy_trigger": bool(abs(spatial_p - freq_p) > 0.10),
            "uncertainty_band_trigger": bool(abs(cal_fake_p - cal_thresh) <= 0.15),
            "decision_threshold": cal_thresh,
            "calibration_temperature": cal_temp,
            "uncertainty_status": pred_dict.get("is_uncertain", False),
            "empirical_reliability": pred_dict.get("empirical_reliability", "NORMAL"),
            "prediction": pred_dict["prediction"],
            "final_decision": pred_dict.get("final_decision", pred_dict["prediction"])
        }

    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "calibration_path": str(cal_path) if cal_path else None,
        "temperature": cal_temp,
        "threshold": cal_thresh,
        "margin_evaluations": margin_results
    }

def main():
    img_path = PROJECT_ROOT / "data/diagnostics/gemini_failure.png"
    img_hash = get_file_sha256(img_path)
    print(f"Auditing Diagnostic Image: {img_path}")
    print(f"SHA-256: {img_hash}")
    assert img_hash == "944cb50e9187a22a46dc6232387aa6769d4513c4654b141832b1eec04cd377fa"

    checkpoints_to_test = [
        {
            "name": "Phase 16 Unified Best",
            "ckpt": PROJECT_ROOT / "experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt",
            "cal": PROJECT_ROOT / "experiments/phase16_unified_generalization/calibration_params_phase16.json"
        },
        {
            "name": "Phase 15 Gemini Candidate",
            "ckpt": PROJECT_ROOT / "experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt",
            "cal": PROJECT_ROOT / "experiments/phase15_gemini_generalization/calibration_params_phase15.json"
        },
        {
            "name": "Phase 10 Modern Multi-Generator",
            "ckpt": PROJECT_ROOT / "checkpoints/lxfd_phase10_modern_generators_best.pt",
            "cal": PROJECT_ROOT / "checkpoints/calibration_params_phase10.json"
        },
        {
            "name": "Phase 8 Diverse Baseline",
            "ckpt": PROJECT_ROOT / "checkpoints/lxfd_multigen_diverse_best.pt",
            "cal": PROJECT_ROOT / "checkpoints/calibration_params.json"
        },
        {
            "name": "Phase 5 Attention Robustness v2",
            "ckpt": PROJECT_ROOT / "checkpoints/lxfd_attention_robustness_v2_best.pt",
            "cal": PROJECT_ROOT / "checkpoints/calibration_params.json"
        }
    ]

    all_results = {
        "diagnostic_image": {
            "filepath": str(img_path),
            "sha256": img_hash,
            "dimensions": [2816, 1536],
            "ground_truth": "FAKE",
            "generator": "Google Gemini Imagen 3"
        },
        "evaluations": {}
    }

    for cinfo in checkpoints_to_test:
        if cinfo["ckpt"].exists():
            print(f"\nEvaluating {cinfo['name']}...")
            res = evaluate_checkpoint_on_diagnostic(
                cinfo["ckpt"],
                cinfo["cal"],
                img_path,
                margins=[0.15, 0.50, 0.75]
            )
            all_results["evaluations"][cinfo["name"]] = res
            for m_key, m_val in res["margin_evaluations"].items():
                print(f"  {m_key}: P(Fake) = {m_val['fused']['calibrated_fake_probability']:.4f}, "
                      f"Decision: {m_val['prediction']}, Spatial: {m_val['spatial_branch']['probability']:.4f}, "
                      f"Freq: {m_val['frequency_branch']['probability']:.4f}, Discrepancy: {m_val['branch_discrepancy']:.4f}")

    out_json = PROJECT_ROOT / "reports/phase17_baseline_predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved predictions to {out_json}")

if __name__ == "__main__":
    main()
