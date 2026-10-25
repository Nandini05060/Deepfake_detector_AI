"""
Direct Inference Comparison: Phase 10 vs Phase 15
Evaluates held-out Gemini faces, genuine real faces (FFHQ & IMDB-WIKI), and diffusion/GAN fakes.
Records Ground Truth, Predicted Label, Real Prob, Fake Prob, Active Checkpoint, and 3-way Decision.
"""

import json
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

import pandas as pd
from PIL import Image
import torch

from src.models.inference import LXDFDPredictor
from src.utils.device import get_device

def main():
    repo_root = Path(__file__).resolve().parent.parent
    device = get_device()
    manifest_path = repo_root / "data" / "verified_faces" / "verified_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    # 1. Initialize Phase 10 Predictor
    p10_ckpt = repo_root / "checkpoints" / "lxfd_phase10_modern_generators_best.pt"
    p10_cal = repo_root / "checkpoints" / "calibration_params_phase10.json"
    predictor_p10 = LXDFDPredictor(
        checkpoint_path=str(p10_ckpt),
        calibration_path=str(p10_cal),
        device=device,
        min_floor=0.25
    )
    predictor_p10.model_version_name = "LX-DFD Phase 10 (Production)"

    # 2. Initialize Phase 15 Predictor
    p15_ckpt = repo_root / "experiments" / "phase15_gemini_generalization" / "lxfd_phase15_generalization_best.pt"
    p15_cal = repo_root / "experiments" / "phase15_gemini_generalization" / "calibration_params_phase15.json"
    predictor_p15 = LXDFDPredictor(
        checkpoint_path=str(p15_ckpt),
        calibration_path=str(p15_cal),
        device=device,
        min_floor=0.25
    )
    predictor_p15.model_version_name = "LX-DFD Phase 15 (Candidate)"

    # Select test cohorts
    test_df = manifest_df[manifest_df["split"] == "test"]
    gemini_samples = test_df[test_df["generator_family"] == "Google_Gemini_Imagen3"]
    ffhq_real_samples = test_df[test_df["generator_family"] == "FFHQ_Real"].head(10)
    imdb_real_samples = test_df[test_df["generator_family"] == "IMDB_WIKI"].head(10)
    stylegan_samples = test_df[test_df["generator_family"] == "StyleGAN2"].head(5)
    sd15_samples = test_df[test_df["generator_family"] == "StableDiffusion_v1.5"].head(5)

    eval_cohorts = [
        ("Gemini Imagen 3 (Held-Out FAKE)", gemini_samples, "FAKE"),
        ("FFHQ Genuine Photographs (REAL)", ffhq_real_samples, "REAL"),
        ("IMDB-WIKI Real Faces (Challenging REAL)", imdb_real_samples, "REAL"),
        ("StyleGAN2 Synthetics (In-Domain FAKE)", stylegan_samples, "FAKE"),
        ("Stable Diffusion v1.5 (In-Domain FAKE)", sd15_samples, "FAKE"),
    ]

    results = []

    print("=" * 110)
    print(f"{'Cohort':<26} | {'Image':<28} | {'GT':<5} | {'P10 Pred':<12} | {'P10 Fake%':<9} | {'P15 Pred':<12} | {'P15 Fake%':<9} | {'Status'}")
    print("-" * 110)

    for cohort_name, cohort_df, gt_label in eval_cohorts:
        for _, row in cohort_df.iterrows():
            img_rel_path = row["relative_path"]
            img_full_path = repo_root / img_rel_path
            if not img_full_path.exists():
                continue

            img = Image.open(img_full_path).convert("RGB")

            # Phase 10 Inference
            res_p10 = predictor_p10.predict(img)
            # Phase 15 Inference
            res_p15 = predictor_p15.predict(img)

            img_name = Path(img_rel_path).name
            status = "RESOLVED" if (gt_label == "FAKE" and res_p10["final_decision"] != "LIKELY_FAKE" and res_p15["final_decision"] == "LIKELY_FAKE") else "CONGRUENT"

            print(f"{cohort_name[:26]:<26} | {img_name[:28]:<28} | {gt_label:<5} | "
                  f"{res_p10['final_decision']:<12} | {res_p10['fake_probability']*100:6.1f}%   | "
                  f"{res_p15['final_decision']:<12} | {res_p15['fake_probability']*100:6.1f}%   | "
                  f"{status}")

            results.append({
                "cohort": cohort_name,
                "image_name": img_name,
                "relative_path": img_rel_path,
                "ground_truth": gt_label,
                "phase10": {
                    "checkpoint": "lxfd_phase10_modern_generators_best.pt",
                    "predicted_label": res_p10["prediction"],
                    "verdict": res_p10["final_decision"],
                    "real_probability": res_p10["real_probability"],
                    "fake_probability": res_p10["fake_probability"],
                    "is_uncertain": res_p10["is_uncertain"]
                },
                "phase15": {
                    "checkpoint": "lxfd_phase15_generalization_best.pt",
                    "predicted_label": res_p15["prediction"],
                    "verdict": res_p15["final_decision"],
                    "real_probability": res_p15["real_probability"],
                    "fake_probability": res_p15["fake_probability"],
                    "is_uncertain": res_p15["is_uncertain"]
                }
            })

    print("=" * 110)

    # Save detailed JSON report
    out_dir = repo_root / "reports" / "phase15_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "p10_vs_p15_comparison.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[REPORT SAVED] Full comparative evaluation logged to: {out_json}")

if __name__ == "__main__":
    main()
