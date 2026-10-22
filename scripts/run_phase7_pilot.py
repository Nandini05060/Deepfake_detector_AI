import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.face_preprocessor import YuNetFacePreprocessor
from src.frequency.dct import DiscreteCosineTransform2D
import torchvision.transforms as T
import torch

def run_pilot():
    print("=" * 80)
    print("          PHASE 7: FACE PREPROCESSING PILOT EVALUATION          ")
    print("=" * 80)

    preprocessor = YuNetFacePreprocessor(
        model_path="yunet.onnx",
        target_size=(224, 224),
        margin=0.25,
        score_threshold=0.60,
        version="phase7_v1"
    )

    out_dir = Path("data/pilot_phase7/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select balanced sample across multiple sources
    pilot_cohorts = [
        {
            "name": "IMDB-WIKI_Real",
            "source_dir": Path("data/unseen_generator_test/real"),
            "label": "REAL",
            "generator": "None",
            "source_dataset": "IMDB-WIKI",
            "count": 20
        },
        {
            "name": "FFHQ_Real",
            "source_dir": Path("Final Dataset/Real"),
            "label": "REAL",
            "generator": "None",
            "source_dataset": "FFHQ",
            "count": 20
        },
        {
            "name": "StyleGAN2_Fake",
            "source_dir": Path("Final Dataset/Fake"),
            "label": "FAKE",
            "generator": "StyleGAN2",
            "source_dataset": "Kaggle_DFD",
            "count": 20
        },
        {
            "name": "StableDiffusion_v1.5_Fake",
            "source_dir": Path("data/unseen_generator_test/StableDiffusion_v1.5"),
            "label": "FAKE",
            "generator": "StableDiffusion_v1.5",
            "source_dataset": "DeepFakeFace",
            "count": 20
        },
        {
            "name": "InsightFace_Swap_Fake",
            "source_dir": Path("data/unseen_generator_test/InsightFace_Swap"),
            "label": "FAKE",
            "generator": "InsightFace_Swap",
            "source_dataset": "DeepFakeFace",
            "count": 20
        }
    ]

    records = []
    dct = DiscreteCosineTransform2D(size=224, log_scale=True)
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for cohort in pilot_cohorts:
        src_files = sorted(list(cohort["source_dir"].glob("*.jpg")) + list(cohort["source_dir"].glob("*.png")))
        selected_files = src_files[:cohort["count"]]
        print(f"[*] Processing cohort: {cohort['name']} ({len(selected_files)} images)...")

        for idx, fpath in enumerate(selected_files):
            try:
                processed_pil, meta = preprocessor.process_image(fpath)
                
                # Check output properties
                w, h = processed_pil.size
                channels = len(processed_pil.getbands())
                is_valid_size = (w == 224 and h == 224 and channels == 3)

                # Save processed image
                out_filename = f"{cohort['name']}_{idx:03d}.jpg"
                out_path = out_dir / out_filename
                processed_pil.save(out_path, "JPEG", quality=95)

                # Compute high-frequency 2D DCT log-energy
                t_img = to_tensor(processed_pil).unsqueeze(0)
                dct_map = dct(t_img).squeeze().numpy()
                high_freq_energy = float(np.mean(dct_map[112:, 112:]))

                records.append({
                    "original_path": str(fpath.resolve()),
                    "output_path": str(out_path.resolve()),
                    "cohort": cohort["name"],
                    "label": cohort["label"],
                    "generator": cohort["generator"],
                    "source_dataset": cohort["source_dataset"],
                    "original_width": meta["original_width"],
                    "original_height": meta["original_height"],
                    "original_aspect_ratio": meta["original_aspect_ratio"],
                    "face_detection_status": meta["face_detection_status"],
                    "fallback_used": meta["fallback_used"],
                    "confidence_score": meta["confidence_score"],
                    "output_width": w,
                    "output_height": h,
                    "output_channels": channels,
                    "is_valid_size": is_valid_size,
                    "high_freq_dct_energy": round(high_freq_energy, 6),
                    "preprocessing_version": meta["preprocessing_version"]
                })
            except Exception as e:
                print(f"    [ERROR] Failed processing {fpath.name}: {e}")

    df_pilot = pd.DataFrame(records)
    pilot_csv = Path("data/pilot_phase7/pilot_metadata.csv")
    df_pilot.to_csv(pilot_csv, index=False)

    print("\n" + "=" * 80)
    print("                    PILOT RESULTS SUMMARY                       ")
    print("=" * 80)
    print(f"Total Processed: {len(df_pilot)}")
    print(f"Real Images:     {(df_pilot['label'] == 'REAL').sum()}")
    print(f"Fake Images:     {(df_pilot['label'] == 'FAKE').sum()}")
    print("\nDetection Status Breakdown by Cohort:")
    print(pd.crosstab(df_pilot["cohort"], df_pilot["face_detection_status"], margins=True))

    print("\nFallback Breakdown by Cohort:")
    print(pd.crosstab(df_pilot["cohort"], df_pilot["fallback_used"], margins=True))

    print("\nMean High-Frequency DCT Energy by Cohort:")
    print(df_pilot.groupby("cohort")["high_freq_dct_energy"].mean().to_string())

    print("\nOutput Shape Verification:")
    print(f"All images strictly (224, 224, 3): {df_pilot['is_valid_size'].all()}")
    print(f"Saved Metadata: {pilot_csv.resolve()}")
    print("=" * 80 + "\n")

    return df_pilot

if __name__ == "__main__":
    run_pilot()
