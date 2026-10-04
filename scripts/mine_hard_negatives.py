import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.lxfd_model import LXDFDModel
from src.utils.device import get_device

def mine_hard_negatives(
    checkpoint_path: str = "checkpoints/lxfd_attention_standard_best.pt",
    dev_csv: str = "data/splits_v2/val.csv",
    output_json: str = "reports/hard_negatives.json",
    output_dir: str = "data/hard_negatives",
    confidence_threshold: float = 0.70
):
    device = get_device()
    dev_path = Path(dev_csv)
    if not dev_path.exists():
        dev_path = Path("data/splits/val.csv")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint {checkpoint_path} not found.")
        return None

    print(f"[MINING] Loading model for hard-negative extraction: {checkpoint_path}")
    model = LXDFDModel(fusion_type="attention", pretrained=False).to(device)
    checkpoint = torch.load(ckpt_file, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = DeepfakeDataset(str(dev_path), split='val')
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    hard_fake_as_real = [] # Fake predicted as REAL with high confidence
    hard_real_as_fake = [] # Real predicted as FAKE with high confidence

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].cpu().numpy()
            paths = batch['path']
            sources = batch.get('source', ['unknown'] * len(imgs))
            generators = batch.get('generator', ['unknown'] * len(imgs))

            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(lbls)):
                true_lbl = lbls[i]
                pred_prob = probs[i]
                img_path = paths[i]

                # Fake (1.0) predicted as REAL (prob < 1 - threshold)
                if true_lbl == 1.0 and pred_prob < (1.0 - confidence_threshold):
                    hard_fake_as_real.append({
                        "path": str(img_path),
                        "true_label": "FAKE",
                        "predicted_fake_probability": round(float(pred_prob), 4),
                        "confidence": round(float(1.0 - pred_prob), 4),
                        "source": sources[i],
                        "generator": generators[i]
                    })
                # Real (0.0) predicted as FAKE (prob > threshold)
                elif true_lbl == 0.0 and pred_prob > confidence_threshold:
                    hard_real_as_fake.append({
                        "path": str(img_path),
                        "true_label": "REAL",
                        "predicted_fake_probability": round(float(pred_prob), 4),
                        "confidence": round(float(pred_prob), 4),
                        "source": sources[i],
                        "generator": generators[i]
                    })

    summary = {
        "checkpoint": str(checkpoint_path),
        "dev_csv": str(dev_path),
        "confidence_threshold": confidence_threshold,
        "total_hard_negatives": len(hard_fake_as_real) + len(hard_real_as_fake),
        "fake_predicted_as_real_count": len(hard_fake_as_real),
        "real_predicted_as_fake_count": len(hard_real_as_fake),
        "hard_fake_as_real": hard_fake_as_real,
        "hard_real_as_fake": hard_real_as_fake
    }

    out_json_path = Path(output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    hard_pool_dir = Path(output_dir)
    hard_pool_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print("           HARD NEGATIVE MINING SUMMARY           ")
    print("==================================================")
    print(f"Total Hard Negative Candidates: {summary['total_hard_negatives']}")
    print(f"  - Fake predicted as REAL with High Conf: {summary['fake_predicted_as_real_count']}")
    print(f"  - Real predicted as FAKE with High Conf: {summary['real_predicted_as_fake_count']}")
    print(f"Metadata exported to: {out_json_path}")
    print(f"Hard negative pool initialized at: {hard_pool_dir}")
    print("==================================================\n")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Mine Hard Negatives from Development/Val Set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lxfd_attention_standard_best.pt")
    parser.add_argument("--dev-csv", type=str, default="data/splits_v2/val.csv")
    parser.add_argument("--output-json", type=str, default="reports/hard_negatives.json")
    parser.add_argument("--output-dir", type=str, default="data/hard_negatives")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    args = parser.parse_args()

    mine_hard_negatives(args.checkpoint, args.dev_csv, args.output_json, args.output_dir, args.confidence_threshold)

if __name__ == "__main__":
    main()
