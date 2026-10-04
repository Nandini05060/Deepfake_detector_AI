import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.lxfd_model import LXDFDModel
from src.evaluation.calibration import TemperatureScaler, evaluate_calibration
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Calibrate Model Confidence Probabilities")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lxfd_attention_standard_best.pt")
    parser.add_argument("--val-csv", type=str, default="data/splits_v2/val.csv")
    parser.add_argument("--fusion", type=str, default="attention")
    parser.add_argument("--output-params", type=str, default="checkpoints/calibration_params.json")
    parser.add_argument("--output-json", type=str, default="reports/calibration_report.json")
    args = parser.parse_args()

    device = get_device()
    val_path = Path(args.val_csv)
    if not val_path.exists():
        val_path = Path("data/splits/val.csv")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"[CALIBRATE] Loading model checkpoint: {ckpt_path}")
    model = LXDFDModel(fusion_type=args.fusion, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_dataset = DeepfakeDataset(str(val_path), split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    val_logits = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            val_logits.extend(logits.cpu().numpy())
            val_targets.extend(lbls.cpu().numpy())

    val_logits = np.array(val_logits)
    val_targets = np.array(val_targets)
    raw_probs = 1.0 / (1.0 + np.exp(-val_logits))

    pre_cal_metrics = evaluate_calibration(val_targets, raw_probs)

    # Fit temperature T on validation set ONLY
    scaler = TemperatureScaler()
    t_opt = scaler.fit(val_logits, val_targets)
    scaler.save(args.output_params)

    cal_probs = scaler.calibrate_probs(raw_probs)
    post_cal_metrics = evaluate_calibration(val_targets, cal_probs)

    report = {
        "checkpoint": str(ckpt_path),
        "val_csv": str(val_path),
        "temperature": t_opt,
        "pre_calibration": pre_cal_metrics,
        "post_calibration": post_cal_metrics
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n==================================================")
    print("           CONFIDENCE CALIBRATION SUMMARY          ")
    print("==================================================")
    print(f"Optimal Temperature (T): {t_opt:.4f}")
    print(f"Pre-Calibration  -> Brier Score: {pre_cal_metrics['brier_score']:.4f} | ECE: {pre_cal_metrics['ece']:.4f} | Log Loss: {pre_cal_metrics['log_loss']:.4f}")
    print(f"Post-Calibration -> Brier Score: {post_cal_metrics['brier_score']:.4f} | ECE: {post_cal_metrics['ece']:.4f} | Log Loss: {post_cal_metrics['log_loss']:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
