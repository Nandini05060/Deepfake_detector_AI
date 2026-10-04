import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.lxfd_model import LXDFDModel
from src.models.simple_cnn import SimpleCNN
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.evaluation.robustness import evaluate_robustness
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Run Systematic Robustness Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lxfd_attention_standard_best.pt")
    parser.add_argument("--test-csv", type=str, default="data/splits_v2/test.csv")
    parser.add_argument("--fusion", type=str, default="attention")
    parser.add_argument("--output-json", type=str, default="reports/robustness_results.json")
    parser.add_argument("--output-csv", type=str, default="reports/robustness_results.csv")
    args = parser.parse_args()

    device = get_device()
    test_path = Path(args.test_csv)
    if not test_path.exists():
        test_path = Path("data/splits/test.csv")

    if not test_path.exists():
        print(f"[ERROR] Test CSV not found at {args.test_csv} or data/splits/test.csv!")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at {args.checkpoint}!")
        sys.exit(1)

    print(f"[ROBUSTNESS] Evaluating checkpoint: {ckpt_path} on test split: {test_path}...")
    test_dataset = DeepfakeDataset(str(test_path), split='test')

    model = LXDFDModel(fusion_type=args.fusion, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)

    rob_data = evaluate_robustness(model, test_dataset, device=device)
    perturbations = rob_data.get("perturbation_results", {})

    table_rows = []
    for cond_name, metrics in perturbations.items():
        table_rows.append({
            "Condition": cond_name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "F1": metrics.get("f1", 0.0)
        })

    df_res = pd.DataFrame(table_rows)
    
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(rob_data, f, indent=2)

    df_res.to_csv(args.output_csv, index=False)

    print("\n==========================================================================================")
    print("                              ROBUSTNESS EVALUATION RESULTS                              ")
    print("==========================================================================================")
    print(f"{'Condition':<25} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 90)
    for row in table_rows[:15]:
        print(f"{row['Condition']:<25} | {row['Accuracy']:<10.4f} | {row['ROC-AUC']:<10.4f} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | {row['F1']:<10.4f}")
    print("==========================================================================================")
    print(f"[ROBUSTNESS] Results exported to {args.output_json} and {args.output_csv}\n")

if __name__ == "__main__":
    main()
