import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.lxfd_model import LXDFDModel
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.simple_cnn import SimpleCNN
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.calibration import compute_ece, compute_brier_score
from src.utils.device import get_device

class ValidatedImageDataset(Dataset):
    """
    Evaluation dataset with strict sample validation.
    Never silently replaces invalid images with zero/black tensors.
    """
    def __init__(self, samples: List[dict], image_size: Tuple[int, int] = (224, 224)):
        self.samples = samples
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        img_path = item["path"]
        
        # Validated images are opened; any unhandled error is raised, never masked
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            tensor_img = self.transform(rgb_img)

        return {
            "image": tensor_img,
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "path": img_path,
            "generator": item.get("generator", "unknown"),
            "dataset": item.get("dataset", "unknown")
        }

def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.05) -> dict:
    """
    Computes non-parametric bootstrap 95% confidence intervals for Accuracy and ROC-AUC.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    rng = np.random.RandomState(42)
    acc_scores = []
    auc_scores = []
    n = len(y_true)

    # Check class balance
    unique_classes = np.unique(y_true)
    single_class = (len(unique_classes) < 2)

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, n)
        b_true = y_true[boot_idx]
        b_prob = y_prob[boot_idx]
        b_pred = (b_prob >= 0.5).astype(int)

        acc_scores.append(accuracy_score(b_true, b_pred))
        if not single_class and len(np.unique(b_true)) == 2:
            try:
                auc_scores.append(roc_auc_score(b_true, b_prob))
            except Exception:
                pass

    acc_ci = (
        float(np.percentile(acc_scores, 100 * (alpha / 2))),
        float(np.percentile(acc_scores, 100 * (1 - alpha / 2)))
    )
    auc_ci = (
        float(np.percentile(auc_scores, 100 * (alpha / 2))) if auc_scores else None,
        float(np.percentile(auc_scores, 100 * (1 - alpha / 2))) if auc_scores else None
    )

    return {
        "accuracy_ci_95": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
        "roc_auc_ci_95": [round(auc_ci[0], 4), round(auc_ci[1], 4)] if auc_ci[0] is not None else None
    }

def detect_model_type(state_dict: dict, checkpoint_path: str) -> str:
    """Auto-detects architecture from state_dict keys or checkpoint filename."""
    ckpt_name = Path(checkpoint_path).name.lower()
    keys = list(state_dict.keys())
    
    if any(k.startswith("spatial_branch") or k.startswith("frequency_branch") or k.startswith("fusion") for k in keys):
        return "lxfd"
    elif any(k.startswith("backbone.") or "efficientnet" in ckpt_name for k in keys):
        return "efficientnet"
    elif any("denseblock" in k or "dense" in ckpt_name for k in keys):
        return "dense_cnn"
    elif any("simple_cnn" in ckpt_name for k in keys):
        return "simple_cnn"
    return "lxfd"

def load_eval_model(checkpoint_path: str, model_type: str = "auto", fusion: str = "attention", device: torch.device = None) -> torch.nn.Module:
    """Instantiates and loads model weights cleanly from checkpoint."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = device or get_device()
    checkpoint = torch.load(ckpt_file, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if model_type == "auto":
        resolved_type = detect_model_type(state_dict, checkpoint_path)
        print(f"[MODEL] Auto-detected architecture: '{resolved_type}' for {ckpt_file.name}")
    else:
        resolved_type = model_type

    if resolved_type == "lxfd":
        is_spectral = "spectral_dropout" in ckpt_file.name.lower()
        model = LXDFDModel(embed_dim=256, fusion_type=fusion, pretrained=False, spectral_dropout=is_spectral).to(device)
    elif resolved_type == "efficientnet":
        model = EfficientNetBaseline(pretrained=False).to(device)
    elif resolved_type == "dense_cnn":
        model = DenseCNNBaseline().to(device)
    elif resolved_type == "simple_cnn":
        model = SimpleCNN().to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def discover_manifest_samples(manifest_path: str) -> Tuple[List[dict], List[dict]]:
    """Loads and validates samples from a CSV or JSON manifest."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if m_path.suffix.lower() == ".json":
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(m_path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if c_low in ["path", "file_path", "image_path", "filepath"]:
            col_map[col] = "path"
        elif c_low in ["label", "class", "target"]:
            col_map[col] = "label"
        elif c_low in ["generator", "generator_name", "gen"]:
            col_map[col] = "generator"
        elif c_low in ["dataset", "dataset_name", "source"]:
            col_map[col] = "dataset"
    df = df.rename(columns=col_map)

    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Manifest must contain 'path' and 'label' columns! Found: {df.columns.tolist()}")

    valid_samples = []
    invalid_samples = []

    for _, row in df.iterrows():
        p = Path(str(row["path"]))
        # Handle relative path relative to manifest directory
        if not p.is_absolute() and not p.exists():
            candidate = m_path.parent / p
            if candidate.exists():
                p = candidate

        lbl_raw = str(row["label"]).strip().upper()
        if lbl_raw in ["1", "1.0", "FAKE", "DEEPFAKE", "SYNTHETIC"]:
            lbl_val = 1.0
        elif lbl_raw in ["0", "0.0", "REAL", "GENUINE", "AUTHENTIC"]:
            lbl_val = 0.0
        else:
            invalid_samples.append({"path": str(p), "reason": f"Unrecognized label value: {row['label']}"})
            continue

        if not p.exists():
            invalid_samples.append({"path": str(p), "reason": "File does not exist on disk"})
            continue

        try:
            with Image.open(p) as img:
                img.verify()
            valid_samples.append({
                "path": str(p.resolve()),
                "label": lbl_val,
                "generator": str(row.get("generator", "unspecified")),
                "dataset": str(row.get("dataset", "external"))
            })
        except Exception as e:
            invalid_samples.append({"path": str(p), "reason": f"Image corrupted or unreadable: {e}"})

    return valid_samples, invalid_samples

def discover_directory_samples(unseen_dir: str) -> Tuple[List[dict], List[dict]]:
    """Scans directory structure (real/ and <generator>/ subfolders)."""
    dir_path = Path(unseen_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Unseen directory not found: {unseen_dir}")

    valid_samples = []
    invalid_samples = []
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for sub in dir_path.iterdir():
        if not sub.is_dir():
            continue
        sub_name = sub.name
        is_real = (sub_name.lower() in ["real", "authentic", "genuine"])
        label_val = 0.0 if is_real else 1.0
        gen_name = "None" if is_real else sub_name

        for fpath in sub.iterdir():
            if fpath.suffix.lower() not in image_exts:
                continue

            try:
                with Image.open(fpath) as img:
                    img.verify()
                valid_samples.append({
                    "path": str(fpath.resolve()),
                    "label": label_val,
                    "generator": gen_name,
                    "dataset": "unseen_directory"
                })
            except Exception as e:
                invalid_samples.append({"path": str(fpath), "reason": f"Corrupted or invalid image: {e}"})

    return valid_samples, invalid_samples

def run_dry_run_audit(valid_samples: List[dict], invalid_samples: List[dict]) -> dict:
    """Executes pre-evaluation integrity dry run."""
    print("\n" + "=" * 80)
    print("                    EXTERNAL DATASET DRY-RUN AUDIT REPORT                       ")
    print("=" * 80)
    print(f"Total Discovered Files: {len(valid_samples) + len(invalid_samples)}")
    print(f"Valid Readable Images:  {len(valid_samples)}")
    print(f"Invalid / Corrupted:    {len(invalid_samples)}")
    print("-" * 80)

    if invalid_samples:
        print("[WARNING] Invalid Images Identified:")
        for inv in invalid_samples[:10]:
            print(f"  - {inv['path']}: {inv['reason']}")
        if len(invalid_samples) > 10:
            print(f"  ... and {len(invalid_samples) - 10} more invalid files.")
        print("-" * 80)

    df_valid = pd.DataFrame(valid_samples)
    if not df_valid.empty:
        summary_tbl = []
        for gen, group in df_valid.groupby("generator"):
            n_real = (group["label"] == 0.0).sum()
            n_fake = (group["label"] == 1.0).sum()
            summary_tbl.append({
                "Generator": gen,
                "Real Count": int(n_real),
                "Fake Count": int(n_fake),
                "Total Valid": len(group)
            })
        df_sum = pd.DataFrame(summary_tbl)
        print(df_sum.to_string(index=False))
    else:
        print("[ERROR] Zero valid images discovered! Check dataset paths.")
    print("=" * 80 + "\n")

    return {
        "status": "dry_run_complete",
        "valid_count": len(valid_samples),
        "invalid_count": len(invalid_samples),
        "invalid_samples": invalid_samples
    }

def evaluate_unseen_generators(
    checkpoint_path: str,
    manifest_path: Optional[str] = None,
    unseen_dir: Optional[str] = None,
    model_type: str = "auto",
    fusion: str = "attention",
    batch_size: int = 32,
    output_json: str = "reports/phase4_external_results.json",
    output_csv: str = "reports/phase4_external_metrics.csv",
    save_predictions: Optional[str] = None,
    dry_run: bool = False
) -> dict:
    """Comprehensive, leak-free external evaluation pipeline."""
    # Step 1: Discover and validate samples
    if manifest_path:
        print(f"[UNSEEN] Loading manifest: {manifest_path}...")
        valid_samples, invalid_samples = discover_manifest_samples(manifest_path)
    elif unseen_dir:
        print(f"[UNSEEN] Scanning directory: {unseen_dir}...")
        valid_samples, invalid_samples = discover_directory_samples(unseen_dir)
    else:
        raise ValueError("Must provide either --manifest or --unseen-dir!")

    if dry_run:
        audit_res = run_dry_run_audit(valid_samples, invalid_samples)
        return audit_res

    if not valid_samples:
        print("[ERROR] No valid images to evaluate! Aborting.")
        return {"status": "error", "message": "No valid images discovered"}

    run_dry_run_audit(valid_samples, invalid_samples)

    # Step 2: Load Model
    device = get_device()
    print(f"[UNSEEN] Loading checkpoint: {checkpoint_path}...")
    model = load_eval_model(checkpoint_path, model_type=model_type, fusion=fusion, device=device)

    # Step 3: Run Inference
    dataset = ValidatedImageDataset(valid_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    predictions = []
    print(f"[UNSEEN] Running inference across {len(valid_samples)} samples on device: {device}...")

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].cpu().numpy()
            paths = batch["path"]
            gens = batch["generator"]
            dsets = batch["dataset"]

            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for p, y, prob, g, d in zip(paths, lbls, probs, gens, dsets):
                pred_binary = 1 if prob >= 0.5 else 0
                predictions.append({
                    "path": p,
                    "dataset": d,
                    "generator": g,
                    "true_label": int(y),
                    "fake_prob": float(prob),
                    "pred_label": pred_binary,
                    "correct": int(pred_binary == int(y))
                })

    df_preds = pd.DataFrame(predictions)

    if save_predictions:
        Path(save_predictions).parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(save_predictions, index=False)
        print(f"[UNSEEN] Per-sample predictions saved to: {save_predictions}")

    # Step 4: Disaggregated Per-Generator Evaluation
    # Separate authentic real images from generator synthetic images
    real_df = df_preds[df_preds["true_label"] == 0]
    fake_df = df_preds[df_preds["true_label"] == 1]

    per_generator_results = {}
    csv_rows = []

    generators = [g for g in fake_df["generator"].unique() if g.lower() not in ["none", "real"]]
    if not generators and len(fake_df) > 0:
        generators = ["all_fakes"]

    for gen in generators:
        gen_fakes = fake_df[fake_df["generator"] == gen] if gen != "all_fakes" else fake_df
        # Pair with all available real control images
        eval_df = pd.concat([real_df, gen_fakes], ignore_index=True)

        y_true = eval_df["true_label"].values
        y_prob = eval_df["fake_prob"].values

        n_real = len(real_df)
        n_fake = len(gen_fakes)

        if n_real > 0 and n_fake > 0:
            metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
            ci = compute_bootstrap_ci(y_true, y_prob)
            ece = compute_ece(y_true, y_prob)
            brier = compute_brier_score(y_true, y_prob)

            metrics.update({
                "generator": gen,
                "num_real": int(n_real),
                "num_fake": int(n_fake),
                "num_total": len(eval_df),
                "ece": round(ece, 4),
                "brier_score": round(brier, 4),
                "confidence_intervals": ci
            })
        else:
            # Single-class dataset
            y_pred = (y_prob >= 0.5).astype(int)
            acc = float(np.mean(y_pred == y_true))
            metrics = {
                "generator": gen,
                "num_real": int(n_real),
                "num_fake": int(n_fake),
                "num_total": len(eval_df),
                "accuracy": round(acc, 4),
                "mean_fake_prob": round(float(np.mean(y_prob)), 4),
                "note": "ROC-AUC and ECE are undefined due to single-class absence of control samples"
            }

        per_generator_results[gen] = metrics
        csv_rows.append({
            "Generator": gen,
            "Real Samples": n_real,
            "Fake Samples": n_fake,
            "Accuracy": metrics.get("accuracy", 0.0),
            "ROC-AUC": metrics.get("roc_auc", None),
            "PR-AUC": metrics.get("pr_auc", None),
            "Precision": metrics.get("precision", None),
            "Recall": metrics.get("recall", None),
            "F1": metrics.get("f1", None),
            "ECE": metrics.get("ece", None),
            "Brier Score": metrics.get("brier_score", None),
            "ROC-AUC 95% CI": str(metrics.get("confidence_intervals", {}).get("roc_auc_ci_95", "N/A"))
        })

    # Step 5: Compute Macro and Weighted Averages
    valid_aucs = [m["roc_auc"] for m in per_generator_results.values() if "roc_auc" in m and m["roc_auc"] is not None]
    valid_accs = [m["accuracy"] for m in per_generator_results.values() if "accuracy" in m]
    valid_f1s = [m["f1"] for m in per_generator_results.values() if "f1" in m and m["f1"] is not None]
    weights = [m["num_total"] for m in per_generator_results.values() if "roc_auc" in m and m["roc_auc"] is not None]

    if valid_aucs:
        macro_auc = float(np.mean(valid_aucs))
        macro_acc = float(np.mean(valid_accs))
        macro_f1 = float(np.mean(valid_f1s))

        weighted_auc = float(np.average(valid_aucs, weights=weights))
        weighted_acc = float(np.average(valid_accs, weights=[m["num_total"] for m in per_generator_results.values() if "accuracy" in m]))
        weighted_f1 = float(np.average(valid_f1s, weights=weights))
    else:
        macro_auc = macro_acc = macro_f1 = weighted_auc = weighted_acc = weighted_f1 = None

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_type": model_type,
        "fusion": fusion,
        "total_evaluated": len(df_preds),
        "total_real": len(real_df),
        "total_fake": len(fake_df),
        "invalid_images_omitted": len(invalid_samples),
        "per_generator_metrics": per_generator_results,
        "macro_average": {
            "accuracy": round(macro_acc, 4) if macro_acc else None,
            "roc_auc": round(macro_auc, 4) if macro_auc else None,
            "f1": round(macro_f1, 4) if macro_f1 else None
        },
        "weighted_average": {
            "accuracy": round(weighted_acc, 4) if weighted_acc else None,
            "roc_auc": round(weighted_auc, 4) if weighted_auc else None,
            "f1": round(weighted_f1, 4) if weighted_f1 else None
        }
    }

    # Save reports
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df_csv = pd.DataFrame(csv_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_csv.to_csv(output_csv, index=False)

    print("\n" + "=" * 90)
    print(f"                 EXTERNAL UNSEEN GENERATOR EVALUATION: {Path(checkpoint_path).name}                ")
    print("=" * 90)
    print(df_csv.to_string(index=False))
    print("-" * 90)
    if macro_auc:
        print(f"Macro Average:    Acc: {macro_acc*100:.2f}% | ROC-AUC: {macro_auc:.4f} | F1: {macro_f1:.4f}")
        print(f"Weighted Average: Acc: {weighted_acc*100:.2f}% | ROC-AUC: {weighted_auc:.4f} | F1: {weighted_f1:.4f}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV:  {output_csv}")
    print("=" * 90 + "\n")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detectors on Unseen AI Generators")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lxfd_attention_robustness_v2_best.pt")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV or JSON")
    parser.add_argument("--unseen-dir", type=str, default="data/unseen_generator_test", help="Path to directory structure")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "lxfd", "efficientnet", "dense_cnn", "simple_cnn"])
    parser.add_argument("--fusion", type=str, default="attention")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=str, default="reports/phase4_external_results.json")
    parser.add_argument("--output-csv", type=str, default="reports/phase4_external_metrics.csv")
    parser.add_argument("--save-predictions", type=str, default="reports/phase4_external_predictions.csv")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight dataset and image integrity audit without model evaluation")
    args = parser.parse_args()

    evaluate_unseen_generators(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        unseen_dir=args.unseen_dir,
        model_type=args.model_type,
        fusion=args.fusion,
        batch_size=args.batch_size,
        output_json=args.output_json,
        output_csv=args.output_csv,
        save_predictions=args.save_predictions,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
