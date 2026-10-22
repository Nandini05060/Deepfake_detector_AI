import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.lxfd_model import LXDFDModel
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.simple_cnn import SimpleCNN
from src.data.dataset import DeepfakeDataset
from src.utils.device import get_device

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(np.maximum(y_prob[in_bin], 1.0 - y_prob[in_bin]))
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)
    ll = float(log_loss(y_true, y_prob_clipped))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "brier_score": brier,
        "ece": ece,
        "log_loss": ll
    }

class CorruptedDataset(Dataset):
    def __init__(self, df, corruption_type, corruption_param):
        self.paths = df['path'].tolist()
        self.labels = [1.0 if str(lbl).upper() in ['FAKE', '1', '1.0'] else 0.0 for lbl in df['label']]
        self.corruption_type = corruption_type
        self.corruption_param = corruption_param
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.corruption_type == 'jpeg':
            import io
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=int(self.corruption_param))
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        elif self.corruption_type == 'resize':
            scale = float(self.corruption_param)
            w, h = img.size
            nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
            img = img.resize((nw, nh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
        elif self.corruption_type == 'gaussian_blur':
            sigma = float(self.corruption_param)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif self.corruption_type == 'brightness':
            factor = float(self.corruption_param)
            img = ImageEnhance.Brightness(img).enhance(factor)
        elif self.corruption_type == 'contrast':
            factor = float(self.corruption_param)
            img = ImageEnhance.Contrast(img).enhance(factor)
        
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(img)

        if self.corruption_type == 'gaussian_noise':
            std = float(self.corruption_param)
            noise = torch.randn_like(tensor) * std
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)

        tensor = self.normalize(tensor)
        return {'image': tensor, 'label': torch.tensor(self.labels[idx], dtype=torch.float32)}

def run_phase5_corruption_evaluation():
    device = get_device()
    print(f"[PHASE 5] Starting Systematic Corruption Evaluation on device: {device}")

    test_csv = "data/splits_v2/test.csv"
    test_df = pd.read_csv(test_csv)
    test_dataset = DeepfakeDataset(test_csv, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"[PHASE 5] Loaded clean test split from {test_csv} with {len(test_df)} samples.")

    # Models to benchmark in Phase 5
    models_to_evaluate = [
        {
            "id": "lxfd_spectral_dropout",
            "name": "LX-DFD Proposed (Spectral Dropout)",
            "type": "lxfd",
            "ckpt": "checkpoints/lxfd_attention_standard_spectral_dropout_best.pt",
            "args": {"fusion_type": "attention", "pretrained": False, "spectral_dropout": True, "spectral_dropout_p": 0.2}
        },
        {
            "id": "lxfd_standard_baseline",
            "name": "LX-DFD Proposed (Standard Aug)",
            "type": "lxfd",
            "ckpt": "checkpoints/lxfd_attention_standard_best.pt",
            "args": {"fusion_type": "attention", "pretrained": False, "spectral_dropout": False}
        },
        {
            "id": "lxfd_robustness_v2",
            "name": "LX-DFD Proposed (Robustness-Aware v2)",
            "type": "lxfd",
            "ckpt": "checkpoints/lxfd_attention_robustness_v2_best.pt",
            "args": {"fusion_type": "attention", "pretrained": False, "spectral_dropout": False}
        }
    ]

    corruptions = [
        ("Clean / Original", "none", None),
        ("JPEG Quality 90", "jpeg", 90),
        ("JPEG Quality 75", "jpeg", 75),
        ("JPEG Quality 50", "jpeg", 50),
        ("JPEG Quality 30", "jpeg", 30),
        ("Resize 0.75x", "resize", 0.75),
        ("Resize 0.50x", "resize", 0.50),
        ("Gaussian Blur (r=1.5)", "gaussian_blur", 1.5),
        ("Gaussian Blur (r=3.0)", "gaussian_blur", 3.0),
        ("Gaussian Noise (std=0.05)", "gaussian_noise", 0.05),
        ("Gaussian Noise (std=0.10)", "gaussian_noise", 0.10),
        ("Brightness Factor 0.7x", "brightness", 0.7),
        ("Brightness Factor 1.3x", "brightness", 1.3),
        ("Contrast Factor 0.7x", "contrast", 0.7),
        ("Contrast Factor 1.3x", "contrast", 1.3)
    ]

    model_results = {}
    corruption_table_rows = []

    for m_info in models_to_evaluate:
        ckpt_path = Path(m_info["ckpt"])
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\nEvaluating {m_info['name']} ({ckpt_path})...")
        if m_info["type"] == "efficientnet":
            model = EfficientNetBaseline(**m_info["args"]).to(device)
        elif m_info["type"] == "lxfd":
            model = LXDFDModel(**m_info["args"]).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        # Measure clean metrics & inference latency
        y_true_clean, y_prob_clean = [], []
        start_t = time.time()
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch['image'].to(device)
                lbls = batch['label'].numpy()
                out = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.sigmoid(logits).cpu().numpy()
                y_true_clean.extend(lbls)
                y_prob_clean.extend(probs)
        total_time = time.time() - start_t
        latency_per_sample_ms = (total_time / len(test_dataset)) * 1000.0

        clean_eval = evaluate_predictions(y_true_clean, y_prob_clean)
        clean_auc = clean_eval["roc_auc"]
        print(f"  [Clean] Acc: {clean_eval['accuracy']*100:.2f}% | ROC-AUC: {clean_eval['roc_auc']:.4f} | PR-AUC: {clean_eval['pr_auc']:.4f} | F1: {clean_eval['f1']:.4f} | Brier: {clean_eval['brier_score']:.4f} | ECE: {clean_eval['ece']:.4f} | Latency: {latency_per_sample_ms:.2f} ms")

        # Evaluate across all corruptions
        corr_results = {}
        auc_drops = []
        for c_name, c_type, c_param in corruptions:
            if c_type == "none":
                c_eval = clean_eval
                auc_drop = 0.0
            else:
                ds = CorruptedDataset(test_df, c_type, c_param)
                loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
                y_t, y_p = [], []
                with torch.no_grad():
                    for batch in loader:
                        imgs = batch['image'].to(device)
                        lbls = batch['label'].numpy()
                        out = model(imgs)
                        logits = out[0] if isinstance(out, tuple) else out
                        probs = torch.sigmoid(logits).cpu().numpy()
                        y_t.extend(lbls)
                        y_p.extend(probs)
                c_eval = evaluate_predictions(y_t, y_p)
                auc_drop = max(0.0, clean_auc - c_eval["roc_auc"])
                auc_drops.append(auc_drop)

            corr_results[c_name] = {
                "accuracy": c_eval["accuracy"],
                "roc_auc": c_eval["roc_auc"],
                "f1": c_eval["f1"],
                "auc_drop": float(auc_drop)
            }
            corruption_table_rows.append({
                "Model": m_info["name"],
                "Corruption Condition": c_name,
                "Clean ROC-AUC": round(clean_auc, 4),
                "Corrupted ROC-AUC": round(c_eval["roc_auc"], 4),
                "ROC-AUC Drop": round(auc_drop, 4),
                "Corrupted Accuracy": f"{c_eval['accuracy']*100:.2f}%",
                "Corrupted F1": round(c_eval["f1"], 4)
            })
            if c_type != "none":
                print(f"    {c_name:<28} | AUC: {c_eval['roc_auc']:.4f} (drop: {auc_drop:.4f}) | Acc: {c_eval['accuracy']*100:.2f}% | F1: {c_eval['f1']:.4f}")

        mean_drop = float(np.mean(auc_drops))
        rob_score = float(max(0.0, 1.0 - mean_drop))
        print(f"  [Summary] Mean AUC Drop: {mean_drop:.4f} | Robustness Score: {rob_score:.4f}")

        model_results[m_info["id"]] = {
            "name": m_info["name"],
            "checkpoint": str(ckpt_path),
            "clean_metrics": clean_eval,
            "corruptions": corr_results,
            "mean_auc_drop": mean_drop,
            "robustness_score": rob_score,
            "latency_ms_per_sample": latency_per_sample_ms
        }

    # Save Phase 5 Corruption Analysis CSV
    df_corr = pd.DataFrame(corruption_table_rows)
    Path("reports").mkdir(exist_ok=True)
    df_corr.to_csv("reports/phase5_corruption_analysis.csv", index=False)
    print("\n[PHASE 5] Saved reports/phase5_corruption_analysis.csv")

    # Save Phase 5 intermediate metrics json
    with open("reports/phase5_corruption_metrics.json", "w") as f:
        json.dump(model_results, f, indent=2)
    print("[PHASE 5] Saved reports/phase5_corruption_metrics.json")

    return model_results

if __name__ == "__main__":
    run_phase5_corruption_evaluation()
