"""
Independent Scientific & Technical Audit Script:
Reconciles Phase 10, Phase 15, and Phase 16 evaluations on the 448 locked test images.
Tests under multiple rigorous evaluation rules:
Rule 1: Pure Binary Classification at standard threshold (tau = 0.50)
Rule 2: Pure Binary Classification at calibrated threshold (tau = 0.57 for P16, 0.55 for P10/P15)
Rule 3: Production API / Inference Pipeline Policy (src/models/inference.py)
Rule 4: Training Script Policy (evaluate_model_comprehensive in train_phase16_unified_generalization.py)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.models.lxfd_model import LXDFDModel
from src.models.inference import LXDFDPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test manifest
manifest_path = repo_root / "reports/phase16_generator_split_manifest.csv"
manifest_df = pd.read_csv(manifest_path)
test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
print(f"Loaded {len(test_df)} test records from {manifest_path}")

# Check uniqueness of test records
print(f"Unique test filenames: {test_df['filename'].nunique()}")
print(f"Unique test relative_paths: {test_df['relative_path'].nunique()}")
print(f"Unique test sha256: {test_df['sha256'].nunique()}")

# Verify all files exist
for p in test_df['relative_path']:
    assert (repo_root / p).exists(), f"File missing: {p}"
print("Verified all 448 files exist on disk.")

# Define test dataset loader with aspect-ratio preserving YuNet crop
import torchvision.transforms as T
import cv2

yunet_path = repo_root / "yunet.onnx"
detector = None
if yunet_path.exists():
    detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), score_threshold=0.5)

def get_crop(pil_img, margin=0.50):
    w, h = pil_img.size
    if detector is None:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return pil_img.crop((left, top, left + min_dim, top + min_dim))

    orig_np = np.array(pil_img)
    detector.setInputSize((w, h))
    bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    _, faces = detector.detect(bgr)

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

        if (x2 - x1) >= 32 and (y2 - y1) >= 32:
            return Image.fromarray(orig_np[y1:y2, x1:x2])

    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    return pil_img.crop((left, top, left + min_dim, top + min_dim))

eval_transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

# Evaluate each model
models_to_audit = [
    {
        "id": "phase10",
        "name": "LX-DFD Phase 10 Modern Multi-Generator",
        "ckpt": repo_root / "checkpoints/lxfd_phase10_modern_generators_best.pt",
        "temp": 1.6344,
        "threshold": 0.55
    },
    {
        "id": "phase15",
        "name": "LX-DFD Phase 15 Gemini Generalization Candidate",
        "ckpt": repo_root / "experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt",
        "temp": 0.7169,
        "threshold": 0.55
    },
    {
        "id": "phase16",
        "name": "LX-DFD Phase 16 Unified Multi-Generator Best",
        "ckpt": repo_root / "experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt",
        "temp": 1.0859,
        "threshold": 0.57
    }
]

audit_results = {}

for m_info in models_to_audit:
    m_id = m_info["id"]
    print(f"\n=======================================================")
    print(f"Auditing Model: {m_info['name']}")
    print(f"Checkpoint: {m_info['ckpt']}")
    print(f"=======================================================")

    model = LXDFDModel(fusion_type="attention", min_floor=0.25, pretrained=False).to(device)
    ckpt = torch.load(m_info["ckpt"], map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    temp = m_info["temp"]
    threshold = m_info["threshold"]

    # Pre-extract all predictions
    records = []
    t_start = time.time()

    with torch.no_grad():
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            img_path = repo_root / row["relative_path"]
            label = 1.0 if str(row["label"]).upper() == "FAKE" else 0.0
            gen = row["generator_family"]

            pil_img = Image.open(img_path).convert("RGB")
            face_img = get_crop(pil_img, margin=0.50)
            tensor_img = eval_transform(face_img).unsqueeze(0).to(device)

            out, meta = model(tensor_img)
            raw_logit = out.item()
            raw_fake_prob = 1.0 / (1.0 + np.exp(-raw_logit))
            cal_fake_prob = 1.0 / (1.0 + np.exp(-raw_logit / temp))

            sp_prob = meta['spatial_fake_prob'].item() if 'spatial_fake_prob' in meta else raw_fake_prob
            fq_prob = meta['frequency_fake_prob'].item() if 'frequency_fake_prob' in meta else raw_fake_prob
            w_s = meta.get('w_s', 0.5)
            w_f = meta.get('w_f', 0.5)
            if torch.is_tensor(w_s): w_s = w_s.item()
            if torch.is_tensor(w_f): w_f = w_f.item()

            records.append({
                "index": idx,
                "image_id": row["image_id"],
                "filename": row["filename"],
                "generator_family": gen,
                "label": int(label),
                "raw_logit": float(raw_logit),
                "raw_fake_prob": float(raw_fake_prob),
                "cal_fake_prob": float(cal_fake_prob),
                "spatial_prob": float(sp_prob),
                "freq_prob": float(fq_prob),
                "w_s": float(w_s),
                "w_f": float(w_f)
            })

    total_time = time.time() - t_start
    latency_ms = (total_time / len(test_df)) * 1000.0

    rec_df = pd.DataFrame(records)

    # 1. Binary Metrics at Standard Threshold tau = 0.50
    bin_preds_50 = (rec_df["cal_fake_prob"] >= 0.50).astype(int)
    cm_bin_50 = confusion_matrix(rec_df["label"], bin_preds_50, labels=[0, 1])
    tn_b50, fp_b50, fn_b50, tp_b50 = cm_bin_50.ravel()

    # 2. Binary Metrics at Calibrated Threshold tau = threshold
    bin_preds_tau = (rec_df["cal_fake_prob"] >= threshold).astype(int)
    cm_bin_tau = confusion_matrix(rec_df["label"], bin_preds_tau, labels=[0, 1])
    tn_bt, fp_bt, fn_bt, tp_bt = cm_bin_tau.ravel()

    # 3. Training Script Comprehensive Policy
    # (train_phase16_unified_generalization.py evaluate_model_comprehensive)
    train_decisions = []
    unc_band = 0.075
    lower_band = threshold - unc_band
    upper_band = threshold + unc_band

    for _, r in rec_df.iterrows():
        p = r["cal_fake_prob"]
        sp = r["spatial_prob"]
        fp = r["freq_prob"]

        if lower_band <= p <= upper_band:
            train_decisions.append("INCONCLUSIVE")
        elif sp < threshold and fp < threshold:
            if fp >= 0.28 and sp < 0.10:
                train_decisions.append("INCONCLUSIVE")
            else:
                train_decisions.append("LIKELY_REAL")
        elif sp >= threshold and fp >= threshold:
            train_decisions.append("LIKELY_FAKE")
        elif p >= 0.80 and (sp >= 0.85 or fp >= 0.85):
            train_decisions.append("LIKELY_FAKE")
        else:
            train_decisions.append("INCONCLUSIVE")

    rec_df["train_policy_decision"] = train_decisions

    # 4. Production API Policy (src/models/inference.py)
    prod_decisions = []
    for _, r in rec_df.iterrows():
        p = r["cal_fake_prob"]
        sp = r["spatial_prob"]
        fp = r["freq_prob"]
        disc = abs(sp - fp)
        high_disc = disc > 0.10

        if lower_band <= p <= upper_band:
            prod_decisions.append("INCONCLUSIVE")
        elif high_disc:
            if sp >= threshold and fp >= threshold:
                prod_decisions.append("LIKELY_FAKE")
            elif sp < threshold and fp < threshold:
                if fp >= 0.28 and sp < 0.10:
                    prod_decisions.append("INCONCLUSIVE")
                else:
                    prod_decisions.append("LIKELY_REAL")
            elif p >= 0.80 and (sp >= 0.85 or fp >= 0.85):
                prod_decisions.append("LIKELY_FAKE")
            else:
                prod_decisions.append("INCONCLUSIVE")
        elif p >= threshold:
            prod_decisions.append("LIKELY_FAKE")
        else:
            prod_decisions.append("LIKELY_REAL")

    rec_df["prod_policy_decision"] = prod_decisions

    # Calculate metrics for Train Policy
    tr_dec_mask = rec_df["train_policy_decision"] != "INCONCLUSIVE"
    tr_dec_df = rec_df[tr_dec_mask]
    tr_preds = (tr_dec_df["train_policy_decision"] == "LIKELY_FAKE").astype(int)
    cm_tr = confusion_matrix(tr_dec_df["label"], tr_preds, labels=[0, 1])
    tn_tr, fp_tr, fn_tr, tp_tr = cm_tr.ravel() if cm_tr.size == 4 else (0, 0, 0, 0)
    inconc_real_tr = int(np.sum((rec_df["label"] == 0) & (~tr_dec_mask)))
    inconc_fake_tr = int(np.sum((rec_df["label"] == 1) & (~tr_dec_mask)))

    # Calculate metrics for Production API Policy
    pr_dec_mask = rec_df["prod_policy_decision"] != "INCONCLUSIVE"
    pr_dec_df = rec_df[pr_dec_mask]
    pr_preds = (pr_dec_df["prod_policy_decision"] == "LIKELY_FAKE").astype(int)
    cm_pr = confusion_matrix(pr_dec_df["label"], pr_preds, labels=[0, 1])
    tn_pr, fp_pr, fn_pr, tp_pr = cm_pr.ravel() if cm_pr.size == 4 else (0, 0, 0, 0)
    inconc_real_pr = int(np.sum((rec_df["label"] == 0) & (~pr_dec_mask)))
    inconc_fake_pr = int(np.sum((rec_df["label"] == 1) & (~pr_dec_mask)))

    # Overall probabilistic scores
    y_true = rec_df["label"].values
    y_prob = rec_df["cal_fake_prob"].values
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(compute_ece(y_prob, y_true))

    print(f"\n--- Probabilistic Scores (N = 448) ---")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")

    print(f"\n--- Pure Binary (tau = {threshold:.2f}, No Inconclusive) ---")
    print(f"TN: {tn_bt}, FP: {fp_bt}, FN: {fn_bt}, TP: {tp_bt}")
    print(f"Accuracy: {accuracy_score(y_true, bin_preds_tau)*100:.2f}% | Specificity: {tn_bt/(tn_bt+fp_bt)*100:.2f}% | Recall: {tp_bt/(tp_bt+fn_bt)*100:.2f}%")

    print(f"\n--- Training Script Policy (evaluate_model_comprehensive) ---")
    print(f"Decisive samples: {len(tr_dec_df)}/{len(rec_df)} (Inconclusive: {len(rec_df)-len(tr_dec_df)})")
    print(f"TN: {tn_tr}, FP: {fp_tr}, FN: {fn_tr}, TP: {tp_tr}")
    print(f"Inconclusive Real: {inconc_real_tr}, Inconclusive Fake: {inconc_fake_tr}")
    if len(tr_dec_df) > 0:
        acc_tr = accuracy_score(tr_dec_df["label"], tr_preds)
        spec_tr = tn_tr / (tn_tr + fp_tr) if (tn_tr + fp_tr) > 0 else 0
        rec_tr = tp_tr / (tp_tr + fn_tr) if (tp_tr + fn_tr) > 0 else 0
        prec_tr = tp_tr / (tp_tr + fp_tr) if (tp_tr + fp_tr) > 0 else 0
        f1_tr = f1_score(tr_dec_df["label"], tr_preds, zero_division=0)
        print(f"Decisive Acc: {acc_tr*100:.2f}% | Decisive Spec: {spec_tr*100:.2f}% | Decisive Recall: {rec_tr*100:.2f}% | F1: {f1_tr:.4f}")

    print(f"\n--- Production API Policy (inference.py) ---")
    print(f"Decisive samples: {len(pr_dec_df)}/{len(rec_df)} (Inconclusive: {len(rec_df)-len(pr_dec_df)})")
    print(f"TN: {tn_pr}, FP: {fp_pr}, FN: {fn_pr}, TP: {tp_pr}")
    print(f"Inconclusive Real: {inconc_real_pr}, Inconclusive Fake: {inconc_fake_pr}")
    if len(pr_dec_df) > 0:
        acc_pr = accuracy_score(pr_dec_df["label"], pr_preds)
        spec_pr = tn_pr / (tn_pr + fp_pr) if (tn_pr + fp_pr) > 0 else 0
        rec_pr = tp_pr / (tp_pr + fn_pr) if (tp_pr + fn_pr) > 0 else 0
        prec_pr = tp_pr / (tp_pr + fp_pr) if (tp_pr + fp_pr) > 0 else 0
        f1_pr = f1_score(pr_dec_df["label"], pr_preds, zero_division=0)
        print(f"Decisive Acc: {acc_pr*100:.2f}% | Decisive Spec: {spec_pr*100:.2f}% | Decisive Recall: {rec_pr*100:.2f}% | F1: {f1_pr:.4f}")

    # Per-generator breakdown under production policy and train policy
    per_gen_breakdown = {}
    for gen in sorted(rec_df["generator_family"].unique()):
        gdf = rec_df[rec_df["generator_family"] == gen]
        g_labels = gdf["label"].values
        g_is_fake = g_labels[0] == 1
        
        # Train policy
        g_tr_dec = gdf[gdf["train_policy_decision"] != "INCONCLUSIVE"]
        g_tr_preds = (g_tr_dec["train_policy_decision"] == "LIKELY_FAKE").astype(int)
        g_tr_acc = accuracy_score(g_tr_dec["label"], g_tr_preds) if len(g_tr_dec) > 0 else 0.0

        # Production policy
        g_pr_dec = gdf[gdf["prod_policy_decision"] != "INCONCLUSIVE"]
        g_pr_preds = (g_pr_dec["prod_policy_decision"] == "LIKELY_FAKE").astype(int)
        g_pr_acc = accuracy_score(g_pr_dec["label"], g_pr_preds) if len(g_pr_dec) > 0 else 0.0

        per_gen_breakdown[gen] = {
            "total_samples": len(gdf),
            "ground_truth_label": "FAKE" if g_is_fake else "REAL",
            "mean_fake_prob": float(gdf["cal_fake_prob"].mean()),
            "train_policy": {
                "decisive_count": len(g_tr_dec),
                "inconclusive_count": len(gdf) - len(g_tr_dec),
                "inconclusive_rate": round((len(gdf) - len(g_tr_dec)) / len(gdf), 4),
                "decisive_accuracy": round(float(g_tr_acc), 4),
                "decisive_detected_fake": int(np.sum(g_tr_preds == 1)),
                "decisive_classified_real": int(np.sum(g_tr_preds == 0))
            },
            "production_policy": {
                "decisive_count": len(g_pr_dec),
                "inconclusive_count": len(gdf) - len(g_pr_dec),
                "inconclusive_rate": round((len(gdf) - len(g_pr_dec)) / len(gdf), 4),
                "decisive_accuracy": round(float(g_pr_acc), 4),
                "decisive_detected_fake": int(np.sum(g_pr_preds == 1)),
                "decisive_classified_real": int(np.sum(g_pr_preds == 0))
            }
        }

    audit_results[m_id] = {
        "model_id": m_id,
        "name": m_info["name"],
        "checkpoint": str(m_info["ckpt"].relative_to(repo_root)).replace("\\", "/"),
        "temperature": temp,
        "threshold": threshold,
        "latency_ms": round(latency_ms, 2),
        "overall_probabilistic": {
            "total_test_samples": len(rec_df),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "ece_10bin": round(ece, 4)
        },
        "binary_evaluation_standard_threshold_50": {
            "threshold": 0.50,
            "tn": int(tn_b50), "fp": int(fp_b50), "fn": int(fn_b50), "tp": int(tp_b50),
            "accuracy": round(float(accuracy_score(y_true, bin_preds_50)), 4),
            "fake_recall": round(float(tp_b50 / (tp_b50 + fn_b50)), 4),
            "real_specificity": round(float(tn_b50 / (tn_b50 + fp_b50)), 4),
            "precision": round(float(tp_b50 / (tp_b50 + fp_b50)), 4),
            "f1_score": round(float(f1_score(y_true, bin_preds_50)), 4)
        },
        "binary_evaluation_calibrated_threshold": {
            "threshold": threshold,
            "tn": int(tn_bt), "fp": int(fp_bt), "fn": int(fn_bt), "tp": int(tp_bt),
            "accuracy": round(float(accuracy_score(y_true, bin_preds_tau)), 4),
            "fake_recall": round(float(tp_bt / (tp_bt + fn_bt)), 4),
            "real_specificity": round(float(tn_bt / (tn_bt + fp_bt)), 4),
            "precision": round(float(tp_bt / (tp_bt + fp_bt)), 4),
            "f1_score": round(float(f1_score(y_true, bin_preds_tau)), 4)
        },
        "train_policy_evaluation": {
            "total_samples": len(rec_df),
            "decisive_samples": len(tr_dec_df),
            "inconclusive_count": len(rec_df) - len(tr_dec_df),
            "inconclusive_rate": round((len(rec_df) - len(tr_dec_df)) / len(rec_df), 4),
            "coverage": round(len(tr_dec_df) / len(rec_df), 4),
            "confusion_matrix": {
                "true_negatives": int(tn_tr),
                "false_positives": int(fp_tr),
                "false_negatives": int(fn_tr),
                "true_positives": int(tp_tr),
                "inconclusive_real": int(inconc_real_tr),
                "inconclusive_fake": int(inconc_fake_tr)
            },
            "decisive_accuracy": round(float(acc_tr), 4),
            "decisive_fake_recall": round(float(rec_tr), 4),
            "decisive_real_specificity": round(float(spec_tr), 4),
            "decisive_precision": round(float(prec_tr), 4),
            "decisive_f1_score": round(float(f1_tr), 4),
            "fpr": round(float(fp_tr / (tn_tr + fp_tr)) if (tn_tr + fp_tr) > 0 else 0, 4),
            "fnr": round(float(fn_tr / (tp_tr + fn_tr)) if (tp_tr + fn_tr) > 0 else 0, 4),
            "selective_error_rate": round(float((fp_tr + fn_tr) / len(tr_dec_df)) if len(tr_dec_df) > 0 else 0, 4)
        },
        "production_api_policy_evaluation": {
            "total_samples": len(rec_df),
            "decisive_samples": len(pr_dec_df),
            "inconclusive_count": len(rec_df) - len(pr_dec_df),
            "inconclusive_rate": round((len(rec_df) - len(pr_dec_df)) / len(rec_df), 4),
            "coverage": round(len(pr_dec_df) / len(rec_df), 4),
            "confusion_matrix": {
                "true_negatives": int(tn_pr),
                "false_positives": int(fp_pr),
                "false_negatives": int(fn_pr),
                "true_positives": int(tp_pr),
                "inconclusive_real": int(inconc_real_pr),
                "inconclusive_fake": int(inconc_fake_pr)
            },
            "decisive_accuracy": round(float(acc_pr), 4),
            "decisive_fake_recall": round(float(rec_pr), 4),
            "decisive_real_specificity": round(float(spec_pr), 4),
            "decisive_precision": round(float(prec_pr), 4),
            "decisive_f1_score": round(float(f1_pr), 4),
            "fpr": round(float(fp_pr / (tn_pr + fp_pr)) if (tn_pr + fp_pr) > 0 else 0, 4),
            "fnr": round(float(fn_pr / (tp_pr + fn_pr)) if (tp_pr + fn_pr) > 0 else 0, 4),
            "selective_error_rate": round(float((fp_pr + fn_pr) / len(pr_dec_df)) if len(pr_dec_df) > 0 else 0, 4)
        },
        "per_generator": per_gen_breakdown
    }

# Save complete audit json
out_audit_path = repo_root / "reports/final_verified_metrics.json"
with open(out_audit_path, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)
print(f"\n[SAVED AUDIT DATA] {out_audit_path}")
