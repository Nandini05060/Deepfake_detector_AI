"""
Script: experiments/phase17_multigenerator_retraining/train_phase17.py
Phase 17 Multi-Generator Retraining:
- Retrains LX-DFD Dual-Branch Spatial-Frequency Architecture
- Incorporates FLUX.1 (Dev/Pro), SDXL, Gemini Imagen 3, SD 1.5, StyleGAN2, InsightFace, FFHQ, and IMDB-WIKI
- Generator-balanced loss weighting with enhanced modern diffusion supervision
- Preserves genuine-image specificity (preventing Phase 15 collapse)
- Uses mixed precision (CUDA fp16) on NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)
- Aspect-ratio preserving YuNet crop (m=0.50)
- Validation-based early stopping using Composite Score (ROC-AUC + Specificity + Recall)
"""

import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

# Add project root to sys path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.lxfd_model import LXDFDModel
from src.evaluation.calibration import TemperatureScaler
from src.utils.device import get_device

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class RealisticForensicAugmentation:
    """Applies mild, label-preserving forensic augmentations: JPEG noise and mild blur."""
    def __init__(self, p_blur=0.25, p_jpeg=0.30):
        self.p_blur = p_blur
        self.p_jpeg = p_jpeg

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p_blur:
            radius = random.uniform(0.2, 0.6)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        if random.random() < self.p_jpeg:
            import io
            buf = io.BytesIO()
            quality = random.randint(75, 95)
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

class Phase17Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str = "train", margin: float = 0.50):
        self.df = df[df["split"].str.lower() == split.lower()].reset_index(drop=True)
        self.split = split.lower()
        self.margin = margin

        self.yunet_path = repo_root / "yunet.onnx"
        self.detector = None
        if self.yunet_path.exists():
            self.detector = cv2.FaceDetectorYN.create(str(self.yunet_path), "", (320, 320), score_threshold=0.5)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == "train":
            self.forensic_aug = RealisticForensicAugmentation(p_blur=0.25, p_jpeg=0.30)
            self.transform = T.Compose([
                T.RandomResizedCrop((224, 224), scale=(0.88, 1.05)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            self.forensic_aug = None
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.df)

    def crop_face_aspect_preserving(self, pil_img: Image.Image) -> Image.Image:
        w, h = pil_img.size
        if self.detector is None:
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            return pil_img.crop((left, top, left + min_dim, top + min_dim))

        orig_np = np.array(pil_img)
        self.detector.setInputSize((w, h))
        bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        _, faces = self.detector.detect(bgr)

        if faces is not None and len(faces) > 0:
            f = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            cx = fx + fw // 2
            cy = fy + fh // 2
            side = int(max(fw, fh) * (1.0 + 2.0 * self.margin))
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = repo_root / row["relative_path"]
        label = 1.0 if str(row["label"]).upper() == "FAKE" else 0.0

        try:
            pil_img = Image.open(img_path).convert("RGB")
            face_img = self.crop_face_aspect_preserving(pil_img)
            if self.forensic_aug is not None:
                face_img = self.forensic_aug(face_img)
            tensor_img = self.transform(face_img)
        except Exception:
            tensor_img = torch.zeros(3, 224, 224)

        return {
            "image": tensor_img,
            "label": torch.tensor(label, dtype=torch.float32),
            "generator": row.get("generator_family", "Unknown"),
            "path": str(img_path)
        }

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if np.sum(in_bin) > 0:
            avg_prob = np.mean(probs[in_bin])
            avg_acc = np.mean(labels[in_bin])
            ece += np.abs(avg_prob - avg_acc) * prop_in_bin
    return float(ece)

def evaluate_model(model, dataloader, device, temp=1.0, threshold=0.55):
    model.eval()
    all_logits = []
    all_labels = []
    all_spatial_probs = []
    all_freq_probs = []
    all_gens = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            logits, meta = model(imgs)

            all_logits.extend(logits.cpu().numpy().tolist())
            all_labels.extend(labels.tolist())
            all_spatial_probs.extend(meta.get("spatial_fake_prob", torch.sigmoid(logits)).cpu().numpy().tolist())
            all_freq_probs.extend(meta.get("frequency_fake_prob", torch.sigmoid(logits)).cpu().numpy().tolist())
            all_gens.extend(batch["generator"])

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_spatial_probs = np.array(all_spatial_probs)
    all_freq_probs = np.array(all_freq_probs)

    scaled_probs = 1.0 / (1.0 + np.exp(-all_logits / temp))

    # Binary metrics at threshold
    bin_preds = (scaled_probs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, bin_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    auc = float(roc_auc_score(all_labels, scaled_probs)) if len(np.unique(all_labels)) > 1 else 0.5
    pr_auc = float(average_precision_score(all_labels, scaled_probs)) if len(np.unique(all_labels)) > 1 else 0.5
    brier = float(brier_score_loss(all_labels, scaled_probs))
    ece = float(compute_ece(scaled_probs, all_labels))

    acc = float((tp + tn) / max(1, len(all_labels)))
    spec = float(tn / max(1, tn + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(f1_score(all_labels, bin_preds, zero_division=0))

    # 3-Way Policy Evaluation (tau +/- 0.15, divergence > 0.10)
    inconc_count = 0
    tri_tp, tri_tn, tri_fp, tri_fn = 0, 0, 0, 0
    for i in range(len(scaled_probs)):
        p = scaled_probs[i]
        sp = all_spatial_probs[i]
        fp_prob = all_freq_probs[i]
        lbl = all_labels[i]

        is_uncertain = (abs(p - threshold) <= 0.15) or (abs(sp - fp_prob) > 0.10)
        if is_uncertain:
            inconc_count += 1
        elif p >= threshold:
            if lbl == 1:
                tri_tp += 1
            else:
                tri_fp += 1
        else:
            if lbl == 0:
                tri_tn += 1
            else:
                tri_fn += 1

    decisive_total = tri_tp + tri_tn + tri_fp + tri_fn
    dec_acc = float((tri_tp + tri_tn) / decisive_total) if decisive_total > 0 else 0.0
    dec_recall = float(tri_tp / max(1, tri_tp + tri_fn)) if (tri_tp + tri_fn) > 0 else 0.0
    dec_spec = float(tri_tn / max(1, tri_tn + tri_fp)) if (tri_tn + tri_fp) > 0 else 0.0

    return {
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "binary_accuracy": round(acc, 4),
        "binary_specificity": round(spec, 4),
        "binary_recall": round(recall, 4),
        "binary_f1": round(f1, 4),
        "decisive_accuracy": round(dec_acc, 4),
        "decisive_recall": round(dec_recall, 4),
        "decisive_specificity": round(dec_spec, 4),
        "coverage": round(decisive_total / max(1, len(all_labels)), 4),
        "inconclusive_rate": round(inconc_count / max(1, len(all_labels)), 4),
        "cm_decisive": {"tn": tri_tn, "fp": tri_fp, "fn": tri_fn, "tp": tri_tp},
        "cm_binary": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "all_probs": scaled_probs,
        "all_labels": all_labels,
        "all_gens": all_gens
    }

def fit_temperature(logits, labels):
    scaler = TemperatureScaler(temperature=1.0)
    scaler.fit(logits, labels)
    fitted_t = float(scaler.temperature)
    clamped_t = max(0.90, min(1.35, fitted_t))
    scaler.temperature = clamped_t
    return clamped_t, scaler

def train_phase17():
    seed_everything(42)
    device = get_device()
    print("=" * 80)
    print("PHASE 17: TRAINING MULTI-GENERATOR DEEPFAKE FORENSICS MODEL")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    manifest_path = repo_root / "experiments/phase17_multigenerator_retraining/data_manifest.csv"
    assert manifest_path.exists(), "Phase 17 manifest missing!"
    df_manifest = pd.read_csv(manifest_path)

    train_ds = Phase17Dataset(df_manifest, split="train")
    val_ds = Phase17Dataset(df_manifest, split="validation")
    test_ds = Phase17Dataset(df_manifest, split="test")

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[DATA LOADED] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    exp_dir = repo_root / "experiments/phase17_multigenerator_retraining"
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = exp_dir / "lxfd_phase17_best.pt"

    # Initialize Model with attention floor min_floor=0.25
    model = LXDFDModel(fusion_type="attention", min_floor=0.25, pretrained=False).to(device)

    # Warm-start with Phase 16 unified weights to accelerate convergence and preserve generalizability
    p16_ckpt = repo_root / "experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt"
    if p16_ckpt.exists():
        print(f"[WARM-START] Loading initial weights from Phase 16: {p16_ckpt.name}")
        p16_state = torch.load(p16_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(p16_state.get("model_state_dict", p16_state), strict=False)

    # Generator-balanced loss weights:
    # Boost modern diffusion (Gemini, SDXL, FLUX) while protecting real-image specificity
    gen_weights = {
        "FFHQ_Real": 1.0,
        "IMDB_WIKI": 1.25,              # Increased weight on vintage genuine images to suppress false alarms
        "StyleGAN2": 0.70,              # Downweight GAN to prevent checkerboard over-reliance
        "StableDiffusion_v1.5": 1.0,
        "InsightFace_Swap": 1.0,
        "StableDiffusion_XL": 1.50,     # Boost modern latent diffusion
        "FLUX1_Dev": 1.40,              # Boost diffusion transformer
        "FLUX1_Pro": 1.50,              # Boost modern DiT
        "Google_Gemini_Imagen3": 2.50   # Strong supervision on photorealistic Gemini diffusion
    }

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-4)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_composite_score = 0.0
    best_epoch = 1
    best_val_metrics = {}
    history = []

    print(f"\n[TRAINING START] {epochs} Epochs on {len(train_ds)} samples...")
    t0_all = time.time()

    for epoch in range(1, epochs + 1):
        t0_ep = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            gens = batch["generator"]

            batch_w = torch.tensor([gen_weights.get(g, 1.0) for g in gens], device=device, dtype=torch.float32)
            weighted_bce = nn.BCEWithLogitsLoss(weight=batch_w)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    fused_logit, meta = model(imgs)
                    loss_fused = weighted_bce(fused_logit, labels)
                    loss_sp = bce(meta["spatial_logit"], labels)
                    loss_freq = bce(meta["frequency_logit"], labels)
                    loss = loss_fused + 0.30 * loss_sp + 0.20 * loss_freq

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                fused_logit, meta = model(imgs)
                loss_fused = weighted_bce(fused_logit, labels)
                loss_sp = bce(meta["spatial_logit"], labels)
                loss_freq = bce(meta["frequency_logit"], labels)
                loss = loss_fused + 0.30 * loss_sp + 0.20 * loss_freq
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)
        ep_time = time.time() - t0_ep

        # Validation Evaluation
        val_eval = evaluate_model(model, val_loader, device, temp=1.0, threshold=0.55)
        # Composite score balancing ROC-AUC, Specificity on genuine faces, and Fake Recall
        composite_score = (
            0.40 * val_eval["roc_auc"] +
            0.35 * val_eval["decisive_specificity"] +
            0.25 * val_eval["decisive_recall"]
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_roc_auc": val_eval["roc_auc"],
            "val_decisive_acc": val_eval["decisive_accuracy"],
            "val_decisive_spec": val_eval["decisive_specificity"],
            "val_decisive_recall": val_eval["decisive_recall"],
            "composite_score": round(composite_score, 4),
            "time_seconds": round(ep_time, 2)
        })

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) - Loss: {train_loss:.4f} | "
              f"Val AUC: {val_eval['roc_auc']:.4f} | Val Spec: {val_eval['decisive_specificity']*100:.1f}% | "
              f"Val Recall: {val_eval['decisive_recall']*100:.1f}% | Composite: {composite_score:.4f}")

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_epoch = epoch
            best_val_metrics = val_eval

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_roc_auc": val_eval["roc_auc"],
                "composite_score": composite_score,
                "val_specificity": val_eval["decisive_specificity"],
                "val_fake_recall": val_eval["decisive_recall"],
                "training_config": {
                    "batch_size": batch_size,
                    "lr": 2.5e-5,
                    "min_floor": 0.25,
                    "seed": 42
                }
            }, best_ckpt_path)
            print(f"   -> [BEST CHECKPOINT SAVED] Epoch {epoch} with Composite Score: {composite_score:.4f}")

    total_train_time = time.time() - t0_all
    print(f"\n[TRAINING FINISHED] Total Time: {total_train_time:.1f}s. Best Epoch: {best_epoch} (Score: {best_composite_score:.4f})")

    # =========================================================================
    # Post-Training: Calibration on Validation Set Only
    # =========================================================================
    print("\n[CALIBRATION] Calibrating Temperature on Validation Set...")
    best_checkpoint_data = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint_data["model_state_dict"])
    model.eval()

    val_logits = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            logits, _ = model(imgs)
            val_logits.extend(logits.cpu().numpy().tolist())
            val_labels.extend(batch["label"].numpy().tolist())

    val_logits = np.array(val_logits)
    val_labels = np.array(val_labels)

    optimal_t, scaler = fit_temperature(val_logits, val_labels)
    calibrated_val_probs = 1.0 / (1.0 + np.exp(-val_logits / optimal_t))

    # Sweep thresholds on validation set to maximize F1 while maintaining >= 75% specificity
    best_thresh = 0.55
    best_f1 = 0.0
    for th in np.arange(0.45, 0.70, 0.01):
        preds = (calibrated_val_probs >= th).astype(int)
        cm_th = confusion_matrix(val_labels, preds, labels=[0, 1])
        tn_th, fp_th, fn_th, tp_th = cm_th.ravel() if cm_th.size == 4 else (0, 0, 0, 0)
        spec_th = tn_th / max(1, tn_th + fp_th)
        f1_th = f1_score(val_labels, preds, zero_division=0)
        if spec_th >= 0.75 and f1_th > best_f1:
            best_f1 = f1_th
            best_thresh = float(th)

    print(f"Optimal Temperature: T = {optimal_t:.4f}")
    print(f"Optimal Calibrated Threshold on Validation: tau = {best_thresh:.4f} (Val F1 = {best_f1:.4f})")

    # Save Calibration Parameters
    cal_params_path = exp_dir / "calibration_params_phase17.json"
    with open(cal_params_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "phase17",
            "temperature": round(optimal_t, 4),
            "optimal_threshold": round(best_thresh, 4),
            "uncertainty_band": 0.15,
            "validation_samples": len(val_labels),
            "fitted_on": "Phase 17 Validation Split"
        }, f, indent=2)

    # Compute checkpoint SHA-256
    sha256 = hashlib.sha256()
    with open(best_ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    ckpt_hash = sha256.hexdigest()

    # Save training logs and validation history
    training_results = {
        "metadata": {
            "model_id": "phase17",
            "model_name": "LX-DFD Phase 17 Multi-Generator Retrained",
            "checkpoint_path": str(best_ckpt_path.relative_to(repo_root)).replace("\\", "/"),
            "checkpoint_sha256": ckpt_hash,
            "best_epoch": best_epoch,
            "training_duration_seconds": round(total_train_time, 2),
            "calibrated_temperature": round(optimal_t, 4),
            "calibrated_threshold": round(best_thresh, 4),
            "device": str(device)
        },
        "training_history": history,
        "best_validation_metrics": best_val_metrics
    }

    train_results_json = exp_dir / "phase17_training_results.json"
    with open(train_results_json, "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2)

    print(f"\n[SAVED COMPLETE] Checkpoint: {best_ckpt_path}")
    print(f"  SHA-256: {ckpt_hash}")

    # =========================================================================
    # Final Locked Evaluation on Held-Out Test Set (N = 543)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON LOCKED HELD-OUT TEST SET (N = 543)")
    print("=" * 80)

    test_eval = evaluate_model(model, test_loader, device, temp=optimal_t, threshold=best_thresh)
    print(f"Test ROC-AUC:              {test_eval['roc_auc']:.4f}")
    print(f"Test PR-AUC:               {test_eval['pr_auc']:.4f}")
    print(f"Test Decisive Accuracy:    {test_eval['decisive_accuracy']*100:.2f}%")
    print(f"Test Decisive Fake Recall: {test_eval['decisive_recall']*100:.2f}%")
    print(f"Test Decisive Specificity: {test_eval['decisive_specificity']*100:.2f}%")
    print(f"Test Decisive Coverage:    {test_eval['coverage']*100:.2f}%")
    print(f"Test Inconclusive Rate:    {test_eval['inconclusive_rate']*100:.2f}%")
    print(f"Decisive Confusion Matrix: {test_eval['cm_decisive']}")

    # Per-generator breakdown on test set
    per_gen_test = {}
    test_probs = test_eval["all_probs"]
    test_labels = test_eval["all_labels"]
    test_gens = np.array(test_eval["all_gens"])

    for gen in sorted(np.unique(test_gens)):
        mask = (test_gens == gen)
        g_probs = test_probs[mask]
        g_lbls = test_labels[mask]
        g_gt = "REAL" if g_lbls[0] == 0 else "FAKE"

        dec_tp, dec_tn, dec_fp, dec_fn, inconc = 0, 0, 0, 0, 0
        bin_tp, bin_tn, bin_fp, bin_fn = 0, 0, 0, 0

        for p, y in zip(g_probs, g_lbls):
            # Binary
            if p >= best_thresh:
                if y == 1: bin_tp += 1
                else: bin_fp += 1
            else:
                if y == 0: bin_tn += 1
                else: bin_fn += 1

            # 3-Way
            if abs(p - best_thresh) <= 0.15:
                inconc += 1
            elif p >= best_thresh:
                if y == 1: dec_tp += 1
                else: dec_fp += 1
            else:
                if y == 0: dec_tn += 1
                else: dec_fn += 1

        g_dec_total = dec_tp + dec_tn + dec_fp + dec_fn
        if g_gt == "REAL":
            dec_acc = dec_tn / max(1, g_dec_total) if g_dec_total > 0 else 0.0
            bin_acc = bin_tn / max(1, len(g_lbls))
        else:
            dec_acc = dec_tp / max(1, g_dec_total) if g_dec_total > 0 else 0.0
            bin_acc = bin_tp / max(1, len(g_lbls))

        per_gen_test[gen] = {
            "total_test": len(g_lbls),
            "ground_truth": g_gt,
            "mean_fake_prob": round(float(np.mean(g_probs)), 4),
            "inconclusive_count": inconc,
            "inconclusive_rate": round(inconc / len(g_lbls), 4),
            "decisive_count": g_dec_total,
            "decisive_tp": dec_tp,
            "decisive_tn": dec_tn,
            "decisive_fp": dec_fp,
            "decisive_fn": dec_fn,
            "decisive_accuracy": round(dec_acc, 4),
            "binary_accuracy": round(bin_acc, 4)
        }
        print(f"  * {gen} ({g_gt}, N={len(g_lbls)}): Mean P={np.mean(g_probs):.4f} | Decisive Acc={dec_acc*100:.1f}% | Inconc={inconc} ({inconc/len(g_lbls)*100:.1f}%)")

    test_results_summary = {
        "metadata": training_results["metadata"],
        "overall_test_metrics": {
            "roc_auc": test_eval["roc_auc"],
            "pr_auc": test_eval["pr_auc"],
            "brier_score": test_eval["brier_score"],
            "ece": test_eval["ece"],
            "binary_accuracy": test_eval["binary_accuracy"],
            "binary_specificity": test_eval["binary_specificity"],
            "binary_recall": test_eval["binary_recall"],
            "binary_f1": test_eval["binary_f1"],
            "decisive_accuracy": test_eval["decisive_accuracy"],
            "decisive_recall": test_eval["decisive_recall"],
            "decisive_specificity": test_eval["decisive_specificity"],
            "coverage": test_eval["coverage"],
            "inconclusive_rate": test_eval["inconclusive_rate"],
            "cm_decisive": test_eval["cm_decisive"],
            "cm_binary": test_eval["cm_binary"]
        },
        "per_generator": per_gen_test
    }

    test_summary_path = exp_dir / "phase17_test_evaluation.json"
    with open(test_summary_path, "w", encoding="utf-8") as f:
        json.dump(test_results_summary, f, indent=2)

    print(f"\nSaved test evaluation to {test_summary_path}")

if __name__ == "__main__":
    train_phase17()
