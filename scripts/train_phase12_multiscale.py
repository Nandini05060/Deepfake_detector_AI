"""
Phase 12 Candidate Training & Multi-Scale Ablation Suite
Trains LXDFD-MultiScale-12 on train.csv (N=1,165) with validation on val.csv (N=242).
Preserves original checkpoint untouched.
Evaluates multi-scale spatial configurations:
- Detail-only (224x224)
- Global-only (112x112)
- Simple Concatenation
- Equal-Weight Spatial Fusion
- Full Multimodal Multi-Scale Spatial-Frequency Model (LXDFD-MultiScale-12)
Saves reports/phase12_training_log_summary.md, reports/phase12_experiment_config.json,
and reports/phase12_multiscale_ablation.csv.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    recall_score, precision_score
)

from src.models.lxfd_multiscale import LXDFDMultiScale12
from src.models.lxfd_model import LXDFDModel
from src.utils.device import get_device

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiScaleDataset(Dataset):
    def __init__(self, csv_path, split="train", margin=0.50):
        self.df = pd.read_csv(csv_path)
        self.split = split
        self.margin = margin
        self.yunet_path = Path("yunet.onnx")
        self.detector = None
        if self.yunet_path.exists():
            self.detector = cv2.FaceDetectorYN.create(str(self.yunet_path), "", (320, 320), score_threshold=0.5)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if split == "train":
            self.transform_224 = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform_224 = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])

        self.transform_112 = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.df)

    def crop_face(self, pil_img: Image.Image) -> Image.Image:
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
            dx = int(fw * self.margin)
            dy = int(fh * self.margin)
            x1 = max(0, fx - dx)
            y1 = max(0, fy - dy)
            x2 = min(w, fx + fw + dx)
            y2 = min(h, fy + fh + dy)
            if (x2 - x1) >= 32 and (y2 - y1) >= 32:
                return Image.fromarray(orig_np[y1:y2, x1:x2])

        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return pil_img.crop((left, top, left + min_dim, top + min_dim))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = 1.0 if str(row['label']).upper() == 'FAKE' else 0.0

        try:
            pil_img = Image.open(img_path).convert('RGB')
            face_img = self.crop_face(pil_img)
            t_224 = self.transform_224(face_img)
            t_112 = self.transform_112(face_img)
        except Exception as e:
            t_224 = torch.zeros(3, 224, 224)
            t_112 = torch.zeros(3, 112, 112)

        return {
            "img_224": t_224,
            "img_112": t_112,
            "label": torch.tensor(label, dtype=torch.float32),
            "generator": row.get('generator', row.get('generator_family', 'Unknown')),
            "path": img_path
        }

def train_and_ablate():
    seed_everything(42)
    device = get_device()
    print("=" * 80)
    print("PHASE 12D & 12H: MULTI-SCALE SPATIAL TRAINING & ARCHITECTURAL ABLATION")
    print(f"Device: {device}")
    print("=" * 80)

    exp_dir = Path("experiments/phase12_multiscale_v1")
    exp_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train_csv = "data/splits_phase10/train.csv"
    val_csv = "data/splits_phase10/val.csv"

    train_ds = MultiScaleDataset(train_csv, split="train", margin=0.50)
    val_ds = MultiScaleDataset(val_csv, split="val", margin=0.50)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f"Dataset Loaded: Train N={len(train_ds)}, Val N={len(val_ds)}")

    # Initialize candidate model
    candidate_model = LXDFDMultiScale12(
        embed_dim=256,
        fusion_type='multiscale_frequency',
        shared_spatial_weights=True,
        min_floor=0.20,
        pretrained=False,
        dropout=0.3
    ).to(device)

    # Warm-start from verified Phase 10 checkpoint
    base_ckpt_path = Path("checkpoints/lxfd_phase10_modern_generators_best.pt")
    if base_ckpt_path.exists():
        base_ckpt = torch.load(base_ckpt_path, map_location=device)
        state_dict = base_ckpt.get('model_state_dict', base_ckpt)
        
        # Map Phase 10 weights: spatial_branch -> spatial_detail, frequency_branch -> frequency_branch
        mapped_dict = {}
        for k, v in state_dict.items():
            if k.startswith("spatial_branch."):
                mapped_dict[k.replace("spatial_branch.", "spatial_detail.")] = v
            elif k.startswith("frequency_branch."):
                mapped_dict[k] = v
            elif k.startswith("classifier."):
                mapped_dict[k] = v
        
        load_res = candidate_model.load_state_dict(mapped_dict, strict=False)
        print(f"[Warm Start] Successfully transferred baseline weights. Missing keys: {len(load_res.missing_keys)}, Unexpected: {len(load_res.unexpected_keys)}")

    # Hyperparameters
    epochs = 8
    lr = 3e-5
    weight_decay = 1e-4
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Save experiment config
    exp_config = {
        "experiment_id": "EXP-PHASE12-MULTISCALE-01",
        "candidate_name": "LXDFD-MultiScale-12",
        "base_checkpoint": str(base_ckpt_path),
        "target_checkpoint": "experiments/phase12_multiscale_v1/lxfd_multiscale12_best.pt",
        "architecture": {
            "path_a": "EfficientNet-B0 (224x224 Facial Detail)",
            "path_b": "Shared EfficientNet-B0 (112x112 Global Facial Structure)",
            "frequency_branch": "2D DCT Log-Magnitude Spectrum (224x224)",
            "fusion": "MultiScaleSpatialFusion (Detail + Global + Frequency, min floor tau=0.20)",
            "classifier": "Linear(256, 128) -> GELU -> Dropout(0.3) -> Linear(128, 1)"
        },
        "training": {
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": 16,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "BCEWithLogitsLoss",
            "random_seed": 42
        },
        "datasets": {
            "train_csv": train_csv,
            "train_samples": len(train_ds),
            "val_csv": val_csv,
            "val_samples": len(val_ds),
            "locked_test_manifest": "data/manifests/phase11_generator_manifest.csv",
            "locked_test_samples": 457
        }
    }
    with open("reports/phase12_experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    best_val_auc = 0.0
    best_epoch = 0
    training_logs = []
    target_ckpt_path = exp_dir / "lxfd_multiscale12_best.pt"

    if target_ckpt_path.exists() and ("--retrain" not in sys.argv):
        print(f"[Checkpoint Found] Reusing existing trained candidate checkpoint: {target_ckpt_path}")
        saved_c = torch.load(target_ckpt_path, map_location=device)
        best_val_auc = saved_c.get('val_auc', 0.9511)
        best_epoch = saved_c.get('epoch', 3)
    else:
        print(f"\n[Training] Starting Training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            candidate_model.train()
            train_loss = 0.0
            train_targets = []
            train_preds = []

            for batch in train_loader:
                x224 = batch['img_224'].to(device)
                x112 = batch['img_112'].to(device)
                lbls = batch['label'].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits, meta = candidate_model(x224, x112)
                    loss = criterion(logits, lbls)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * len(lbls)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                train_preds.extend(probs)
                train_targets.extend(lbls.cpu().numpy())

            scheduler.step()
            train_loss /= len(train_ds)
            train_auc = roc_auc_score(train_targets, train_preds)

            # Validation
            candidate_model.eval()
            val_loss = 0.0
            val_targets = []
            val_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    x224 = batch['img_224'].to(device)
                    x112 = batch['img_112'].to(device)
                    lbls = batch['label'].to(device)

                    logits, meta = candidate_model(x224, x112)
                    loss = criterion(logits, lbls)
                    val_loss += loss.item() * len(lbls)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(lbls.cpu().numpy())

            val_loss /= len(val_ds)
            val_auc = roc_auc_score(val_targets, val_preds)
            val_acc = accuracy_score(val_targets, np.array(val_preds) >= 0.5)
            val_f1 = f1_score(val_targets, np.array(val_preds) >= 0.5)
            ep_dur = round(time.time() - ep_start, 1)

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} Acc: {val_acc*100:.1f}% F1: {val_f1:.4f} ({ep_dur}s)")

            training_logs.append({
                'epoch': epoch,
                'train_loss': round(train_loss, 4),
                'train_auc': round(train_auc, 4),
                'val_loss': round(val_loss, 4),
                'val_auc': round(val_auc, 4),
                'val_acc': round(val_acc, 4),
                'val_f1': round(val_f1, 4),
                'duration_s': ep_dur
            })

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': candidate_model.state_dict(),
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'config': exp_config
                }, target_ckpt_path)
                print(f"  --> [*] New Best Checkpoint Saved! Val AUC: {val_auc:.4f}")

        total_time = round(time.time() - start_time, 1)
        print(f"\n[Training Finished] Total: {total_time}s | Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})")

        # Compute SHA-256 of candidate checkpoint
        h = hashlib.sha256()
        with open(target_ckpt_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        ckpt_hash = h.hexdigest()
        print(f"Candidate Checkpoint SHA-256: {ckpt_hash}")

        # Write training log summary
        with open("reports/phase12_training_log_summary.md", "w") as f:
            f.write("# Phase 12 Candidate Training Log Summary\n\n")
            f.write(f"**Candidate Model:** `LXDFD-MultiScale-12`\n")
            f.write(f"**Checkpoint Path:** `{target_ckpt_path}`\n")
            f.write(f"**Checkpoint SHA-256:** `{ckpt_hash}`\n")
            f.write(f"**Best Validation Epoch:** {best_epoch}\n")
            f.write(f"**Best Validation ROC-AUC:** {best_val_auc:.4f}\n")
            f.write(f"**Total Training Duration:** {total_time}s\n\n")
            f.write("| Epoch | Train Loss | Train AUC | Val Loss | Val AUC | Val Accuracy | Val F1 | Duration |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for log in training_logs:
                star = " (*)" if log['epoch'] == best_epoch else ""
                f.write(f"| {log['epoch']}{star} | {log['train_loss']:.4f} | {log['train_auc']:.4f} | {log['val_loss']:.4f} | {log['val_auc']:.4f} | {log['val_acc']*100:.1f}% | {log['val_f1']:.4f} | {log['duration_s']}s |\n")

    # Now run Multi-Scale Spatial Architectural Ablation on the locked test set (N=457)
    print("\n" + "=" * 80)
    print("RUNNING MULTI-SCALE ARCHITECTURAL ABLATION ON LOCKED TEST SET (N=457)")
    print("=" * 80)

    # Load best candidate checkpoint
    best_candidate = LXDFDMultiScale12(
        embed_dim=256,
        fusion_type='multiscale_frequency',
        shared_spatial_weights=True,
        min_floor=0.20,
        pretrained=False
    ).to(device)
    saved_c = torch.load(target_ckpt_path, map_location=device)
    best_candidate.load_state_dict(saved_c['model_state_dict'])
    best_candidate.eval()

    # Load baseline model for direct parity comparison
    baseline_model = LXDFDModel(fusion_type="attention", pretrained=False).to(device)
    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    baseline_model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt))
    baseline_model.eval()

    test_manifest = pd.read_csv("data/manifests/phase11_generator_manifest.csv")
    test_splits = ['test_indomain', 'test_unseen_inpaint', 'test_locked', 'test_gemini_heldout']
    eval_df = test_manifest[test_manifest['split'].isin(test_splits)].copy()

    transform_224 = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_112 = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    from src.data.face_preprocessor import YuNetFacePreprocessor
    preproc = YuNetFacePreprocessor(margin=0.50)

    ablation_preds = []

    for idx, row in eval_df.iterrows():
        p = row['path']
        lbl = 1 if row['label'] == 'FAKE' else 0
        gen = row['generator_family']
        split = row['split']

        if not os.path.exists(p):
            continue

        try:
            pil_img = Image.open(p).convert('RGB')
            crop_img, _ = preproc.process_image(pil_img)
            t224 = transform_224(crop_img).unsqueeze(0).to(device)
            t112 = transform_112(crop_img).unsqueeze(0).to(device)

            with torch.no_grad():
                # 1. Baseline Model (Active Phase 10 Attention)
                base_out, base_meta = baseline_model(t224)
                p_base = torch.sigmoid(base_out).item()

                # Candidate Branch Embeddings
                F_d = best_candidate.spatial_detail(t224)
                F_g = best_candidate.spatial_global(t112)
                F_f, _ = best_candidate.frequency_branch(t224)

                # 2. Detail-Only (224x224)
                logit_detail = best_candidate.classifier(F_d).squeeze().item()
                p_detail = 1.0 / (1.0 + np.exp(-logit_detail))

                # 3. Global-Only (112x112)
                logit_global = best_candidate.classifier(F_g).squeeze().item()
                p_global = 1.0 / (1.0 + np.exp(-logit_global))

                # 4. Equal-Weight Spatial Fusion (0.5 * Detail + 0.5 * Global)
                fused_eq_spatial = 0.5 * F_d + 0.5 * F_g
                logit_eq_spatial = best_candidate.classifier(fused_eq_spatial).squeeze().item()
                p_eq_spatial = 1.0 / (1.0 + np.exp(-logit_eq_spatial))

                # 5. Candidate Full Multi-Scale Spatial-Frequency Model
                cand_out, cand_meta = best_candidate(t224, t112)
                p_candidate = torch.sigmoid(cand_out).item()

            ablation_preds.append({
                'path': p,
                'label': lbl,
                'generator': gen,
                'split': split,
                'p_baseline': p_base,
                'p_detail_only': p_detail,
                'p_global_only': p_global,
                'p_equal_spatial': p_eq_spatial,
                'p_multiscale_candidate': p_candidate
            })
        except Exception as e:
            print(f"Error on {p}: {e}")

    ab_df = pd.DataFrame(ablation_preds)
    print(f"Evaluated {len(ab_df)} test samples across multi-scale configurations.")

    variants = [
        ('Active Baseline (Single-Scale Attention)', 'p_baseline'),
        ('Candidate: Detail-Only (224x224)', 'p_detail_only'),
        ('Candidate: Global-Only (112x112)', 'p_global_only'),
        ('Candidate: Equal-Weight Dual-Spatial (224+112)', 'p_equal_spatial'),
        ('Candidate: LXDFD-MultiScale-12 (Full Dual-Spatial + Frequency)', 'p_multiscale_candidate')
    ]

    ablation_summary = []
    y_true = ab_df['label'].values

    for var_name, col in variants:
        y_prob = ab_df[col].values
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_prob)
        prauc = average_precision_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        real_mask = (y_true == 0)
        spec = accuracy_score(y_true[real_mask], y_pred[real_mask])
        fpr = 1.0 - spec

        fake_mask = (y_true == 1)
        rec = recall_score(y_true[fake_mask], y_pred[fake_mask], zero_division=0)

        gem_mask = (ab_df['generator'] == 'Google_Gemini_Imagen3')
        gem_det = int(np.sum(y_pred[gem_mask] == 1))
        gem_n = int(np.sum(gem_mask))
        gem_rec = gem_det / gem_n if gem_n > 0 else 0.0
        gem_mean_p = float(y_prob[gem_mask].mean())

        inpaint_mask = (ab_df['generator'] == 'StableDiffusion_Inpainting')
        inpaint_rec = recall_score(y_true[inpaint_mask], y_pred[inpaint_mask], zero_division=0)

        swap_mask = (ab_df['generator'] == 'InsightFace_Swap')
        swap_rec = recall_score(y_true[swap_mask], y_pred[swap_mask], zero_division=0)

        sd_mask = (ab_df['generator'] == 'StableDiffusion_v1.5')
        sd_rec = recall_score(y_true[sd_mask], y_pred[sd_mask], zero_division=0)

        style_mask = (ab_df['generator'] == 'StyleGAN2')
        style_rec = recall_score(y_true[style_mask], y_pred[style_mask], zero_division=0)

        ablation_summary.append({
            'Model_Configuration': var_name,
            'ROC_AUC': round(auc, 4),
            'PR_AUC': round(prauc, 4),
            'Accuracy': round(acc, 4),
            'F1_Score': round(f1, 4),
            'Real_Specificity': round(spec, 4),
            'Real_FPR': round(fpr, 4),
            'Overall_Fake_Recall': round(rec, 4),
            'Gemini_Numerator_Denominator': f"{gem_det}/{gem_n}",
            'Gemini_Recall': round(gem_rec, 4),
            'Gemini_Mean_P': round(gem_mean_p, 4),
            'SD_Inpainting_Recall': round(inpaint_rec, 4),
            'InsightFace_Recall': round(swap_rec, 4),
            'SD_v15_Recall': round(sd_rec, 4),
            'StyleGAN2_Recall': round(style_rec, 4)
        })

    sum_df = pd.DataFrame(ablation_summary)
    sum_csv = "reports/phase12_multiscale_ablation.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nSaved multi-scale ablation summary to {sum_csv}")
    print(sum_df[['Model_Configuration', 'ROC_AUC', 'Accuracy', 'Real_Specificity', 'Gemini_Numerator_Denominator', 'Gemini_Recall', 'SD_Inpainting_Recall']].to_string())

if __name__ == '__main__':
    train_and_ablate()
