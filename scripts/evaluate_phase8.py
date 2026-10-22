import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
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

def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.05) -> dict:
    rng = np.random.RandomState(42)
    acc_scores = []
    auc_scores = []
    n = len(y_true)
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

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    unique_classes = np.unique(y_true)
    if len(unique_classes) >= 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = 0.5
    else:
        roc_auc = None
        pr_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)
    ll = float(log_loss(y_true, y_prob_clipped))
    ci = compute_bootstrap_ci(y_true, y_prob) if len(unique_classes) >= 2 else {}

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": cm,
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "log_loss": round(ll, 4),
        "ci_95": ci
    }

class Phase8EvalDataset(Dataset):
    """
    Standardized evaluation dataset supporting square center-cropping
    and aspect-ratio preservation to prevent frequency squashing artifacts.
    """
    def __init__(self, samples: List[dict], image_size: Tuple[int, int] = (224, 224), corruption_fn=None, legacy_mode: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.corruption_fn = corruption_fn
        self.legacy_mode = legacy_mode
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        img_path = item["path"]

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            w, h = rgb_img.size

            if not self.legacy_mode:
                # Aspect-ratio preserving square crop if non-square
                if w != h:
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    rgb_img = rgb_img.crop((left, top, left + min_dim, top + min_dim))

            if self.corruption_fn is not None:
                rgb_img = self.corruption_fn(rgb_img)

            # Resize to model expected size
            rgb_img = rgb_img.resize(self.image_size, Image.BILINEAR)
            tensor_img = TF.to_tensor(rgb_img)
            tensor_img = self.normalize(tensor_img)

        return {
            "image": tensor_img,
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "path": img_path,
            "generator": str(item.get("generator", "None")),
            "source_dataset": str(item.get("source_dataset", "unknown"))
        }

def load_checkpoint_model(ckpt_path: str, device: torch.device):
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_name = p.name.lower()
    checkpoint = torch.load(p, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if "efficientnet" in ckpt_name:
        model = EfficientNetBaseline(pretrained=False).to(device)
    elif "dense" in ckpt_name:
        model = DenseCNNBaseline().to(device)
    elif "simple" in ckpt_name:
        model = SimpleCNN().to(device)
    else:
        # LX-DFD Dual-Branch
        is_spectral = "spectral_dropout" in ckpt_name
        model = LXDFDModel(embed_dim=256, fusion_type="attention", pretrained=False, spectral_dropout=is_spectral).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def run_model_inference(model, loader, device):
    all_targets = []
    all_probs = []
    all_paths = []
    all_gens = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].cpu().numpy()
            paths = batch["path"]
            gens = batch["generator"]

            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_targets.extend(lbls)
            all_probs.extend(probs)
            all_paths.extend(paths)
            all_gens.extend(gens)

    return np.array(all_targets), np.array(all_probs), all_paths, all_gens

def load_split_samples(csv_path: str) -> List[dict]:
    df = pd.read_csv(csv_path)
    samples = []
    for _, r in df.iterrows():
        lbl_str = str(r["label"]).upper()
        lbl_val = 1.0 if lbl_str in ["FAKE", "1", "1.0"] else 0.0
        samples.append({
            "path": str(r["path"]),
            "label": lbl_val,
            "generator": str(r.get("generator", "None")),
            "source_dataset": str(r.get("source_dataset", r.get("dataset", "unknown")))
        })
    return samples

# Standardized 14 Corruption Conditions
def get_corruption_functions():
    corruptions = {}
    import io

    # 1. JPEG Compression (Q90, Q75, Q50, Q30)
    for q in [90, 75, 50, 30]:
        def make_jpeg_fn(quality=q):
            def jpeg_fn(img: Image.Image) -> Image.Image:
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=quality)
                buf.seek(0)
                return Image.open(buf).convert('RGB')
            return jpeg_fn
        corruptions[f"jpeg_q{q}"] = make_jpeg_fn(q)

    # 2. Resolution Downsampling (0.75x, 0.50x)
    for s in [0.75, 0.50]:
        def make_downsample_fn(scale=s):
            def downsample_fn(img: Image.Image) -> Image.Image:
                w, h = img.size
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                down = img.resize((nw, nh), Image.BILINEAR)
                return down.resize((w, h), Image.BILINEAR)
            return downsample_fn
        corruptions[f"downsample_{int(s*100)}"] = make_downsample_fn(s)

    # 3. Gaussian Blur (r=1.5, r=3.0)
    for r in [1.5, 3.0]:
        def make_blur_fn(radius=r):
            def blur_fn(img: Image.Image) -> Image.Image:
                return img.filter(ImageFilter.GaussianBlur(radius=radius))
            return blur_fn
        corruptions[f"blur_r{str(r).replace('.', '_')}"] = make_blur_fn(r)

    # 4. Gaussian Noise (sigma=0.01, 0.05, 0.10)
    for sig in [0.01, 0.05, 0.10]:
        def make_noise_fn(sigma=sig):
            def noise_fn(img: Image.Image) -> Image.Image:
                arr = np.array(img, dtype=np.float32) / 255.0
                rng = np.random.RandomState(42)
                noise = rng.normal(0, sigma, arr.shape)
                noisy = np.clip(arr + noise, 0.0, 1.0) * 255.0
                return Image.fromarray(noisy.astype(np.uint8))
            return noise_fn
        corruptions[f"noise_sig{int(sig*100)}"] = make_noise_fn(sig)

    # 5. Brightness / Contrast Perturbations (0.7x, 1.3x)
    for b in [0.7, 1.3]:
        def make_bright_fn(factor=b):
            def bright_fn(img: Image.Image) -> Image.Image:
                return ImageEnhance.Brightness(img).enhance(factor)
            return bright_fn
        corruptions[f"brightness_{int(b*100)}"] = make_bright_fn(b)

    return corruptions

def run_phase8_benchmarks():
    print("=" * 80)
    print("                PHASE 8: COMPREHENSIVE BENCHMARK EVALUATION              ")
    print("=" * 80)

    device = get_device()
    print(f"[*] Evaluation Device: {device}")

    # Checkpoint paths
    models_to_eval = [
        {"id": "efficientnet", "name": "EfficientNet-B0 Baseline", "path": "checkpoints/efficientnet_best.pt"},
        {"id": "lxfd_standard", "name": "LX-DFD Standard", "path": "checkpoints/lxfd_attention_standard_best.pt"},
        {"id": "lxfd_robustness_v2", "name": "LX-DFD Robustness v2", "path": "checkpoints/lxfd_attention_robustness_v2_best.pt"},
        {"id": "lxfd_multigen", "name": "LX-DFD Multi-Generator", "path": "checkpoints/lxfd_multigen_diverse_best.pt"}
    ]

    # Verify checkpoints exist
    for m in models_to_eval:
        if not Path(m["path"]).exists():
            raise FileNotFoundError(f"Checkpoint for '{m['name']}' not found at: {m['path']}")

    # 1. Load Datasets
    print("\n[*] Loading Evaluation Datasets...")
    indomain_samples = load_split_samples("data/splits_multigen/test_indomain.csv")
    logo_samples = load_split_samples("data/splits_multigen/test_unseen_logo.csv")
    eval_samples = load_split_samples("data/splits_multigen/test_unseen_eval.csv")
    exploratory_samples = load_split_samples("data/external/manifest.csv")

    print(f"  - In-Domain Held-Out:   {len(indomain_samples)} samples")
    print(f"  - LOGO Unseen Inpaint:  {len(logo_samples)} samples")
    print(f"  - Seen-Family Eval:     {len(eval_samples)} samples")
    print(f"  - Exploratory 800-set:  {len(exploratory_samples)} samples")

    batch_size = 32
    results = {}
    comparison_rows = []
    corruption_rows = []

    corruptions = get_corruption_functions()

    for m_info in models_to_eval:
        m_id = m_info["id"]
        m_name = m_info["name"]
        print("\n" + "-" * 80)
        print(f"[*] Evaluating Model: {m_name} ({m_info['path']})")
        print("-" * 80)

        model = load_checkpoint_model(m_info["path"], device)
        results[m_id] = {"name": m_name, "checkpoint": m_info["path"]}

        # Suite 1: In-Domain Clean Test
        ds_in = Phase8EvalDataset(indomain_samples)
        loader_in = DataLoader(ds_in, batch_size=batch_size, shuffle=False)
        y_true_in, y_prob_in, _, _ = run_model_inference(model, loader_in, device)
        m_in = evaluate_predictions(y_true_in, y_prob_in)
        results[m_id]["in_domain"] = m_in
        print(f"  -> In-Domain:      ROC-AUC={m_in['roc_auc']:.4f} | Acc={m_in['accuracy']:.4f} | F1={m_in['f1']:.4f} | ECE={m_in['ece']:.4f}")

        # Suite 2: LOGO Unseen Inpainting Test (100 Real IMDB-WIKI + 200 Fake Inpainting)
        ds_logo = Phase8EvalDataset(logo_samples)
        loader_logo = DataLoader(ds_logo, batch_size=batch_size, shuffle=False)
        y_true_logo, y_prob_logo, _, _ = run_model_inference(model, loader_logo, device)
        m_logo = evaluate_predictions(y_true_logo, y_prob_logo)
        results[m_id]["logo_inpainting"] = m_logo
        print(f"  -> LOGO Inpaint:   ROC-AUC={m_logo['roc_auc']:.4f} | Acc={m_logo['accuracy']:.4f} | F1={m_logo['f1']:.4f} | Rec={m_logo['recall']:.4f}")

        # Suite 3: Seen-Family External Test (15 Real IMDB-WIKI + 30 SD1.5 + 30 InsightFace)
        ds_eval = Phase8EvalDataset(eval_samples)
        loader_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False)
        y_true_eval, y_prob_eval, _, _ = run_model_inference(model, loader_eval, device)
        m_eval = evaluate_predictions(y_true_eval, y_prob_eval)
        results[m_id]["seen_family_eval"] = m_eval
        print(f"  -> Seen-Family:    ROC-AUC={m_eval['roc_auc']:.4f} | Acc={m_eval['accuracy']:.4f} | F1={m_eval['f1']:.4f}")

        # Suite 4: Exploratory Benchmark Disaggregated Per Generator (SD1.5, InsightFace, Inpainting)
        ds_exp = Phase8EvalDataset(exploratory_samples)
        loader_exp = DataLoader(ds_exp, batch_size=batch_size, shuffle=False)
        y_true_exp, y_prob_exp, _, gens_exp = run_model_inference(model, loader_exp, device)

        exp_df = pd.DataFrame({"y_true": y_true_exp, "y_prob": y_prob_exp, "generator": gens_exp})
        real_subset = exp_df[exp_df["y_true"] == 0]

        results[m_id]["exploratory_per_generator"] = {}
        gen_aucs = []

        for gen in ["StableDiffusion_v1.5", "InsightFace_Swap", "StableDiffusion_Inpainting"]:
            gen_subset = exp_df[exp_df["generator"] == gen]
            combined = pd.concat([real_subset, gen_subset], ignore_index=True)
            m_gen = evaluate_predictions(combined["y_true"].values, combined["y_prob"].values)
            results[m_id]["exploratory_per_generator"][gen] = m_gen
            gen_aucs.append(m_gen["roc_auc"])
            print(f"     * {gen:26s}: ROC-AUC={m_gen['roc_auc']:.4f} | Acc={m_gen['accuracy']:.4f}")

        macro_auc = float(np.mean(gen_aucs))
        results[m_id]["exploratory_macro_auc"] = round(macro_auc, 4)
        print(f"  -> Exploratory Macro ROC-AUC: {macro_auc:.4f}")

        # Suite 5: Corruption Robustness Benchmark (14 conditions)
        print("  -> Evaluating 14 Corruption Stress Conditions...")
        clean_auc = m_in["roc_auc"]
        auc_drops = []

        # Subsample 500 images for fast, representative corruption testing
        rng = np.random.RandomState(42)
        sub_indices = rng.choice(len(indomain_samples), size=min(500, len(indomain_samples)), replace=False)
        corr_samples = [indomain_samples[idx] for idx in sub_indices]

        # Baseline clean on subsample
        ds_clean_sub = Phase8EvalDataset(corr_samples)
        loader_clean_sub = DataLoader(ds_clean_sub, batch_size=batch_size, shuffle=False)
        y_t_c, y_p_c, _, _ = run_model_inference(model, loader_clean_sub, device)
        clean_sub_auc = float(roc_auc_score(y_t_c, y_p_c))

        for c_name, c_fn in corruptions.items():
            ds_corr = Phase8EvalDataset(corr_samples, corruption_fn=c_fn)
            loader_corr = DataLoader(ds_corr, batch_size=batch_size, shuffle=False)
            y_t_corr, y_p_corr, _, _ = run_model_inference(model, loader_corr, device)
            m_corr = evaluate_predictions(y_t_corr, y_p_corr)
            corr_auc = m_corr["roc_auc"]
            auc_drop = max(0.0, clean_sub_auc - corr_auc)
            auc_drops.append(auc_drop)

            corruption_rows.append({
                "model_id": m_id,
                "model_name": m_name,
                "corruption": c_name,
                "clean_auc": round(clean_sub_auc, 4),
                "corrupted_auc": round(corr_auc, 4),
                "auc_drop": round(auc_drop, 4),
                "accuracy": m_corr["accuracy"],
                "f1": m_corr["f1"]
            })

        mean_auc_drop = float(np.mean(auc_drops))
        rob_score = max(0.0, 1.0 - mean_auc_drop)
        results[m_id]["robustness"] = {
            "mean_auc_drop": round(mean_auc_drop, 4),
            "composite_robustness_score": round(rob_score, 4)
        }
        print(f"  -> Robustness: Mean AUC Drop={mean_auc_drop:.4f} | Robustness Score={rob_score:.4f}")

        # Summary Row for comparison CSV
        comparison_rows.append({
            "model_id": m_id,
            "model_name": m_name,
            "in_domain_auc": m_in["roc_auc"],
            "in_domain_acc": m_in["accuracy"],
            "in_domain_f1": m_in["f1"],
            "logo_inpainting_auc": m_logo["roc_auc"],
            "logo_inpainting_acc": m_logo["accuracy"],
            "seen_family_eval_auc": m_eval["roc_auc"],
            "exploratory_macro_auc": round(macro_auc, 4),
            "sd15_auc": results[m_id]["exploratory_per_generator"]["StableDiffusion_v1.5"]["roc_auc"],
            "insightface_auc": results[m_id]["exploratory_per_generator"]["InsightFace_Swap"]["roc_auc"],
            "sd_inpaint_auc": results[m_id]["exploratory_per_generator"]["StableDiffusion_Inpainting"]["roc_auc"],
            "mean_auc_drop": round(mean_auc_drop, 4),
            "composite_robustness_score": round(rob_score, 4)
        })

    # Save deliverables
    rep_dir = Path("reports")
    rep_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = rep_dir / "phase8_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved metrics JSON to: {metrics_json_path.resolve()}")

    comparison_df = pd.DataFrame(comparison_rows)
    comp_csv_path = rep_dir / "phase8_model_comparison.csv"
    comparison_df.to_csv(comp_csv_path, index=False)
    print(f"[+] Saved model comparison table to: {comp_csv_path.resolve()}")

    corr_df = pd.DataFrame(corruption_rows)
    corr_csv_path = rep_dir / "phase8_corruption_analysis.csv"
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"[+] Saved corruption analysis to: {corr_csv_path.resolve()}")

    print("\n" + "=" * 80)
    print("                    PHASE 8 MODEL COMPARISON TABLE                       ")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")
    return results

if __name__ == "__main__":
    run_phase8_benchmarks()

