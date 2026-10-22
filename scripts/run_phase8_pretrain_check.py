import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import torch

def pretrain_check():
    print("=" * 80)
    print("              PHASE 8: PRE-TRAINING VERIFICATION AUDIT              ")
    print("=" * 80)

    # 1. Manifest and Splits
    manifest_path = Path("data/manifests/phase7_multigen_manifest.csv")
    splits_dir = Path("data/splits_multigen")
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"
    assert splits_dir.exists(), f"Splits dir missing: {splits_dir}"

    splits = ["train", "val", "test_indomain", "test_unseen_logo", "test_unseen_eval"]
    dfs = {}
    for s in splits:
        p = splits_dir / f"{s}.csv"
        assert p.exists(), f"Split file missing: {p}"
        dfs[s] = pd.read_csv(p)
        n_real = (dfs[s]["label"].str.upper() == "REAL").sum()
        n_fake = (dfs[s]["label"].str.upper() == "FAKE").sum()
        gens = dfs[s]["generator"].dropna().unique().tolist()
        print(f"[*] Split '{s:16s}': Total={len(dfs[s]):5d} | Real={n_real:5d} | Fake={n_fake:5d} | Generators={gens}")

    # 2. Config verification
    config_path = Path("configs/config_multigen.yaml")
    assert config_path.exists(), f"Config file missing: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n[*] Configuration Parameters in configs/config_multigen.yaml:")
    print(f"  - data.splits_dir:            {cfg['data']['splits_dir']}")
    print(f"  - model.fusion_type:          {cfg['model']['fusion_type']}")
    print(f"  - model.dropout:              {cfg['model']['dropout']}")
    print(f"  - model.spectral_dropout:     {cfg['model']['spectral_dropout']}")
    print(f"  - augmentation.mode:          {cfg['augmentation']['mode']}")
    print(f"  - training.model_name:        {cfg['training']['model_name']}")
    print(f"  - training.epochs:            {cfg['training']['epochs']}")
    print(f"  - training.batch_size:        {cfg['training']['batch_size']}")
    print(f"  - training.learning_rate:     {cfg['training']['learning_rate']}")
    print(f"  - training.target_checkpoint: {cfg['training']['target_checkpoint']}")
    print(f"  - training.checkpoint_metric: {cfg['training']['checkpoint_selection_metric']}")

    # 3. Quarantined LOGO Check
    train_gens = dfs["train"]["generator"].dropna().unique().tolist()
    val_gens = dfs["val"]["generator"].dropna().unique().tolist()
    logo_gen = "StableDiffusion_Inpainting"

    assert logo_gen not in train_gens, f"[LEAKAGE] {logo_gen} found in train!"
    assert logo_gen not in val_gens, f"[LEAKAGE] {logo_gen} found in val!"
    print(f"\n[+] Quarantine Audit: '{logo_gen}' has ZERO samples in train and val splits. (PASSED)")

    # 4. Check existing baseline checkpoints
    print("\n[*] Baseline Checkpoints Integrity Check:")
    ckpts = [
        "checkpoints/lxfd_attention_standard_best.pt",
        "checkpoints/lxfd_attention_robustness_v2_best.pt",
        "checkpoints/lxfd_attention_standard_spectral_dropout_best.pt",
        "checkpoints/efficientnet_best.pt",
        "checkpoints/dense_cnn_best.pt",
        "checkpoints/simple_cnn_best.pt"
    ]
    for c in ckpts:
        p = Path(c)
        status = f"EXISTS ({p.stat().st_size:,} bytes)" if p.exists() else "MISSING"
        print(f"  - {c:62s} : {status}")

    target_ckpt = Path("checkpoints/lxfd_multigen_diverse_best.pt")
    print(f"\n[*] Target Output Checkpoint: {target_ckpt} (Exists: {target_ckpt.exists()})")

    # 5. Check GPU
    print("\n[*] Compute Environment:")
    cuda_avail = torch.cuda.is_available()
    print(f"  - CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"  - GPU Name:       {torch.cuda.get_device_name(0)}")
        print(f"  - PyTorch AMP:    Supported (torch.amp)")

    print("=" * 80)
    print("                     ALL PRE-TRAINING CHECKS PASSED                     ")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    pretrain_check()
