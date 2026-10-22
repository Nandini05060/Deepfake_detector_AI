import os
import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.face_preprocessor import YuNetFacePreprocessor

def build_phase7_manifest():
    print("=" * 80)
    print("    PHASE 7: BUILDING LEAKAGE-RESISTANT MULTI-GENERATOR MANIFEST    ")
    print("=" * 80)

    # Output paths
    manifest_dir = Path("data/manifests")
    splits_dir = Path("data/splits_multigen")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # 1. Ingest In-Domain Splits (splits_v2: Kaggle StyleGAN2 / FFHQ)
    print("[*] Ingesting in-domain StyleGAN2 & FFHQ splits (splits_v2)...")
    splits_v2_files = {
        "train": Path("data/splits_v2/train.csv"),
        "val": Path("data/splits_v2/val.csv"),
        "test": Path("data/splits_v2/test.csv")
    }

    for split_name, s_path in splits_v2_files.items():
        if not s_path.exists():
            raise FileNotFoundError(f"Missing required split file: {s_path}")
        df_s = pd.read_csv(s_path)
        dest_split = "test_indomain" if split_name == "test" else split_name

        for _, r in df_s.iterrows():
            img_path = str(r["path"])
            label = str(r["label"]).upper()
            src_ds = "FFHQ" if label == "REAL" else "Kaggle_DFD"
            gen = "None" if label == "REAL" else "StyleGAN2"
            orig_grp = str(r.get("augmentation_group", r.get("cluster_id", "indomain_cluster")))
            ident = str(r.get("original_image_id", r.get("cluster_id", "indomain_ident")))

            rows.append({
                "path": img_path,
                "label": label,
                "source_dataset": src_ds,
                "generator": gen,
                "identity_id": ident,
                "original_group_id": orig_grp,
                "split": dest_split,
                "face_detection_status": "pre_aligned_face",
                "preprocessing_version": "phase7_v1"
            })

    # 2. Ingest External Multi-Generator Dataset (data/external/manifest.csv)
    print("[*] Ingesting external multi-generator dataset...")
    ext_manifest_path = Path("data/external/manifest.csv")
    if not ext_manifest_path.exists():
        raise FileNotFoundError(f"External manifest not found: {ext_manifest_path}")

    df_ext = pd.read_csv(ext_manifest_path)
    # Pre-select strictly quarantined Leave-One-Generator-Out (LOGO) generator
    # LOGO QUARANTINE: StableDiffusion_Inpainting will NEVER appear in train or val!
    logo_generator = "StableDiffusion_Inpainting"

    # Seed for deterministic splitting of external samples
    rng = np.random.RandomState(42)

    for gen in df_ext["generator"].unique():
        if pd.isna(gen) or str(gen).lower() in ["none", "nan"]:
            # Real IMDB-WIKI images (200 images)
            gen_df = df_ext[df_ext["label"] == "REAL"].copy().reset_index(drop=True)
            gen_name = "None"
            label_val = "REAL"
            src_ds = "IMDB-WIKI"

            # Allocate 70 train, 15 val, 15 test_unseen_eval, 100 test_unseen_logo
            n = len(gen_df)
            indices = np.arange(n)
            rng.shuffle(indices)

            train_idx = set(indices[:70])
            val_idx = set(indices[70:85])
            eval_idx = set(indices[85:100])
            logo_idx = set(indices[100:])

            for i, (_, r) in enumerate(gen_df.iterrows()):
                if i in train_idx:
                    sp = "train"
                elif i in val_idx:
                    sp = "val"
                elif i in eval_idx:
                    sp = "test_unseen_eval"
                else:
                    sp = "test_unseen_logo"

                rows.append({
                    "path": str(r["path"]),
                    "label": label_val,
                    "source_dataset": src_ds,
                    "generator": gen_name,
                    "identity_id": f"imdb_{Path(str(r['path'])).stem}",
                    "original_group_id": f"group_imdb_{i}",
                    "split": sp,
                    "face_detection_status": "yunet_candidate",
                    "preprocessing_version": "phase7_v1"
                })
        else:
            gen_df = df_ext[df_ext["generator"] == gen].copy().reset_index(drop=True)
            gen_name = str(gen)
            label_val = "FAKE"
            src_ds = str(gen_df["dataset"].iloc[0])

            if gen_name == logo_generator:
                # QUARANTINED LEAVE-ONE-GENERATOR-OUT TEST (100% held-out)
                for i, (_, r) in enumerate(gen_df.iterrows()):
                    rows.append({
                        "path": str(r["path"]),
                        "label": label_val,
                        "source_dataset": src_ds,
                        "generator": gen_name,
                        "identity_id": f"logo_{Path(str(r['path'])).stem}",
                        "original_group_id": f"group_logo_{gen_name}_{i}",
                        "split": "test_unseen_logo",
                        "face_detection_status": "yunet_candidate",
                        "preprocessing_version": "phase7_v1"
                    })
            else:
                # 140 train (70%) / 30 val (15%) / 30 test_unseen_eval (15%)
                n = len(gen_df)
                indices = np.arange(n)
                rng.shuffle(indices)

                train_idx = set(indices[:140])
                val_idx = set(indices[140:170])

                for i, (_, r) in enumerate(gen_df.iterrows()):
                    if i in train_idx:
                        sp = "train"
                    elif i in val_idx:
                        sp = "val"
                    else:
                        sp = "test_unseen_eval"

                    rows.append({
                        "path": str(r["path"]),
                        "label": label_val,
                        "source_dataset": src_ds,
                        "generator": gen_name,
                        "identity_id": f"ext_{Path(str(r['path'])).stem}",
                        "original_group_id": f"group_ext_{gen_name}_{i}",
                        "split": sp,
                        "face_detection_status": "yunet_candidate",
                        "preprocessing_version": "phase7_v1"
                    })

    df_manifest = pd.DataFrame(rows)
    # Deduplicate paths
    df_manifest = df_manifest.drop_duplicates(subset=["path"]).reset_index(drop=True)

    # Save complete unified manifest
    manifest_csv = manifest_dir / "phase7_multigen_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"\n[+] Saved unified manifest ({len(df_manifest)} images): {manifest_csv.resolve()}")

    # Save split files
    split_counts = {}
    for sp in ["train", "val", "test_indomain", "test_unseen_logo", "test_unseen_eval"]:
        df_sp = df_manifest[df_manifest["split"] == sp].copy()
        sp_csv = splits_dir / f"{sp}.csv"
        df_sp.to_csv(sp_csv, index=False)
        split_counts[sp] = len(df_sp)
        print(f"  - Split '{sp}': {len(df_sp)} images -> {sp_csv.resolve()}")

    print("\n" + "=" * 80)
    print("                     SPLIT DISTRIBUTION TABLE                   ")
    print("=" * 80)
    print(pd.crosstab(df_manifest["split"], [df_manifest["label"], df_manifest["generator"]], margins=True))

    # Audit Item: Verify LOGO generator leakage
    train_val_gens = df_manifest[df_manifest["split"].isin(["train", "val"])]["generator"].unique().tolist()
    print("\n[*] Audit Check — Generators present in Train/Val:")
    print("   ", train_val_gens)
    print(f"[*] Quarantined LOGO Generator: '{logo_generator}'")
    if logo_generator in train_val_gens:
        print("[CRITICAL ERROR] LOGO generator leaked into train/val!")
        sys.exit(1)
    else:
        print(f"[PASS] Zero leakage of '{logo_generator}' into train or validation splits!")

    print("=" * 80 + "\n")
    return df_manifest

if __name__ == "__main__":
    build_phase7_manifest()
