import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def create_leak_free_splits(
    audit_csv_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[SPLIT] Loading metadata from {audit_csv_path}...")
    df = pd.read_csv(audit_csv_path)
    
    # Filter out corrupted
    if 'corrupted' in df.columns:
        valid_df = df[~df['corrupted']].copy()
    else:
        valid_df = df.copy()
        
    print(f"[SPLIT] Total valid records: {len(valid_df)}")
    
    # Build connected components for ZERO LEAKAGE
    # Group by augmentation_group, original_image_id, hash (MD5), phash (if present), identity (if present)
    uf = UnionFind()
    
    for idx, row in valid_df.iterrows():
        img_id = f"img_{idx}"
        uf.find(img_id)
        
        # Link augmentation group
        aug_grp = str(row.get('augmentation_group', row.get('original_image_id', idx)))
        if pd.notna(aug_grp) and aug_grp != '':
            uf.union(img_id, f"aug_{aug_grp}")
            
        # Link MD5 / exact hash
        md5_val = str(row.get('hash', row.get('md5', '')))
        if pd.notna(md5_val) and md5_val != '':
            uf.union(img_id, f"md5_{md5_val}")
            
        # Link dHash / phash
        phash_val = str(row.get('phash', row.get('dhash', '')))
        if pd.notna(phash_val) and phash_val != '':
            uf.union(img_id, f"phash_{phash_val}")
            
        # Link identity if present and not unknown
        identity_val = str(row.get('identity', 'unknown'))
        if identity_val != 'unknown' and identity_val != '':
            uf.union(img_id, f"id_{identity_val}")
            
    # Assign cluster ID
    valid_df['cluster_id'] = [uf.find(f"img_{idx}") for idx in valid_df.index]
    
    # Stratified split at cluster level
    cluster_df = valid_df.groupby('cluster_id').agg({
        'label': lambda s: s.mode()[0] if not s.empty else s.iloc[0],
        'path': 'count'
    }).reset_index().rename(columns={'path': 'cluster_size'})
    
    print(f"[SPLIT] Grouped into {len(cluster_df)} isolated macro-clusters.")
    
    temp_ratio = val_ratio + test_ratio
    val_within_temp = val_ratio / temp_ratio
    
    train_clusters, temp_clusters = train_test_split(
        cluster_df,
        test_size=temp_ratio,
        stratify=cluster_df['label'],
        random_state=seed
    )
    
    val_clusters, test_clusters = train_test_split(
        temp_clusters,
        test_size=(1.0 - val_within_temp),
        stratify=temp_clusters['label'],
        random_state=seed
    )
    
    train_set = set(train_clusters['cluster_id'])
    val_set = set(val_clusters['cluster_id'])
    test_set = set(test_clusters['cluster_id'])
    
    def assign_split(cid):
        if cid in train_set:
            return 'train'
        elif cid in val_set:
            return 'val'
        elif cid in test_set:
            return 'test'
        return 'unknown'
        
    valid_df['split'] = valid_df['cluster_id'].apply(assign_split)
    
    # EXPLICIT LEAKAGE VERIFICATION
    train_df = valid_df[valid_df['split'] == 'train'].copy()
    val_df = valid_df[valid_df['split'] == 'val'].copy()
    test_df = valid_df[valid_df['split'] == 'test'].copy()
    
    # 1. Augmentation group overlap
    train_aug = set(train_df['augmentation_group'])
    val_aug = set(val_df['augmentation_group'])
    test_aug = set(test_df['augmentation_group'])
    
    leak_aug_train_val = train_aug.intersection(val_aug)
    leak_aug_train_test = train_aug.intersection(test_aug)
    leak_aug_val_test = val_aug.intersection(test_aug)
    
    # 2. Hash overlap
    train_hash = set(train_df['hash'])
    val_hash = set(val_df['hash'])
    test_hash = set(test_df['hash'])
    
    leak_hash_train_val = train_hash.intersection(val_hash)
    leak_hash_train_test = train_hash.intersection(test_hash)
    leak_hash_val_test = val_hash.intersection(test_hash)
    
    has_leakage = bool(
        leak_aug_train_val or leak_aug_train_test or leak_aug_val_test or
        leak_hash_train_val or leak_hash_train_test or leak_hash_val_test
    )
    
    leakage_report = {
        "leakage_detected": has_leakage,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_class_counts": train_df['label'].value_counts().to_dict(),
        "val_class_counts": val_df['label'].value_counts().to_dict(),
        "test_class_counts": test_df['label'].value_counts().to_dict(),
        "augmentation_group_leakage": {
            "train_val_overlap": len(leak_aug_train_val),
            "train_test_overlap": len(leak_aug_train_test),
            "val_test_overlap": len(leak_aug_val_test)
        },
        "hash_leakage": {
            "train_val_overlap": len(leak_hash_train_val),
            "train_test_overlap": len(leak_hash_train_test),
            "val_test_overlap": len(leak_hash_val_test)
        }
    }
    
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "leakage_report.json", "w") as f:
        json.dump(leakage_report, f, indent=2)
        
    print(f"[SPLIT] Leakage verification completed. Leakage detected: {has_leakage}")
    
    if has_leakage:
        print("[ERROR] DATA LEAKAGE DETECTED IN SPLITS!")
        raise AssertionError("Data leakage detected between splits!")
        
    # Save split CSVs
    train_path = out_path / "train.csv"
    val_path = out_path / "val.csv"
    test_path = out_path / "test.csv"
    all_splits_path = out_path / "all_splits.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    valid_df.to_csv(all_splits_path, index=False)
    
    print(f"[SPLIT] Leakage-free splits saved successfully to {out_path}:")
    print(f"  - Train: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
    print(f"  - Val:   {len(val_df)} ({val_df['label'].value_counts().to_dict()})")
    print(f"  - Test:  {len(test_df)} ({test_df['label'].value_counts().to_dict()})")
    
    return leakage_report

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
        
    audit_csv = Path(config['data']['reports_audit_dir']) / "dataset_audit.csv"
    if not audit_csv.exists():
        audit_csv = Path("data/metadata.csv")
        
    output_dir = "data/splits_v2"
    
    create_leak_free_splits(
        str(audit_csv),
        output_dir,
        train_ratio=config['data']['split_ratios']['train'],
        val_ratio=config['data']['split_ratios']['val'],
        test_ratio=config['data']['split_ratios']['test'],
        seed=config['training']['seed']
    )

if __name__ == "__main__":
    main()
