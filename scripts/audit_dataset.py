import os
import json
import hashlib
import csv
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import yaml
from collections import Counter, defaultdict

def compute_md5(file_path: Path, block_size=65536) -> str:
    """Computes exact MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(image: Image.Image, hash_size=8) -> str:
    """
    Computes difference hash (perceptual hash) of an image.
    Difference hash works on gradients between adjacent pixels.
    """
    try:
        gray = ImageOps.grayscale(image)
        resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.int32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:].zfill(hash_size * hash_size // 4)
    except Exception:
        return ""

def extract_augmentation_group(filename: str) -> tuple:
    """
    Extracts base image ID and augmentation group from filename.
    Examples:
      'real_378_aug_1.jpg' -> ('real_378', 'real_378')
      'fake_273_aug_2.jpg' -> ('fake_273', 'fake_273')
      '001DDU0NI4.jpg'     -> ('001DDU0NI4', '001DDU0NI4')
    """
    stem = Path(filename).stem
    # Match patterns like _aug_1 or _aug_02 or _aug1
    match = re.search(r'^(.*?)(?:_aug(?:_|\d+).*)$', stem, re.IGNORECASE)
    if match:
        base_id = match.group(1)
        return base_id, base_id
    return stem, stem

def audit_dataset(dataset_dir: str, output_dir: str):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        fallback_path = Path("Final Dataset")
        if fallback_path.exists():
            dataset_path = fallback_path
            print(f"[AUDIT] Primary path not found. Falling back to local workspace path: {dataset_path.resolve()}")
        else:
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist!")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[AUDIT] Starting dataset audit for: {dataset_path.resolve()}")
    
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"[AUDIT] Found subdirectories: {[d.name for d in subdirs]}")
    
    class_mapping = {}
    for d in subdirs:
        lname = d.name.lower()
        if 'real' in lname:
            class_mapping[d.name] = 'REAL'
        elif 'fake' in lname or 'deepfake' in lname or 'spoof' in lname:
            class_mapping[d.name] = 'FAKE'
        else:
            class_mapping[d.name] = d.name.upper()

    image_records = []
    md5_hash_map = defaultdict(list)
    dhash_map = defaultdict(list)
    aug_group_map = defaultdict(list)
    
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    total_scanned = 0
    corrupted_files = []
    missing_files = []
    
    for folder, label in class_mapping.items():
        folder_path = dataset_path / folder
        print(f"[AUDIT] Scanning class '{label}' in {folder_path}...")
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext not in supported_exts:
                    continue
                
                total_scanned += 1
                rel_path = str(file_path.relative_to(dataset_path))
                orig_id, aug_group = extract_augmentation_group(file_path.name)
                
                try:
                    file_size = file_path.stat().st_size
                except Exception:
                    missing_files.append(str(file_path))
                    continue
                    
                md5_hash = compute_md5(file_path)
                md5_hash_map[md5_hash].append(rel_path)
                aug_group_map[aug_group].append(rel_path)
                
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    
                    with Image.open(file_path) as img:
                        width, height = img.size
                        dhash = compute_dhash(img)
                        if dhash:
                            dhash_map[dhash].append(rel_path)
                        
                        record = {
                            "path": str(file_path),
                            "file_path": str(file_path),
                            "relative_path": rel_path,
                            "label": label,
                            "source": "unknown",
                            "generator": "unknown",
                            "identity": "unknown",
                            "augmentation_group": aug_group,
                            "original_image_id": orig_id,
                            "width": width,
                            "height": height,
                            "file_format": ext[1:],
                            "file_size": file_size,
                            "hash": md5_hash,
                            "phash": dhash,
                            "md5": md5_hash,
                            "dhash": dhash,
                            "corrupted": False
                        }
                        image_records.append(record)
                except Exception as e:
                    corrupted_files.append({
                        "file_path": str(file_path),
                        "relative_path": rel_path,
                        "label": label,
                        "error": str(e)
                    })
                    image_records.append({
                        "path": str(file_path),
                        "file_path": str(file_path),
                        "relative_path": rel_path,
                        "label": label,
                        "source": "unknown",
                        "generator": "unknown",
                        "identity": "unknown",
                        "augmentation_group": aug_group,
                        "original_image_id": orig_id,
                        "width": None,
                        "height": None,
                        "file_format": ext[1:],
                        "file_size": file_size,
                        "hash": md5_hash,
                        "phash": "",
                        "md5": md5_hash,
                        "dhash": "",
                        "corrupted": True
                    })

    df = pd.DataFrame(image_records)
    
    exact_duplicates = {k: v for k, v in md5_hash_map.items() if len(v) > 1}
    num_exact_dup_files = sum(len(v) - 1 for v in exact_duplicates.values())
    
    perceptual_duplicates = {k: v for k, v in dhash_map.items() if len(v) > 1 and k != ""}
    num_perceptual_dup_files = sum(len(v) - 1 for v in perceptual_duplicates.values())
    
    aug_groups_multi = {k: v for k, v in aug_group_map.items() if len(v) > 1}
    
    label_counts = df['label'].value_counts().to_dict()
    valid_df = df[~df['corrupted']]
    
    resolution_counts = valid_df.groupby(['width', 'height']).size().reset_index(name='count').to_dict('records')
    format_counts = df['file_format'].value_counts().to_dict()
    
    audit_summary = {
        "dataset_root": str(dataset_path),
        "total_images": len(df),
        "total_valid_images": len(valid_df),
        "class_distribution": label_counts,
        "missing_files_count": len(missing_files),
        "corrupted_files_count": len(corrupted_files),
        "exact_duplicates_unique_hashes": len(exact_duplicates),
        "exact_duplicate_files_count": num_exact_dup_files,
        "perceptual_duplicates_unique_hashes": len(perceptual_duplicates),
        "perceptual_duplicate_files_count": num_perceptual_dup_files,
        "total_augmentation_groups": len(aug_group_map),
        "multi_image_augmentation_groups_count": len(aug_groups_multi),
        "missing_metadata_fields": ["source", "generator", "identity"],
        "file_formats": format_counts,
        "resolution_distribution": resolution_counts[:10]
    }
    
    # 1. Save JSON report
    json_path = out_path / "dataset_audit.json"
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[AUDIT] Saved JSON audit report to {json_path}")
    
    # 2. Save TXT report per prompt specification
    txt_path = out_path / "dataset_audit.txt"
    with open(txt_path, "w") as f:
        f.write("==================================================\n")
        f.write("            LX-DFD DATASET AUDIT REPORT           \n")
        f.write("==================================================\n\n")
        f.write(f"Dataset Directory: {dataset_path}\n")
        f.write(f"Total Scanned Images: {audit_summary['total_images']}\n")
        f.write(f"Total Valid Images:   {audit_summary['total_valid_images']}\n")
        f.write(f"Missing Files:        {audit_summary['missing_files_count']}\n")
        f.write(f"Corrupted Files:      {audit_summary['corrupted_files_count']}\n\n")
        f.write("Class Counts:\n")
        for k, v in label_counts.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\nFile Formats:\n")
        for k, v in format_counts.items():
            f.write(f"  - .{k}: {v}\n")
        f.write("\nDuplicates & Augmentation Groups:\n")
        f.write(f"  - Exact Duplicate Hashes (MD5): {audit_summary['exact_duplicates_unique_hashes']} (Files: {num_exact_dup_files})\n")
        f.write(f"  - Perceptual Hashes (dHash):    {audit_summary['perceptual_duplicates_unique_hashes']} (Files: {num_perceptual_dup_files})\n")
        f.write(f"  - Total Augmentation Groups:    {audit_summary['total_augmentation_groups']}\n")
        f.write(f"  - Multi-variant Aug Groups:     {audit_summary['multi_image_augmentation_groups_count']}\n\n")
        f.write("Metadata Field Status:\n")
        f.write("  - path: AVAILABLE\n")
        f.write("  - label: AVAILABLE\n")
        f.write("  - augmentation_group: INFERRED FROM FILENAME\n")
        f.write("  - original_image_id: INFERRED FROM FILENAME\n")
        f.write("  - width/height/format/size/hash/phash: COMPUTED\n")
        f.write("  - source: UNKNOWN (Not provided in dataset)\n")
        f.write("  - generator: UNKNOWN (Not provided in dataset)\n")
        f.write("  - identity: UNKNOWN (Not provided in dataset)\n")
    print(f"[AUDIT] Saved TXT audit report to {txt_path}")
    
    # 3. Save CSV report and data/metadata.csv
    csv_path = out_path / "dataset_audit.csv"
    df.to_csv(csv_path, index=False)
    
    meta_path = Path("data/metadata.csv")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_path, index=False)
    print(f"[AUDIT] Saved full metadata to {csv_path} and {meta_path}")

    return audit_summary

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    audit_dataset(config['data']['dataset_dir'], config['data']['reports_audit_dir'])

