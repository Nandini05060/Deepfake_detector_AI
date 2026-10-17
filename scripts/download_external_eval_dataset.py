import os
import io
import time
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import fsspec
import zipfile

def download_dataset(num_per_class: int = 200, output_dir: str = "data/external", unseen_dir: str = "data/unseen_generator_test"):
    """
    Downloads curated, balanced external evaluation subsets directly from the official
    OpenRL/DeepFakeFace repository (arXiv:2309.02218, OpenRAIL/Apache-2.0).
    
    Generators:
      - Real: IMDB-WIKI real celebrity portraits (disjoint from FFHQ)
      - Fake 1: StableDiffusion_v1.5 (Latent Text-to-Image Diffusion)
      - Fake 2: InsightFace_Swap (High-resolution face swap manipulation)
      - Fake 3: StableDiffusion_Inpainting (Diffusion inpainting manipulation)
      
    Extracts without downloading full gigabyte archives by utilizing HTTP 206 Partial Content range requests.
    """
    sources = [
        {
            "zip_file": "wiki.zip",
            "generator": "None",
            "dataset": "IMDB-WIKI",
            "label": "REAL",
            "dir_name": "real",
            "subfolder": "IMDB_WIKI_real"
        },
        {
            "zip_file": "text2img.zip",
            "generator": "StableDiffusion_v1.5",
            "dataset": "DeepFakeFace",
            "label": "FAKE",
            "dir_name": "StableDiffusion_v1.5",
            "subfolder": "StableDiffusion_v1.5"
        },
        {
            "zip_file": "insight.zip",
            "generator": "InsightFace_Swap",
            "dataset": "DeepFakeFace",
            "label": "FAKE",
            "dir_name": "InsightFace_Swap",
            "subfolder": "InsightFace_Swap"
        },
        {
            "zip_file": "inpainting.zip",
            "generator": "StableDiffusion_Inpainting",
            "dataset": "DeepFakeFace",
            "label": "FAKE",
            "dir_name": "StableDiffusion_Inpainting",
            "subfolder": "StableDiffusion_Inpainting"
        }
    ]

    base_url = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main"
    fs = fsspec.filesystem("http")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    unseen_path = Path(unseen_dir)
    unseen_path.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    print(f"[DOWNLOAD] Starting external evaluation dataset curation ({num_per_class} images per category)...")
    t0 = time.time()

    for src in sources:
        zip_url = f"{base_url}/{src['zip_file']}"
        target_dir = out_path / src["subfolder"]
        target_dir.mkdir(parents=True, exist_ok=True)
        unseen_target = unseen_path / src["dir_name"]
        unseen_target.mkdir(parents=True, exist_ok=True)

        print(f"[*] Accessing remote archive: {src['zip_file']} via HTTP Range stream...")
        try:
            with fs.open(zip_url, "rb") as f:
                with zipfile.ZipFile(f) as zf:
                    all_files = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    # Sort deterministically
                    all_files.sort()
                    
                    saved_count = 0
                    for fname in all_files:
                        if saved_count >= num_per_class:
                            break
                        
                        try:
                            data = zf.read(fname)
                            # Verify image can be decoded and is valid
                            img = Image.open(io.BytesIO(data))
                            img.verify()
                            
                            # Re-open for actual saving (verify mutates internal state)
                            img = Image.open(io.BytesIO(data)).convert("RGB")
                            
                            out_fname = f"{src['subfolder']}_{saved_count:04d}.jpg"
                            local_dest = target_dir / out_fname
                            img.save(local_dest, "JPEG", quality=95)

                            # Also copy or save to unseen_dir for directory-mode compatibility
                            unseen_dest = unseen_target / out_fname
                            img.save(unseen_dest, "JPEG", quality=95)

                            manifest_rows.append({
                                "path": str(local_dest.resolve()),
                                "unseen_path": str(unseen_dest.resolve()),
                                "filename": out_fname,
                                "label": src["label"],
                                "generator": src["generator"],
                                "dataset": src["dataset"],
                                "source_archive": src["zip_file"],
                                "original_path": fname,
                                "width": img.width,
                                "height": img.height
                            })
                            saved_count += 1
                        except Exception as e:
                            # Skip corrupted/unreadable sample
                            continue

            print(f"[+] Successfully extracted {saved_count} valid images for {src['subfolder']}")
        except Exception as e:
            print(f"[ERROR] Failed to access {zip_url}: {e}")

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_csv = out_path / "manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    
    # Also save manifest in unseen_generator_test
    df_unseen = df_manifest.copy()
    df_unseen["path"] = df_unseen["unseen_path"]
    df_unseen.to_csv(unseen_path / "manifest.csv", index=False)

    elapsed = time.time() - t0
    print("\n=======================================================")
    print("      EXTERNAL DATASET DOWNLOAD & CURATION COMPLETE    ")
    print("=======================================================")
    print(f"Total Valid Images:  {len(df_manifest)}")
    print(f"Real Images:         {(df_manifest['label'] == 'REAL').sum()}")
    print(f"Fake Images:         {(df_manifest['label'] == 'FAKE').sum()}")
    print(f"Generators:          {df_manifest['generator'].unique().tolist()}")
    print(f"Manifest File:       {manifest_csv.resolve()}")
    print(f"Unseen Directory:    {unseen_path.resolve()}")
    print(f"Elapsed Time:        {elapsed:.1f}s")
    print("=======================================================\n")
    return manifest_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Verified External Evaluation Dataset")
    parser.add_argument("--num-per-class", type=int, default=200, help="Number of images per category (default: 200)")
    parser.add_argument("--output-dir", type=str, default="data/external")
    parser.add_argument("--unseen-dir", type=str, default="data/unseen_generator_test")
    args = parser.parse_args()

    download_dataset(num_per_class=args.num_per_class, output_dir=args.output_dir, unseen_dir=args.unseen_dir)
