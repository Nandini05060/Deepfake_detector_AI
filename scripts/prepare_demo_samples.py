"""
Script: scripts/prepare_demo_samples.py
Prepares verified demonstration samples in samples/ for the Faculty Presentation Mode:
1. real_authentic_sample.jpg (FFHQ Studio Genuine)
2. stylegan2_sample.jpg (StyleGAN2 Progressive GAN)
3. sd15_diffusion_sample.jpg (Stable Diffusion v1.5 Latent Diffusion)
4. gemini_imagen3_sample.png (Google Gemini Imagen 3)
5. gemini_failure_diagnostic.png (Diagnostic failure image)
6. flux1_unseen_sample.jpg (FLUX.1 Pro Modern DiT)
7. vintage_imdb_sample.jpg (IMDB-WIKI Challenging Vintage Genuine)
"""

import shutil
from pathlib import Path
import pandas as pd
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
manifest_path = repo_root / "experiments/phase17_multigenerator_retraining/data_manifest.csv"
samples_dir = repo_root / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(manifest_path)

# 1. FFHQ Studio Real
ffhq_rows = df[(df["label"] == "REAL") & (df["generator_family"] == "FFHQ_Real")]
if len(ffhq_rows) > 0:
    src = repo_root / ffhq_rows.iloc[0]["relative_path"]
    dst = samples_dir / "real_authentic_sample.jpg"
    Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)
    print(f"Saved {dst.name} from {src}")

# 2. StyleGAN2
sg2_rows = df[(df["label"] == "FAKE") & (df["generator_family"] == "StyleGAN2")]
if len(sg2_rows) > 0:
    src = repo_root / sg2_rows.iloc[0]["relative_path"]
    dst = samples_dir / "stylegan2_sample.jpg"
    Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)
    print(f"Saved {dst.name} from {src}")

# 3. Stable Diffusion v1.5
sd_rows = df[(df["label"] == "FAKE") & (df["generator_family"] == "StableDiffusion_v1.5")]
if len(sd_rows) > 0:
    src = repo_root / sd_rows.iloc[0]["relative_path"]
    dst = samples_dir / "sd15_diffusion_sample.jpg"
    Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)
    print(f"Saved {dst.name} from {src}")

# 4. Diagnostic Gemini Image
diag_src = repo_root / "data/diagnostics/gemini_failure.png"
if diag_src.exists():
    dst = samples_dir / "gemini_failure_diagnostic.png"
    shutil.copy2(diag_src, dst)
    print(f"Saved {dst.name} from {diag_src}")

# 5. FLUX.1 Pro Unseen Generator
flux_rows = df[(df["label"] == "FAKE") & (df["generator_family"] == "FLUX1_Pro")]
if len(flux_rows) > 0:
    src = repo_root / flux_rows.iloc[0]["relative_path"]
    dst = samples_dir / "flux1_unseen_sample.jpg"
    Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)
    print(f"Saved {dst.name} from {src}")

# 6. Vintage IMDB-WIKI Genuine
imdb_rows = df[(df["label"] == "REAL") & (df["generator_family"] == "IMDB_WIKI")]
if len(imdb_rows) > 0:
    src = repo_root / imdb_rows.iloc[0]["relative_path"]
    dst = samples_dir / "vintage_imdb_sample.jpg"
    Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)
    print(f"Saved {dst.name} from {src}")

print("All sample demonstration images ready in samples/.")
