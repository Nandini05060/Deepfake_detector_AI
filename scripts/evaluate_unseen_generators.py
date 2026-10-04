import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.lxfd_model import LXDFDModel
from src.evaluation.metrics import compute_all_metrics
from src.utils.device import get_device

class UnseenGeneratorDataset(Dataset):
    def __init__(self, image_paths, labels, image_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor_img = self.transform(img)
        except Exception:
            tensor_img = torch.zeros((3, 224, 224), dtype=torch.float32)
        return {'image': tensor_img, 'label': torch.tensor(label, dtype=torch.float32), 'path': str(path)}

def evaluate_unseen_generators(checkpoint_path: str, unseen_dir: str = "data/unseen_generator_test", output_json: str = "reports/unseen_generator_results.json"):
    device = get_device()
    unseen_path = Path(unseen_dir)
    unseen_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure standard directory structure exists
    real_dir = unseen_path / "real"
    real_dir.mkdir(exist_ok=True)
    gen_a_dir = unseen_path / "generator_A"
    gen_a_dir.mkdir(exist_ok=True)
    gen_b_dir = unseen_path / "generator_B"
    gen_b_dir.mkdir(exist_ok=True)
    
    # Load model checkpoint
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        return None
        
    print(f"[UNSEEN] Loading model from {checkpoint_path}...")
    checkpoint = torch.load(ckpt_file, map_location=device)

    # Detect model configuration if present
    model = LXDFDModel(embed_dim=256, fusion_type="attention", pretrained=False).to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Discover generator subdirectories (excluding real)
    gen_dirs = [d for d in unseen_path.iterdir() if d.is_dir() and d.name.lower() != 'real']
    real_images = list(real_dir.glob("*.*"))

    if not gen_dirs or all(len(list(d.glob("*.*"))) == 0 for d in gen_dirs):
        print("\n==================================================")
        print("[UNSEEN BENCHMARK STATUS REPORT]")
        print(f"Directory: {unseen_path.resolve()}")
        print("Status: Template structure initialized. No external unseen generator images found.")
        print("To run unseen generator benchmark, place test images in:")
        print(f"  - Real images: {real_dir.resolve()}")
        print(f"  - Generator subfolders: e.g. {unseen_path.resolve() / 'StyleGAN3'}, {unseen_path.resolve() / 'FLUX'}")
        print("==================================================\n")
        
        report = {
            "checkpoint": str(checkpoint_path),
            "unseen_dir": str(unseen_path),
            "status": "No unseen generator images provided yet.",
            "generators_evaluated": [],
            "macro_average": {},
            "weighted_average": {}
        }
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        return report

    results = {}
    all_y_true = []
    all_y_prob = []

    for gen_dir in gen_dirs:
        fake_images = list(gen_dir.glob("*.*"))
        if not fake_images:
            continue
            
        # Pair with available real images for paired ROC-AUC calculation
        eval_paths = real_images + fake_images
        eval_labels = [0.0] * len(real_images) + [1.0] * len(fake_images)
        
        dataset = UnseenGeneratorDataset(eval_paths, eval_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        y_true_list = []
        y_prob_list = []

        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(device)
                lbls = batch['label'].cpu().numpy()
                out = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.sigmoid(logits).cpu().numpy()

                y_true_list.extend(lbls)
                y_prob_list.extend(probs)

        y_true = np.array(y_true_list)
        y_prob = np.array(y_prob_list)

        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
        metrics['generator_name'] = gen_dir.name
        metrics['num_images'] = len(eval_paths)
        metrics['num_fake'] = len(fake_images)
        metrics['num_real'] = len(real_images)

        results[gen_dir.name] = metrics
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    # Compute Macro and Weighted Averages
    if results:
        macro_acc = float(np.mean([m['accuracy'] for m in results.values()]))
        macro_auc = float(np.mean([m['roc_auc'] for m in results.values()]))
        macro_f1 = float(np.mean([m['f1'] for m in results.values()]))

        weights = [m['num_images'] for m in results.values()]
        weighted_acc = float(np.average([m['accuracy'] for m in results.values()], weights=weights))
        weighted_auc = float(np.average([m['roc_auc'] for m in results.values()], weights=weights))
        weighted_f1 = float(np.average([m['f1'] for m in results.values()], weights=weights))

        summary = {
            "checkpoint": str(checkpoint_path),
            "unseen_dir": str(unseen_path),
            "per_generator_metrics": results,
            "macro_average": {
                "accuracy": macro_acc,
                "roc_auc": macro_auc,
                "f1": macro_f1
            },
            "weighted_average": {
                "accuracy": weighted_acc,
                "roc_auc": weighted_auc,
                "f1": weighted_f1
            }
        }
    else:
        summary = {"checkpoint": str(checkpoint_path), "per_generator_metrics": {}}

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[UNSEEN] Results saved to {output_json}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model on Unseen AI Generators")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lxfd_attention_standard_best.pt")
    parser.add_argument("--unseen-dir", type=str, default="data/unseen_generator_test")
    parser.add_argument("--output-json", type=str, default="reports/unseen_generator_results.json")
    args = parser.parse_args()

    evaluate_unseen_generators(args.checkpoint, args.unseen_dir, args.output_json)

if __name__ == "__main__":
    main()
