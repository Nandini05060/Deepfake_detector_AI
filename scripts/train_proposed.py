import sys
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.lxfd_model import LXDFDModel
from src.training.trainer import Trainer
from src.utils.seed import seed_everything
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Train Proposed LX-DFD Model and Ablation Variants")
    parser.add_argument("--fusion", type=str, default="attention", 
                        choices=["spatial_only", "frequency_only", "concat", "weighted", "attention"])
    parser.add_argument("--aug-mode", type=str, default="standard", choices=["standard", "robustness", "robustness_v2"])
    parser.add_argument("--spectral-dropout", action="store_true", help="Enable 2D DCT Spectral Dropout in frequency branch")
    parser.add_argument("--spectral-dropout-p", type=float, default=0.2, help="Probability for spectral dropout")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint file to resume from")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    seed_everything(config['training']['seed'])
    device = get_device()

    splits_dir = Path(config['data']['splits_dir'])
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        print("[ERROR] Split files not found in configs/config.yaml splits_dir!")
        sys.exit(1)

    train_dataset = DeepfakeDataset(str(train_csv), split='train', aug_mode=args.aug_mode)
    val_dataset = DeepfakeDataset(str(val_csv), split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LXDFDModel(
        embed_dim=config['model']['spatial_embed_dim'],
        fusion_type=args.fusion,
        pretrained=config['model']['spatial_pretrained'],
        dropout=config['model']['dropout'],
        spectral_dropout=args.spectral_dropout,
        spectral_dropout_p=args.spectral_dropout_p
    )

    suffix = ""
    if args.spectral_dropout:
        suffix = "_spectral_dropout"
    model_name = f"lxfd_{args.fusion}_{args.aug_mode}{suffix}"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['early_stopping_patience'],
        checkpoint_dir="checkpoints",
        model_name=model_name,
        use_amp=config['training']['use_amp'],
        resume=args.resume,
        resume_path=args.resume_path
    )

    results = trainer.train()
    print(f"[SUCCESS] Proposed model variant '{model_name}' trained! Best Val ROC-AUC: {results['best_val_auc']:.4f}")

if __name__ == "__main__":
    main()
