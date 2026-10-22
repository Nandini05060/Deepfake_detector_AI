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
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--fusion", type=str, default="attention", 
                        choices=["spatial_only", "frequency_only", "concat", "weighted", "attention"])
    parser.add_argument("--aug-mode", type=str, default=None, choices=["standard", "robustness", "robustness_v2"])
    parser.add_argument("--spectral-dropout", action="store_true", help="Enable 2D DCT Spectral Dropout in frequency branch")
    parser.add_argument("--spectral-dropout-p", type=float, default=0.2, help="Probability for spectral dropout")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None, help="Explicit model checkpoint name")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint file to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 verification epoch on 2 batches to validate training pipeline")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed_everything(config.get('training', {}).get('seed', 42))
    device = get_device()

    splits_dir = Path(config['data']['splits_dir'])
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        print(f"[ERROR] Split files not found in {splits_dir}!")
        sys.exit(1)

    aug_mode = args.aug_mode or config.get('augmentation', {}).get('mode', 'standard')
    batch_size = args.batch_size or config.get('training', {}).get('batch_size', 32)
    epochs = args.epochs or config.get('training', {}).get('epochs', 15)

    train_dataset = DeepfakeDataset(str(train_csv), split='train', aug_mode=aug_mode)
    val_dataset = DeepfakeDataset(str(val_csv), split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    is_spectral = args.spectral_dropout or config.get('model', {}).get('spectral_dropout', False)
    spectral_p = args.spectral_dropout_p if args.spectral_dropout else config.get('model', {}).get('spectral_dropout_p', 0.2)

    model = LXDFDModel(
        embed_dim=config['model'].get('spatial_embed_dim', 256),
        fusion_type=args.fusion or config['model'].get('fusion_type', 'attention'),
        pretrained=config['model'].get('spatial_pretrained', True),
        dropout=config['model'].get('dropout', 0.3),
        spectral_dropout=is_spectral,
        spectral_dropout_p=spectral_p
    )

    if args.model_name:
        model_name = args.model_name
    elif config.get('training', {}).get('model_name'):
        model_name = config['training']['model_name']
    else:
        suffix = "_spectral_dropout" if is_spectral else ""
        model_name = f"lxfd_{args.fusion}_{aug_mode}{suffix}"

    print(f"[TRAIN SETUP] Model: '{model_name}' | Config: '{config_path}' | Splits: '{splits_dir}'")
    print(f"[TRAIN SETUP] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Batch size: {batch_size}")

    if args.dry_run:
        print("\n" + "=" * 80)
        print("          EXECUTING TRAINING CONFIGURATION DRY-RUN VERIFICATION          ")
        print("=" * 80)
        model = model.to(device)
        model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 1. Forward and backward on 2 train batches
        print("[*] Testing train forward and backward passes...")
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            optimizer.zero_grad()
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            print(f"  - Train Batch {i+1}: Image Shape={tuple(imgs.shape)}, Loss={loss.item():.4f}")

        # 2. Evaluation on 2 val batches
        print("[*] Testing validation forward pass...")
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 2:
                    break
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)
                out = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.sigmoid(logits).cpu().numpy()
                print(f"  - Val Batch {i+1}: Image Shape={tuple(imgs.shape)}, Mean Prob={probs.mean():.4f}")

        print("\n[PASS] Training configuration dry run successfully verified! All tensors and gradients valid.")
        print("=" * 80 + "\n")
        return

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        patience=int(config['training']['early_stopping_patience']),
        checkpoint_dir=config['training'].get('checkpoint_dir', "checkpoints"),
        model_name=model_name,
        use_amp=config['training'].get('use_amp', True),
        resume=args.resume,
        resume_path=args.resume_path
    )

    results = trainer.train()
    print(f"[SUCCESS] Proposed model variant '{model_name}' trained! Best Val ROC-AUC: {results['best_val_auc']:.4f}")

if __name__ == "__main__":
    main()
