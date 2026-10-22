import unittest
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DeepfakeDataset
from src.models.lxfd_model import LXDFDModel

class TestTrainingLoaderCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = Path("configs/config_multigen.yaml")
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def test_config_structure(self):
        self.assertIn("data", self.config)
        self.assertIn("training", self.config)
        self.assertIn("model", self.config)
        self.assertEqual(self.config["data"]["splits_dir"], "data/splits_multigen")
        self.assertEqual(self.config["training"]["model_name"], "lxfd_multigen_diverse")

    def test_multigen_dataset_loading(self):
        splits_dir = Path(self.config["data"]["splits_dir"])
        train_csv = splits_dir / "train.csv"
        val_csv = splits_dir / "val.csv"

        self.assertTrue(train_csv.exists(), "train.csv must exist in splits_multigen")
        self.assertTrue(val_csv.exists(), "val.csv must exist in splits_multigen")

        # Load first few samples
        train_ds = DeepfakeDataset(str(train_csv), split="train", aug_mode="standard")
        self.assertGreater(len(train_ds), 9000, "Multi-generator train split should have >9,000 samples")

        sample = train_ds[0]
        self.assertIn("image", sample)
        self.assertIn("label", sample)
        self.assertEqual(sample["image"].shape, (3, 224, 224))
        self.assertIsInstance(sample["label"], torch.Tensor)
        self.assertIn(sample["label"].item(), [0.0, 1.0])

    def test_dataloader_batch_generation(self):
        splits_dir = Path(self.config["data"]["splits_dir"])
        val_csv = splits_dir / "val.csv"
        val_ds = DeepfakeDataset(str(val_csv), split="val")
        loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape, (4, 3, 224, 224))
        self.assertEqual(batch["label"].shape, (4,))

    def test_model_forward_with_multigen_batch(self):
        splits_dir = Path(self.config["data"]["splits_dir"])
        val_csv = splits_dir / "val.csv"
        val_ds = DeepfakeDataset(str(val_csv), split="val")
        loader = DataLoader(val_ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        model = LXDFDModel(embed_dim=256, fusion_type="attention", pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(batch["image"])
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits)

        self.assertIn(logits.shape, [(2,), (2, 1)])
        self.assertIn(probs.shape, [(2,), (2, 1)])
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

if __name__ == "__main__":
    unittest.main()
