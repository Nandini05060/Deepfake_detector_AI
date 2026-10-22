import unittest
from pathlib import Path
import pandas as pd

class TestPhase7Manifest(unittest.TestCase):
    def setUp(self):
        self.manifest_path = Path("data/manifests/phase7_multigen_manifest.csv")
        self.splits_dir = Path("data/splits_multigen")
        self.required_columns = [
            "path", "label", "source_dataset", "generator",
            "identity_id", "original_group_id", "split",
            "face_detection_status", "preprocessing_version"
        ]

    def test_manifest_existence_and_columns(self):
        self.assertTrue(self.manifest_path.exists(), "Unified manifest CSV must exist")
        df = pd.read_csv(self.manifest_path)
        for col in self.required_columns:
            self.assertIn(col, df.columns, f"Required column '{col}' missing from manifest")

        self.assertGreater(len(df), 10000, "Manifest should contain >10,000 images")
        self.assertEqual(df["preprocessing_version"].iloc[0], "phase7_v1")

    def test_split_files_existence(self):
        expected_splits = ["train", "val", "test_indomain", "test_unseen_logo", "test_unseen_eval"]
        for sp in expected_splits:
            sp_path = self.splits_dir / f"{sp}.csv"
            self.assertTrue(sp_path.exists(), f"Split file '{sp}.csv' must exist")
            df_sp = pd.read_csv(sp_path)
            self.assertGreater(len(df_sp), 0, f"Split '{sp}.csv' must not be empty")

    def test_zero_logo_leakage(self):
        train_df = pd.read_csv(self.splits_dir / "train.csv")
        val_df = pd.read_csv(self.splits_dir / "val.csv")
        logo_df = pd.read_csv(self.splits_dir / "test_unseen_logo.csv")

        # Quarantined generator: StableDiffusion_Inpainting
        logo_gen = "StableDiffusion_Inpainting"

        train_gens = train_df["generator"].dropna().unique().tolist()
        val_gens = val_df["generator"].dropna().unique().tolist()

        self.assertNotIn(logo_gen, train_gens, f"LOGO generator '{logo_gen}' leaked into train split!")
        self.assertNotIn(logo_gen, val_gens, f"LOGO generator '{logo_gen}' leaked into val split!")

        # Verify logo generator is present in test_unseen_logo
        logo_test_gens = logo_df["generator"].dropna().unique().tolist()
        self.assertIn(logo_gen, logo_test_gens, f"LOGO generator '{logo_gen}' must be present in test_unseen_logo!")

    def test_disjoint_partitions(self):
        train_df = pd.read_csv(self.splits_dir / "train.csv")
        val_df = pd.read_csv(self.splits_dir / "val.csv")
        test_indomain_df = pd.read_csv(self.splits_dir / "test_indomain.csv")
        test_logo_df = pd.read_csv(self.splits_dir / "test_unseen_logo.csv")

        train_paths = set(train_df["path"].str.lower())
        val_paths = set(val_df["path"].str.lower())
        test_in_paths = set(test_indomain_df["path"].str.lower())
        test_logo_paths = set(test_logo_df["path"].str.lower())

        self.assertEqual(len(train_paths.intersection(val_paths)), 0, "Train and Val paths must be disjoint!")
        self.assertEqual(len(train_paths.intersection(test_in_paths)), 0, "Train and Test In-domain paths must be disjoint!")
        self.assertEqual(len(train_paths.intersection(test_logo_paths)), 0, "Train and Test LOGO paths must be disjoint!")
        self.assertEqual(len(val_paths.intersection(test_logo_paths)), 0, "Val and Test LOGO paths must be disjoint!")

    def test_class_presence_in_splits(self):
        for sp in ["train", "val", "test_indomain", "test_unseen_logo"]:
            df_sp = pd.read_csv(self.splits_dir / f"{sp}.csv")
            labels = df_sp["label"].str.upper().unique().tolist()
            self.assertIn("REAL", labels, f"Split '{sp}' must contain REAL images")
            self.assertIn("FAKE", labels, f"Split '{sp}' must contain FAKE images")

if __name__ == "__main__":
    unittest.main()
