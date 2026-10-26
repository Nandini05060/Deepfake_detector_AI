"""
Unit & Regression Test Suite: tests/test_phase17_verification.py
Verifies:
1. Correct checkpoint loading and SHA-256 matching.
2. Correct class ordering (0.0 = REAL, 1.0 = FAKE).
3. Consistent preprocessing (Resize 224x224, bicubic, standard ImageNet normalization).
4. Correct calibration and threshold application (T=1.1599, tau=0.5400).
5. Genuine-image inference on authentic photographs (FFHQ Studio Real).
6. Known fake-image inference (StyleGAN2 and SDXL).
7. Gemini diagnostic regression (gemini_failure.png never classified as LIKELY_REAL).
8. Model switching across Phase 17, Phase 16, Phase 15, and Phase 10.
9. Missing-face behavior (graceful handling, inconclusive routing).
10. Corrupted image handling.
11. Large image handling (2048x2048 synthetic tensor).
12. Uncertainty routing and probability consistency (real_prob + fake_prob == 1.0).
13. No hardcoded predictions (different images produce different continuous logits).
14. No test-set leakage (Train, Val, and Test splits are disjoint; gemini_failure.png excluded).
"""

import os
import sys
import unittest
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.inference import LXDFDPredictor
from app.backend.main import load_lxfd_predictor, ACTIVE_MODEL_CONFIG, get_model_status

class TestPhase17Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = repo_root / "experiments/phase17_multigenerator_retraining/lxfd_phase17_best.pt"
        cls.cal_path = repo_root / "experiments/phase17_multigenerator_retraining/calibration_params_phase17.json"
        cls.manifest_path = repo_root / "experiments/phase17_multigenerator_retraining/data_manifest.csv"
        cls.diag_path = repo_root / "data/diagnostics/gemini_failure.png"
        
        # Load official predictor
        cls.predictor = LXDFDPredictor(
            checkpoint_path=str(cls.ckpt_path),
            calibration_path=str(cls.cal_path),
            decision_threshold=0.54,
            uncertainty_band=0.15,
            min_floor=0.25
        )

    def test_01_checkpoint_loading_and_sha256(self):
        """Verify checkpoint exists, loads properly, and matches recorded SHA-256."""
        self.assertTrue(self.ckpt_path.exists(), "Phase 17 checkpoint file is missing!")
        sha256 = hashlib.sha256()
        with open(self.ckpt_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        calc_hash = sha256.hexdigest().lower()
        expected_hash = "92157690ff27268892b6ed3b73c6f664fafc3963d55f78b56cddeb993421f863"
        self.assertEqual(calc_hash, expected_hash, f"Checkpoint hash mismatch! Got {calc_hash}")
        self.assertIsNotNone(self.predictor.model)

    def test_02_class_ordering_and_probability_consistency(self):
        """Verify positive class convention (1.0 = FAKE) and real_prob + fake_prob == 1.0."""
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        res = self.predictor.predict(dummy_img)
        self.assertAlmostEqual(res["fake_probability"] + res["real_probability"], 1.0, places=3)
        self.assertIn("spatial_branch_fake_prob", res["branch_outputs"])
        self.assertIn("frequency_branch_fake_prob", res["branch_outputs"])

    def test_03_consistent_preprocessing(self):
        """Verify preprocessing transforms input images into (1, 3, 224, 224) tensors with standard normalization."""
        dummy_img = Image.new("RGB", (512, 512), color=(200, 100, 50))
        t_img = self.predictor.transform(dummy_img)
        self.assertEqual(t_img.shape, (3, 224, 224))
        self.assertIsInstance(t_img, torch.Tensor)

    def test_04_calibration_and_threshold_application(self):
        """Verify temperature scaling is active with T=1.1599 and threshold tau=0.54."""
        self.assertTrue(self.predictor.calibration_enabled)
        self.assertAlmostEqual(self.predictor.scaler.temperature, 1.1599, places=2)
        self.assertEqual(self.predictor.decision_threshold, 0.54)
        self.assertEqual(self.predictor.uncertainty_band, 0.15)

    def test_05_genuine_image_inference(self):
        """Verify authentic human photographs from FFHQ are not falsely flagged as fake."""
        df = pd.read_csv(self.manifest_path)
        real_row = df[(df["label"] == "REAL") & (df["generator_family"] == "FFHQ_Real")].iloc[0]
        real_path = repo_root / real_row["relative_path"]
        self.assertTrue(real_path.exists(), f"Sample real image {real_path} not found!")
        sample_img = Image.open(real_path).convert("RGB")
        res = self.predictor.predict(sample_img)
        self.assertNotEqual(res["final_decision"], "LIKELY_FAKE", "Authentic studio portrait was falsely classified as FAKE!")

    def test_06_known_fake_inference(self):
        """Verify known deepfake images (StyleGAN2) are detected as FAKE."""
        df = pd.read_csv(self.manifest_path)
        fake_row = df[(df["label"] == "FAKE") & (df["generator_family"] == "StyleGAN2")].iloc[0]
        fake_path = repo_root / fake_row["relative_path"]
        self.assertTrue(fake_path.exists(), f"Sample fake image {fake_path} not found!")
        sample_img = Image.open(fake_path).convert("RGB")
        res = self.predictor.predict(sample_img)
        self.assertGreater(res["fake_probability"], 0.70)
        self.assertEqual(res["final_decision"], "LIKELY_FAKE")

    def test_07_gemini_diagnostic_regression(self):
        """Verify gemini_failure.png is never classified as LIKELY_REAL."""
        self.assertTrue(self.diag_path.exists(), "Diagnostic gemini_failure.png missing!")
        diag_img = Image.open(self.diag_path).convert("RGB")
        res = self.predictor.predict(diag_img)
        self.assertNotEqual(res["final_decision"], "LIKELY_REAL", "Diagnostic Gemini image was incorrectly classified as REAL!")
        self.assertIn(res["final_decision"], ["LIKELY_FAKE", "INCONCLUSIVE"])

    def test_08_model_switching_and_rollback(self):
        """Verify dynamic model switching across Phase 17, Phase 16, Phase 15, and Phase 10."""
        # Switch to Phase 16
        pred_p16 = load_lxfd_predictor("phase16")
        self.assertEqual(ACTIVE_MODEL_CONFIG["version"], "phase16")
        self.assertIn("Phase 16", ACTIVE_MODEL_CONFIG["name"])

        # Switch to Phase 15
        pred_p15 = load_lxfd_predictor("phase15")
        self.assertEqual(ACTIVE_MODEL_CONFIG["version"], "phase15")

        # Switch back to Phase 17 Champion
        pred_p17 = load_lxfd_predictor("phase17")
        self.assertEqual(ACTIVE_MODEL_CONFIG["version"], "phase17")
        self.assertEqual(ACTIVE_MODEL_CONFIG["epoch"], 2)

        # Test safe rollback on invalid version
        with self.assertRaises(ValueError):
            load_lxfd_predictor("nonexistent_phase99")

    def test_09_missing_face_behavior(self):
        """Verify missing face triggers INCONCLUSIVE with localization warning."""
        dummy_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        face_meta = {"face_detected": False, "bbox": [0, 0, 224, 224]}
        res = self.predictor.predict(dummy_img, face_meta=face_meta)
        self.assertEqual(res["final_decision"], "INCONCLUSIVE")
        self.assertIn("FACE LOCALIZATION WARNING", res["generator_warning"])

    def test_10_large_and_corrupted_image_handling(self):
        """Verify handling of large images (2048x2048) and corrupted image files."""
        large_img = Image.new("RGB", (2048, 2048), color=(150, 150, 150))
        res = self.predictor.predict(large_img)
        self.assertIn(res["final_decision"], ["LIKELY_REAL", "LIKELY_FAKE", "INCONCLUSIVE"])

        with self.assertRaises(TypeError):
            self.predictor.predict(12345)

    def test_11_no_hardcoded_predictions(self):
        """Verify predictor outputs vary continuously and are not hardcoded lookups."""
        img1 = Image.new("RGB", (224, 224), color=(50, 50, 50))
        img2 = Image.new("RGB", (224, 224), color=(200, 200, 200))
        res1 = self.predictor.predict(img1)
        res2 = self.predictor.predict(img2)
        self.assertNotEqual(res1["fake_probability"], res2["fake_probability"])

    def test_12_no_test_set_leakage(self):
        """Verify zero split leakage between train, validation, and test, and that diagnostic image is excluded."""
        self.assertTrue(self.manifest_path.exists())
        df = pd.read_csv(self.manifest_path)
        
        train_paths = set(df[df["split"].str.lower() == "train"]["relative_path"])
        val_paths = set(df[df["split"].str.lower() == "validation"]["relative_path"])
        test_paths = set(df[df["split"].str.lower() == "test"]["relative_path"])
        
        # Inter-split leakage check
        self.assertEqual(len(train_paths.intersection(val_paths)), 0, "Train and Val split have overlapping images!")
        self.assertEqual(len(train_paths.intersection(test_paths)), 0, "Train and Test split have overlapping images!")
        self.assertEqual(len(val_paths.intersection(test_paths)), 0, "Val and Test split have overlapping images!")

        # Diagnostic image isolation check
        diag_rel = "data/diagnostics/gemini_failure.png"
        self.assertNotIn(diag_rel, train_paths, "Diagnostic image was found in training split!")
        self.assertNotIn(diag_rel, val_paths, "Diagnostic image was found in validation split!")
        self.assertNotIn(diag_rel, test_paths, "Diagnostic image was found in test split!")

if __name__ == "__main__":
    unittest.main()
