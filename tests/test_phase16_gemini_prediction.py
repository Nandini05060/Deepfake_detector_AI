"""
Phase 16 Regression & Validation Test Suite
Tests Gemini generalization, problematic sample reproduction, genuine specificity,
face localization, and probability pipeline integrity.
"""

import sys
import unittest
import hashlib
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.inference import LXDFDPredictor
from app.backend.main import extract_face_crop_with_meta


class TestPhase16GeminiPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = "experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt"
        cls.cal_path = "experiments/phase15_gemini_generalization/calibration_params_phase15.json"
        cls.predictor = LXDFDPredictor(
            checkpoint_path=cls.ckpt_path,
            calibration_path=cls.cal_path,
            min_floor=0.25
        )

    def test_01_checkpoint_and_calibration_integrity(self):
        """Verifies active checkpoint SHA-256, existence, and calibrated temperature."""
        ckpt_file = Path(self.ckpt_path)
        self.assertTrue(ckpt_file.exists(), f"Phase 15 checkpoint missing at {self.ckpt_path}")

        sha256 = hashlib.sha256()
        with open(ckpt_file, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        ckpt_hash = sha256.hexdigest()
        self.assertTrue(ckpt_hash.startswith("2552ab2b"), f"Unexpected checkpoint hash: {ckpt_hash}")

        self.assertTrue(self.predictor.calibration_enabled, "Calibration should be enabled")
        self.assertAlmostEqual(self.predictor.scaler.temperature, 0.7169, places=3)

    def test_02_known_gemini_benchmark_sample(self):
        """Verifies that the benchmark Gemini sample is classified as LIKELY_FAKE with high confidence."""
        sample_path = Path("samples/gemini_imagen3_sample.png")
        if not sample_path.exists():
            sample_path = Path("app/frontend/samples/gemini_imagen3_sample.png")
        self.assertTrue(sample_path.exists(), f"Benchmark sample missing at {sample_path}")

        img = Image.open(sample_path).convert("RGB")
        crop, meta = extract_face_crop_with_meta(img, margin=0.50)
        res = self.predictor.predict(crop, face_meta=meta)

        self.assertEqual(res["final_decision"], "LIKELY_FAKE")
        self.assertGreater(res["fake_probability"], 0.95)
        self.assertGreater(res["branch_outputs"]["spatial_branch_fake_prob"], 0.95)

    def test_03_problematic_gemini_sample_reproduced_trace(self):
        """
        Reproduces the exact live failure on the problematic Gemini upload:
        - Under standard crop (margin 0.50), face bounds clamp vertically.
        - Spatial branch exhibits blindness on interior skin (fake prob < 0.05).
        - Frequency branch detects spectral suspicion (fake prob >= 0.28).
        - Vigilant Forensic Policy intercepts divergence: verdict is INCONCLUSIVE (NOT REAL).
        - Under extended context (margin 0.75), boundary artifacts trigger LIKELY_FAKE (> 0.90).
        """
        prob_path = Path("data/diagnostic_uploads/gemini_user_uploaded_portrait.jpg")
        self.assertTrue(prob_path.exists(), f"Diagnostic sample missing at {prob_path}")

        img = Image.open(prob_path).convert("RGB")
        self.assertEqual(img.size, (1024, 558))

        # Standard crop (0.50)
        crop_std, meta_std = extract_face_crop_with_meta(img, margin=0.50)
        res_std = self.predictor.predict(crop_std, face_meta=meta_std)

        # 1. Assert standard crop reproduces the exact reported state
        self.assertEqual(res_std["final_decision"], "INCONCLUSIVE")
        self.assertEqual(res_std["confidence_level"], "LOW")
        self.assertIn("Spectral Suspicion with Spatial Blindness", res_std["empirical_reliability"])
        self.assertLess(res_std["fake_probability"], 0.05)
        self.assertGreater(res_std["real_probability"], 0.95)
        self.assertLess(res_std["branch_outputs"]["spatial_branch_fake_prob"], 0.05)
        self.assertGreaterEqual(res_std["branch_outputs"]["frequency_branch_fake_prob"], 0.28)

        # 2. Epistemic confidence fix check: uncertain predictions must have low numeric confidence
        self.assertLessEqual(res_std["confidence"], 0.50)

        # 3. Assert extended context crop resolves the boundary artifacts
        crop_ext, meta_ext = extract_face_crop_with_meta(img, margin=0.75)
        res_ext = self.predictor.predict(crop_ext, face_meta=meta_ext)
        self.assertEqual(res_ext["final_decision"], "LIKELY_FAKE")
        self.assertGreater(res_ext["fake_probability"], 0.85)
        self.assertGreater(res_ext["branch_outputs"]["spatial_branch_fake_prob"], 0.85)

    def test_04_other_available_gemini_fakes(self):
        """Verifies high recall on other available Gemini Phase 11 samples."""
        p11_samples = [
            Path("data/gemini_dataset/phase11/gemini_p11_08_1792780802191.jpg"),
            Path("data/gemini_dataset/phase11/gemini_p11_09_1792780886282.jpg")
        ]
        for p in p11_samples:
            if p.exists():
                img = Image.open(p).convert("RGB")
                crop, meta = extract_face_crop_with_meta(img, margin=0.50)
                res = self.predictor.predict(crop, face_meta=meta)
                self.assertEqual(res["final_decision"], "LIKELY_FAKE", f"Failed on {p.name}")
                self.assertGreater(res["fake_probability"], 0.95)

    def test_05_genuine_faces_ffhq(self):
        """Verifies zero false positives on genuine FFHQ human camera photographs."""
        ffhq_samples = [
            Path("data/verified_faces/validation/real/FFHQ_Real_00699.jpg"),
            Path("data/verified_faces/validation/real/FFHQ_Real_00701.jpg"),
            Path("data/verified_faces/validation/real/FFHQ_Real_00702.jpg")
        ]
        for p in ffhq_samples:
            if p.exists():
                img = Image.open(p).convert("RGB")
                crop, meta = extract_face_crop_with_meta(img, margin=0.50)
                res = self.predictor.predict(crop, face_meta=meta)
                # Genuine faces must NOT be classified as LIKELY_FAKE
                self.assertNotEqual(res["final_decision"], "LIKELY_FAKE", f"False fake on {p.name}")
                self.assertLess(res["fake_probability"], 0.20)

    def test_06_challenging_imdb_wiki_genuine_specificity(self):
        """
        Monitors challenging historical portraits from IMDB-WIKI.
        Ensures false positives remain zero under standard crop margin 0.50.
        """
        imdb_samples = [
            Path("data/verified_faces/validation/real/IMDB_WIKI_IMDB_WIKI_real_0100.jpg"),
            Path("data/verified_faces/validation/real/IMDB_WIKI_IMDB_WIKI_real_0101.jpg")
        ]
        for p in imdb_samples:
            if p.exists():
                img = Image.open(p).convert("RGB")
                crop, meta = extract_face_crop_with_meta(img, margin=0.50)
                res = self.predictor.predict(crop, face_meta=meta)
                # Ensure no false positive fake
                self.assertNotEqual(res["final_decision"], "LIKELY_FAKE", f"False fake on {p.name}")

    def test_07_image_without_detectable_face(self):
        """Verifies that an image without a face triggers INCONCLUSIVE and LOW confidence."""
        # 300x300 uniform color / noise image
        noise_arr = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        pil_noise = Image.fromarray(noise_arr)

        crop, meta = extract_face_crop_with_meta(pil_noise, margin=0.50)
        self.assertFalse(meta["face_detected"])

        res = self.predictor.predict(crop, face_meta=meta)
        self.assertEqual(res["final_decision"], "INCONCLUSIVE")
        self.assertEqual(res["confidence_level"], "LOW")
        self.assertIn("Face Localization Unverified", res["empirical_reliability"])
        self.assertTrue(res["is_uncertain"])

    def test_08_invalid_inputs(self):
        """Verifies that non-image inputs raise a TypeError."""
        with self.assertRaises(TypeError):
            self.predictor.predict(12345)


if __name__ == "__main__":
    unittest.main()
