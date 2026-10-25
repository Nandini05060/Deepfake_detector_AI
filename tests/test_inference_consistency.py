import unittest
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.backend.main import app, get_lxfd_predictor
from src.models.inference import LXDFDPredictor


class TestInferenceConsistencySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.predictor = get_lxfd_predictor()
        cls.test_set_dir = Path("final_dataset/val")
        if not cls.test_set_dir.exists():
            cls.test_set_dir = Path("test_set")

    # 1. Known real images
    def test_known_real_images(self):
        """Validates that verified authentic face images predict LIKELY_REAL with high specificity."""
        real_dirs = [
            Path(r"C:\Users\nandi\.cache\kagglehub\datasets\kshitizbhargava\deepfake-face-images\versions\1\Final Dataset\Real"),
            Path("data/external/IMDB_WIKI_real"),
            Path("final_dataset/val/real"),
            Path("test_set/real"),
        ]
        real_files = []
        for rd in real_dirs:
            if rd.exists():
                real_files.extend(list(rd.glob("*.jpg")) + list(rd.glob("*.png")))
        self.assertGreater(len(real_files), 0, "Evaluation suite must have real test images available.")

        # Test first 5 real images
        for rf in real_files[:5]:
            img = Image.open(rf).convert("RGB")
            res = self.predictor.predict(img)
            self.assertIn(res["final_decision"], ["LIKELY_REAL", "INCONCLUSIVE"])
            self.assertNotEqual(res["final_decision"], "LIKELY_FAKE", f"Real sample {rf.name} misclassified as LIKELY_FAKE")
            self.assertAlmostEqual(res["fake_probability"] + res["real_probability"], 1.0, places=3)

    # 2. Known fake images
    def test_known_fake_images(self):
        """Validates that verified fake faces from known generators predict LIKELY_FAKE."""
        fake_dirs = [
            Path(r"C:\Users\nandi\.cache\kagglehub\datasets\kshitizbhargava\deepfake-face-images\versions\1\Final Dataset\Fake"),
            Path("data/external/InsightFace_Swap"),
            Path("data/external/StableDiffusion_v1.5"),
            Path("final_dataset/val/fake"),
        ]
        fake_files = []
        for fd in fake_dirs:
            if fd.exists():
                fake_files.extend(list(fd.glob("*.jpg")) + list(fd.glob("*.png")))
        self.assertGreater(len(fake_files), 0, "Evaluation suite must have fake test images available.")

        # Test known fake images (e.g. StyleGAN2 / ProGAN)
        tested = 0
        for ff in fake_files:
            if "gemini" not in ff.name.lower() and "inpainting" not in ff.name.lower():
                img = Image.open(ff).convert("RGB")
                res = self.predictor.predict(img)
                # Known generators should trigger LIKELY_FAKE
                self.assertEqual(res["final_decision"], "LIKELY_FAKE", f"Known fake {ff.name} failed LIKELY_FAKE")
                tested += 1
                if tested >= 5:
                    break
        self.assertGreaterEqual(tested, 1)

    # 3. Unfamiliar-generator image (Gemini / Imagen 3 or Inpainting)
    def test_unfamiliar_generator_image_safe_handling(self):
        """Validates that novel-generator images with high discrepancy are safely routed to INCONCLUSIVE."""
        unfamiliar_dirs = [
            Path("data/gemini_dataset/test"),
            Path("data/external/StableDiffusion_Inpainting"),
        ]
        unfamiliar_files = []
        for gd in unfamiliar_dirs:
            if gd.exists():
                unfamiliar_files.extend([p for p in gd.glob("*.*") if p.suffix.lower() in [".jpg", ".png"]])

        self.assertGreater(len(unfamiliar_files), 0, "Must have unfamiliar generator test samples available.")
        sample = unfamiliar_files[0]
        img = Image.open(sample).convert("RGB")
        res = self.predictor.predict(img)
        # Must NOT be decisively marked as LIKELY_REAL
        self.assertIn(
            res["final_decision"],
            ["LIKELY_FAKE", "INCONCLUSIVE"],
            f"Unfamiliar fake image {sample.name} must never be decisively certified as LIKELY_REAL"
        )
        if res["final_decision"] == "INCONCLUSIVE":
            self.assertTrue(res["is_uncertain"])
            self.assertIsNotNone(res.get("generator_warning"))

    # 4. Invalid image uploads
    def test_invalid_image_uploads(self):
        """Validates that non-image MIME types and corrupted byte streams return HTTP 400."""
        # Test text file upload
        fake_txt = io.BytesIO(b"Hello, this is a plain text file, not an image.")
        res_txt = self.client.post("/predict", files={"file": ("invalid.txt", fake_txt, "text/plain")})
        self.assertEqual(res_txt.status_code, 400)
        self.assertIn("image", res_txt.json()["detail"].lower())

        # Test corrupt image bytes to /predict
        corrupt_bytes = io.BytesIO(b"\xFF\xD8\xFF\x00NotAValidJpegStream")
        res_corrupt_p = self.client.post("/predict", files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")})
        self.assertEqual(res_corrupt_p.status_code, 400)

        # Test corrupt image bytes to /explain
        corrupt_bytes2 = io.BytesIO(b"\x89PNG\r\n\x1a\nCorruptPayload")
        res_corrupt_e = self.client.post("/explain", files={"file": ("corrupt.png", corrupt_bytes2, "image/png")})
        self.assertEqual(res_corrupt_e.status_code, 400)

    # 5. Images with no detectable face
    def test_images_with_no_detectable_face(self):
        """Validates that images lacking a localized human face route safely to INCONCLUSIVE."""
        # Solid gray synthetic canvas
        blank_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        res = self.predictor.predict(blank_img, face_meta={"face_detected": False, "detected_face_count": 0})
        self.assertEqual(res["final_decision"], "INCONCLUSIVE")
        self.assertTrue(res["is_uncertain"])
        self.assertIn("FACE LOCALIZATION WARNING", res["generator_warning"])

    # 6. Very small or unusually sized images
    def test_very_small_or_unusually_sized_images(self):
        """Validates preprocessing and forward pass on extreme aspect ratios and micro-resolutions."""
        test_dimensions = [(16, 16), (24, 380), (450, 32), (64, 64)]
        for w, h in test_dimensions:
            canvas = Image.new("RGB", (w, h), color=(100, 150, 200))
            # Direct predictor call
            res = self.predictor.predict(canvas)
            self.assertIn("final_decision", res)
            self.assertIn("fake_probability", res)
            self.assertAlmostEqual(res["fake_probability"] + res["real_probability"], 1.0, places=3)

    # 7. Probability values and their sum
    def test_probability_values_and_sum(self):
        """Validates that all output probabilities are strictly bounded [0, 1] and sum to 1.0."""
        canvas = Image.new("RGB", (224, 224), color=(80, 120, 160))
        res = self.predictor.predict(canvas)

        fake_p = res["fake_probability"]
        real_p = res["real_probability"]
        self.assertGreaterEqual(fake_p, 0.0)
        self.assertLessEqual(fake_p, 1.0)
        self.assertGreaterEqual(real_p, 0.0)
        self.assertLessEqual(real_p, 1.0)
        self.assertAlmostEqual(fake_p + real_p, 1.0, places=3)

        # Branch probabilities
        sp = res["branch_outputs"]["spatial_branch_fake_prob"]
        fp = res["branch_outputs"]["frequency_branch_fake_prob"]
        self.assertGreaterEqual(sp, 0.0)
        self.assertLessEqual(sp, 1.0)
        self.assertGreaterEqual(fp, 0.0)
        self.assertLessEqual(fp, 1.0)

    # 8. Consistency between backend endpoints (/predict and /explain)
    def test_consistency_between_backend_endpoints(self):
        """Validates that /predict and /explain return identical predictions, probabilities, and verdicts."""
        buf = io.BytesIO()
        test_img = Image.new("RGB", (224, 224), color=(180, 140, 110))
        test_img.save(buf, format="JPEG")
        buf.seek(0)
        raw_bytes = buf.getvalue()

        resp_p = self.client.post("/predict", files={"file": ("test.jpg", io.BytesIO(raw_bytes), "image/jpeg")})
        resp_e = self.client.post("/explain", files={"file": ("test.jpg", io.BytesIO(raw_bytes), "image/jpeg")})

        self.assertEqual(resp_p.status_code, 200)
        self.assertEqual(resp_e.status_code, 200)

        data_p = resp_p.json()
        data_e = resp_e.json()

        # Both endpoints must agree exactly
        self.assertEqual(data_p["prediction"], data_e["prediction"])
        self.assertEqual(data_p["final_decision"], data_e["final_decision"])
        self.assertAlmostEqual(data_p["fake_probability"], data_e["fake_probability"], places=3)
        self.assertAlmostEqual(data_p["real_probability"], data_e["real_probability"], places=3)
        self.assertEqual(data_p["is_uncertain"], data_e["is_uncertain"])
        self.assertIn("visualizations", data_e)
        self.assertIn("gradcam_heatmap", data_e["visualizations"])

    # 9. Missing or incompatible checkpoints
    def test_missing_or_incompatible_checkpoint(self):
        """Validates that initializing a predictor with a nonexistent checkpoint raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            LXDFDPredictor(checkpoint_path="checkpoints/non_existent_checkpoint_xyz.pt")

    # 10. Uncertain predictions and warning generation
    def test_uncertain_predictions_and_warnings(self):
        """Validates that ambiguous or divergent predictions trigger INCONCLUSIVE verdict and explicit warnings."""
        # Test simulated divergence
        img = Image.new("RGB", (224, 224), color=(50, 60, 70))
        res = self.predictor.predict(img)

        # Inconclusive predictions must have an explicit advisory warning
        if res["is_uncertain"]:
            self.assertEqual(res["final_decision"], "INCONCLUSIVE")
            self.assertIsNotNone(res["generator_warning"])
            self.assertTrue(len(res["generator_warning"]) > 20)


if __name__ == "__main__":
    unittest.main()
