"""
Comprehensive Regression & Generalization Test Suite
Tests:
- Known real faces
- Known fake faces
- Held-out Gemini-generated faces
- Held-out faces from other generators (Inpainting)
- Genuine photographs from unseen sources (IMDB-WIKI)
- Images with no detectable face
- Invalid and corrupted image files
- Various image resolutions
- Application endpoint execution (/predict, /explain)
- Phase 15 candidate checkpoint integrity
"""

import unittest
import io
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.backend.main import app, get_lxfd_predictor
from src.models.lxfd_model import LXDFDModel
from src.models.inference import LXDFDPredictor

class TestInferenceRegressionSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.predictor = get_lxfd_predictor()
        cls.repo_root = Path(__file__).resolve().parent.parent

        # Load verified manifest
        manifest_path = cls.repo_root / "data" / "verified_faces" / "verified_manifest.csv"
        if manifest_path.exists():
            import pandas as pd
            cls.manifest_df = pd.read_csv(manifest_path)
        else:
            cls.manifest_df = None

    # 1. Known real faces
    def test_known_real_faces(self):
        """Verifies that known authentic faces from FFHQ photographic controls predict LIKELY_REAL."""
        if self.manifest_df is not None:
            reals = self.manifest_df[(self.manifest_df["split"] == "test") & (self.manifest_df["generator_family"] == "FFHQ_Real")]
            self.assertGreater(len(reals), 0)
            sample_path = self.repo_root / reals.iloc[0]["relative_path"]
        else:
            sample_path = Path("app/frontend/samples/real_authentic_sample.jpg")

        self.assertTrue(sample_path.exists())
        img = Image.open(sample_path).convert("RGB")
        res = self.predictor.predict(img)

        self.assertIn(res["final_decision"], ["LIKELY_REAL", "INCONCLUSIVE"])
        self.assertNotEqual(res["final_decision"], "LIKELY_FAKE")
        self.assertLess(res["fake_probability"], 0.50)

    # 2. Known fake faces
    def test_known_fake_faces(self):
        """Verifies that in-domain fake faces from StyleGAN2 predict LIKELY_FAKE with high confidence."""
        if self.manifest_df is not None:
            fakes = self.manifest_df[(self.manifest_df["split"] == "test") & (self.manifest_df["generator_family"] == "StyleGAN2")]
            self.assertGreater(len(fakes), 0)
            sample_path = self.repo_root / fakes.iloc[0]["relative_path"]
        else:
            sample_path = Path("app/frontend/samples/stylegan2_sample.jpg")

        self.assertTrue(sample_path.exists())
        img = Image.open(sample_path).convert("RGB")
        res = self.predictor.predict(img)

        self.assertEqual(res["final_decision"], "LIKELY_FAKE")
        self.assertGreater(res["fake_probability"], 0.80)

    # 3. Held-out Gemini-generated faces
    def test_held_out_gemini_generated_faces(self):
        """Verifies that dedicated held-out Gemini faces are never certified as LIKELY_REAL."""
        if self.manifest_df is not None:
            geminis = self.manifest_df[(self.manifest_df["split"] == "test") & (self.manifest_df["generator_family"] == "Google_Gemini_Imagen3")]
            self.assertGreater(len(geminis), 0)
            sample_path = self.repo_root / geminis.iloc[0]["relative_path"]
        else:
            sample_path = Path("app/frontend/samples/gemini_imagen3_sample.png")

        self.assertTrue(sample_path.exists())
        img = Image.open(sample_path).convert("RGB")
        res = self.predictor.predict(img)

        # Must strictly NEVER produce an affirmative real certification
        self.assertIn(res["final_decision"], ["LIKELY_FAKE", "INCONCLUSIVE"])
        self.assertNotEqual(res["final_decision"], "LIKELY_REAL", "Unfamiliar Gemini fake must never be certified as LIKELY_REAL!")

    # 4. Held-out faces from other generators (Inpainting)
    def test_held_out_faces_other_generators(self):
        """Verifies that completely unseen inpainting fakes are safely classified without false real certifications."""
        if self.manifest_df is not None:
            inpaints = self.manifest_df[(self.manifest_df["split"] == "test") & (self.manifest_df["generator_family"] == "StableDiffusion_Inpainting")]
            self.assertGreater(len(inpaints), 0)
            sample_path = self.repo_root / inpaints.iloc[0]["relative_path"]
        else:
            inpaint_dir = Path("data/external/StableDiffusion_Inpainting")
            sample_path = list(inpaint_dir.glob("*.jpg"))[0]

        self.assertTrue(sample_path.exists())
        img = Image.open(sample_path).convert("RGB")
        res = self.predictor.predict(img)

        self.assertIn(res["final_decision"], ["LIKELY_FAKE", "INCONCLUSIVE"])

    # 5. Genuine photographs from unseen sources (IMDB-WIKI)
    def test_genuine_photographs_unseen_sources(self):
        """Verifies that real faces from the independent IMDB-WIKI cohort are evaluated safely."""
        if self.manifest_df is not None:
            imdbs = self.manifest_df[(self.manifest_df["split"] == "test") & (self.manifest_df["generator_family"] == "IMDB_WIKI")]
            self.assertGreater(len(imdbs), 0)
            sample_path = self.repo_root / imdbs.iloc[0]["relative_path"]
        else:
            sample_path = list(Path("data/external/IMDB_WIKI_real").glob("*.jpg"))[0]

        self.assertTrue(sample_path.exists())
        img = Image.open(sample_path).convert("RGB")
        res = self.predictor.predict(img)

        self.assertIn(res["final_decision"], ["LIKELY_REAL", "INCONCLUSIVE"])
        self.assertNotEqual(res["final_decision"], "LIKELY_FAKE")

    # 6. Images with no detectable face
    def test_images_with_no_detectable_face(self):
        """Verifies that canvases with no localized human face route to INCONCLUSIVE with localization warning."""
        blank_canvas = Image.new("RGB", (256, 256), color=(140, 140, 140))
        res = self.predictor.predict(blank_canvas, face_meta={"face_detected": False, "detected_face_count": 0})

        self.assertEqual(res["final_decision"], "INCONCLUSIVE")
        self.assertTrue(res["is_uncertain"])
        self.assertIn("FACE LOCALIZATION WARNING", res["generator_warning"])

    # 7. Invalid and corrupted image files
    def test_invalid_and_corrupted_image_files(self):
        """Verifies that corrupted payloads and non-image uploads return HTTP 400 Bad Request."""
        # Non-image plain text
        txt_buf = io.BytesIO(b"Plain text payload - not a JPEG or PNG")
        resp_txt = self.client.post("/predict", files={"file": ("test.txt", txt_buf, "text/plain")})
        self.assertEqual(resp_txt.status_code, 400)

        # Corrupted JPEG header
        corrupt_buf = io.BytesIO(b"\xFF\xD8\xFF\xDB\x00\x43TruncatedDamagedStream")
        resp_corrupt = self.client.post("/predict", files={"file": ("bad.jpg", corrupt_buf, "image/jpeg")})
        self.assertEqual(resp_corrupt.status_code, 400)

    # 8. Various image resolutions
    def test_various_image_resolutions(self):
        """Verifies robust bicubic preprocessing across extreme resolutions and aspect ratios."""
        resolutions = [(16, 16), (64, 64), (640, 180), (1024, 1024)]
        for w, h in resolutions:
            stimulus = Image.new("RGB", (w, h), color=(120, 80, 160))
            res = self.predictor.predict(stimulus)
            self.assertIn("final_decision", res)
            self.assertAlmostEqual(res["fake_probability"] + res["real_probability"], 1.0, places=3)

    # 9. Application endpoint execution (/predict and /explain)
    def test_application_endpoints_live_execution(self):
        """Verifies that live FastAPI endpoints return 200 OK with complete metadata schemas."""
        buf = io.BytesIO()
        test_img = Image.new("RGB", (224, 224), color=(150, 120, 90))
        test_img.save(buf, format="JPEG")
        raw = buf.getvalue()

        resp_p = self.client.post("/predict", files={"file": ("sample.jpg", io.BytesIO(raw), "image/jpeg")})
        self.assertEqual(resp_p.status_code, 200)
        data_p = resp_p.json()
        self.assertIn("final_decision", data_p)
        self.assertIn("fake_probability", data_p)
        self.assertIn("model_version", data_p)

        resp_e = self.client.post("/explain", files={"file": ("sample.jpg", io.BytesIO(raw), "image/jpeg")})
        self.assertEqual(resp_e.status_code, 200)
        data_e = resp_e.json()
        self.assertIn("visualizations", data_e)
        self.assertIn("gradcam_heatmap", data_e["visualizations"])
        self.assertIn("dct_spectrum", data_e["visualizations"])

    # 10. Phase 15 candidate checkpoint integrity
    def test_phase15_candidate_checkpoint_integrity(self):
        """Verifies that the trained Phase 15 candidate checkpoint loads and executes forward pass."""
        cand_path = self.repo_root / "experiments" / "phase15_gemini_generalization" / "lxfd_phase15_generalization_best.pt"
        self.assertTrue(cand_path.exists(), "Phase 15 candidate checkpoint must exist in experiments/")

        cand_model = LXDFDModel(fusion_type="attention", min_floor=0.25, pretrained=False)
        cand_ckpt = torch.load(cand_path, map_location="cpu", weights_only=False)
        cand_model.load_state_dict(cand_ckpt["model_state_dict"])
        cand_model.eval()

        dummy_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logit, meta = cand_model(dummy_tensor)

        self.assertEqual(logit.shape, (1,))
        self.assertIn("w_s", meta)
        self.assertIn("w_f", meta)

if __name__ == "__main__":
    unittest.main()
