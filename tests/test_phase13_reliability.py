import unittest
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from fastapi.testclient import TestClient

from app.backend.main import app, get_lxfd_predictor
from src.models.lxfd_model import LXDFDModel
from src.evaluation.calibration import TemperatureScaler


class TestPhase13ReliabilitySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prod_ckpt_path = Path("checkpoints/lxfd_phase10_modern_generators_best.pt")
        cls.prod_expected_sha256 = "13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05"
        cls.candidate_ckpt_path = Path("experiments/phase12_multiscale_v1/lxfd_multiscale12_best.pt")
        cls.candidate_expected_sha256 = "a2b9989205f255248ca7c378e25164b88fbc47691ce8b9b9b815fcd00a72e773"
        cls.calib_path = Path("checkpoints/calibration_params_phase10.json")

    # 1. Checkpoint Identity & Separation
    def test_checkpoint_identity_and_separation(self):
        """Verifies production and research checkpoints exist with distinct, authenticated hashes."""
        self.assertTrue(self.prod_ckpt_path.exists())
        self.assertTrue(self.candidate_ckpt_path.exists())

        h_prod = hashlib.sha256(self.prod_ckpt_path.read_bytes()).hexdigest()
        h_cand = hashlib.sha256(self.candidate_ckpt_path.read_bytes()).hexdigest()

        self.assertEqual(h_prod, self.prod_expected_sha256)
        self.assertEqual(h_cand, self.candidate_expected_sha256)
        self.assertNotEqual(h_prod, h_cand, "Production and candidate checkpoints must be separate files!")

    # 2. Calibration Consistency
    def test_calibration_consistency(self):
        """Verifies temperature parameter loading and monotonic probability scaling."""
        self.assertTrue(self.calib_path.exists())
        scaler = TemperatureScaler()
        scaler.load(str(self.calib_path))

        self.assertAlmostEqual(float(scaler.temperature), 1.6344, places=3)
        # 0.5 raw probability should map exactly to 0.5 calibrated probability
        cal_half = float(scaler.calibrate_probs(np.array([0.5]))[0])
        self.assertAlmostEqual(cal_half, 0.5, places=4)

        # Monotonicity test
        probs = np.linspace(0.01, 0.99, 50)
        cal_probs = scaler.calibrate_probs(probs)
        self.assertTrue(np.all(np.diff(cal_probs) >= 0), "Temperature scaling must be strictly monotonic!")

    # 3. Threshold Boundary Behavior
    def test_threshold_boundary_behavior(self):
        """Verifies that decision threshold and uncertainty boundaries behave properly."""
        predictor = get_lxfd_predictor()
        # Default decision threshold must be 0.50
        self.assertEqual(predictor.decision_threshold, 0.50)
        # Uncertainty band must cover [0.35, 0.65] or [0.40, 0.60]
        self.assertGreaterEqual(predictor.uncertainty_band, 0.10)
        self.assertLessEqual(predictor.uncertainty_band, 0.20)

    # 4. Inconclusive Decision Routing
    def test_inconclusive_decision_routing(self):
        """Verifies ambiguous or out-of-distribution cases route to INCONCLUSIVE."""
        predictor = get_lxfd_predictor()

        # Simulate image prediction with face_meta where face detection is unverified
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        res_no_face = predictor.predict(dummy_img, face_meta={"face_detected": False})

        self.assertEqual(res_no_face["prediction"], "INCONCLUSIVE")
        self.assertEqual(res_no_face["final_decision"], "INCONCLUSIVE")
        self.assertIn("FACE LOCALIZATION WARNING", res_no_face.get("generator_warning", ""))

    # 5. Face Detection Failure Handling
    def test_face_detection_failures(self):
        """Verifies that blank / uniform images trigger graceful fallback without pipeline crash."""
        predictor = get_lxfd_predictor()
        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        res = predictor.predict(black_img, face_meta={"face_detected": False, "detected_face_count": 0})
        self.assertIn("final_decision", res)
        self.assertEqual(res["final_decision"], "INCONCLUSIVE")

    # 6. Corrupted or Unsupported Inputs
    def test_corrupted_or_unsupported_inputs(self):
        """Verifies API returns 400 Bad Request on corrupted bytes or non-image types."""
        # Non-image file upload
        fake_txt = io.BytesIO(b"This is not a valid JPEG/PNG file.")
        res_txt = self.client.post("/predict", files={"file": ("test.txt", fake_txt, "text/plain")})
        self.assertEqual(res_txt.status_code, 400)

        # Corrupted image bytes
        fake_corrupted = io.BytesIO(b"\xFF\xD8\xFF\xE0\x00\x10JFIF corrupted garbage bytes...")
        res_corrupt = self.client.post("/predict", files={"file": ("corrupt.jpg", fake_corrupted, "image/jpeg")})
        self.assertEqual(res_corrupt.status_code, 400)

    # 7. Severe Compression and Low-Resolution Inputs
    def test_severe_compression_and_low_resolution_inputs(self):
        """Verifies model forward pass handles tiny inputs without shape mismatch exceptions."""
        model = LXDFDModel(fusion_type="attention", min_floor=0.25, pretrained=False)
        model.eval()
        # Test extreme small tensor resized to 224x224
        tiny_img = Image.new("RGB", (16, 16), color=(200, 100, 50))
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_in = transform(tiny_img).unsqueeze(0)
        with torch.no_grad():
            logit, meta = model(t_in)
        self.assertEqual(logit.shape, (1,))
        self.assertIn("w_s", meta)

    # 8. Model Selection Behavior
    def test_model_selection_behavior(self):
        """Verifies API registers both production and candidate models with explicit status."""
        res = self.client.get("/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        models = {m["id"]: m for m in data.get("models", [])}

        self.assertIn("lxfd_attention", models)
        self.assertTrue(models["lxfd_attention"]["is_active"])
        self.assertTrue(models["lxfd_attention"]["recommended"])

        self.assertIn("lxfd_multiscale12", models)
        self.assertFalse(models["lxfd_multiscale12"]["is_active"])
        self.assertFalse(models["lxfd_multiscale12"]["recommended"])

    # 9. API Response Schema Phase 13
    def test_api_response_schema_phase13(self):
        """Verifies /predict response schema includes 3-way decision taxonomy and disclaimer."""
        img = Image.new("RGB", (224, 224), color=(150, 150, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        res = self.client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        payload = res.json()

        self.assertIn("final_decision", payload)
        self.assertIn(payload["final_decision"], ["LIKELY_REAL", "LIKELY_FAKE", "INCONCLUSIVE"])
        self.assertIn("verdict", payload)
        self.assertIn("forensic_disclaimer", payload)
        self.assertIn("fake_probability", payload)
        self.assertIn("real_probability", payload)
        self.assertIn("calibrated", payload)
        self.assertIn("checkpoint_used", payload)

    # 10. Preservation of Locked Evaluation Manifest
    def test_preservation_of_locked_evaluation_manifest(self):
        """Verifies data/manifests/phase11_generator_manifest.csv is preserved intact with exactly 1,864 rows."""
        manifest_path = Path("data/manifests/phase11_generator_manifest.csv")
        self.assertTrue(manifest_path.exists())

        import csv
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 1864)

        test_splits = {'test_indomain', 'test_unseen_inpaint', 'test_locked', 'test_gemini_heldout'}
        test_rows = [r for r in reader if r.get('split') in test_splits]
        self.assertEqual(len(test_rows), 457)
        self.assertEqual(sum(1 for r in test_rows if r.get('label') == 'REAL'), 120)
        self.assertEqual(sum(1 for r in test_rows if r.get('label') == 'FAKE'), 337)

    # 11. Phase 13 Metrics and Reports Endpoints
    def test_phase13_metrics_and_reports_endpoints(self):
        """Verifies /research/phase13/metrics and /reports contain Phase 13 deliverables."""
        res_m = self.client.get("/research/phase13/metrics")
        self.assertEqual(res_m.status_code, 200)
        m_data = res_m.json()
        self.assertIn("comparison_benchmark", m_data)
        self.assertEqual(len(m_data["comparison_benchmark"]), 4)

        res_r = self.client.get("/reports")
        self.assertEqual(res_r.status_code, 200)
        r_data = res_r.json()
        report_ids = [r["id"] for r in r_data.get("reports", [])]

        self.assertIn("REP-PHASE13-BASE-VERIF", report_ids)
        self.assertIn("REP-PHASE13-THRESH-CALIB", report_ids)
        self.assertIn("REP-PHASE13-UNCERTAINTY", report_ids)
        self.assertIn("REP-PHASE13-CROSS-GEN", report_ids)
        self.assertIn("REP-PHASE13-FP-AUDIT", report_ids)
        self.assertIn("REP-PHASE13-ROBUSTNESS", report_ids)
        self.assertIn("REP-PHASE13-FINAL-COMPARE", report_ids)
        self.assertIn("REP-PHASE13-PROVENANCE", report_ids)

        # Download test
        res_dl = self.client.get("/reports/download/phase13_threshold_analysis.md")
        self.assertEqual(res_dl.status_code, 200)
        self.assertIn("Validation-Only Threshold and Calibration Study", res_dl.json()["content"])


if __name__ == "__main__":
    unittest.main()
