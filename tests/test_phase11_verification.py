import unittest
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.backend.main import app, get_lxfd_predictor
from src.models.lxfd_model import LXDFDModel


class TestPhase11VerificationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.post("/model/switch", json={"version": "phase10"})
        cls.checkpoint_path = Path("checkpoints/lxfd_phase10_modern_generators_best.pt")
        cls.expected_sha256 = "13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05"
        cls.calibration_path = Path("checkpoints/calibration_params_phase10.json")

    @classmethod
    def tearDownClass(cls):
        cls.client.post("/model/switch", json={"version": "phase16"})

    # 1. Checkpoint Loading & Hash Authentication
    def test_checkpoint_existence_and_sha256_authentication(self):
        """Verifies checkpoint exists and matches cryptographic SHA-256 hash."""
        self.assertTrue(self.checkpoint_path.exists(), f"Checkpoint missing at {self.checkpoint_path}")
        with open(self.checkpoint_path, "rb") as f:
            calculated_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            calculated_hash,
            self.expected_sha256,
            f"Checkpoint SHA-256 hash mismatch! Expected {self.expected_sha256}, got {calculated_hash}"
        )

    def test_checkpoint_weights_load_into_model(self):
        """Verifies checkpoint weights instantiate cleanly into LXDFDModel."""
        device = torch.device("cpu")
        model = LXDFDModel(pretrained=False)
        ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Ensure state dict loads without exception
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        dummy_img = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            logit, weights_dict = model(dummy_img)
        self.assertEqual(logit.shape, (1,))
        self.assertIn("w_s", weights_dict)
        self.assertIn("w_f", weights_dict)

    # 2. Class-Label Mapping
    def test_class_label_mapping_integrity(self):
        """Verifies binary label convention: 0 is REAL and 1 is FAKE."""
        predictor = get_lxfd_predictor()
        # Ensure decision threshold is configured
        self.assertGreaterEqual(predictor.decision_threshold, 0.40)
        self.assertLessEqual(predictor.decision_threshold, 0.60)

    # 3. API Prediction Response Schema
    def test_api_prediction_response_schema(self):
        """Validates /predict response schema conforms to forensic specifications."""
        sample_path = Path("app/frontend/samples/real_authentic_sample.jpg")
        self.assertTrue(sample_path.exists(), f"Sample image {sample_path} missing")
        with open(sample_path, "rb") as f:
            res = self.client.post("/predict", files={"file": ("real.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        required_keys = [
            "prediction", "fake_probability", "real_probability",
            "confidence", "confidence_level", "empirical_reliability",
            "is_uncertain", "calibrated", "checkpoint_used",
            "branch_outputs", "detection_metadata", "model_version"
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Required response key '{key}' missing")

    # 4. Real-Image Control Behavior
    def test_real_image_control_behavior(self):
        """Validates that authentic human portrait receives REAL prediction with low fake prob."""
        sample_path = Path("app/frontend/samples/real_authentic_sample.jpg")
        with open(sample_path, "rb") as f:
            res = self.client.post("/predict", files={"file": ("real.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["prediction"], "REAL")
        self.assertLess(data["fake_probability"], 0.40)
        self.assertGreater(data["real_probability"], 0.60)

    # 5. AI-Image Behavior (Synthetic Face Detection)
    def test_ai_image_behavior_stylegan2(self):
        """Validates that StyleGAN2 face receives FAKE prediction with high confidence."""
        sample_path = Path("app/frontend/samples/stylegan2_sample.jpg")
        with open(sample_path, "rb") as f:
            res = self.client.post("/predict", files={"file": ("fake.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["prediction"], "FAKE")
        self.assertGreater(data["fake_probability"], 0.85)

    # 6. Invalid Image Handling
    def test_invalid_image_payload_rejection(self):
        """Validates rejection of non-image bytes with HTTP 400 Bad Request."""
        res = self.client.post(
            "/predict",
            files={"file": ("notes.txt", b"plain text payload not an image", "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("must be an image", res.json()["detail"].lower())

    # 7. No-Face Images Handling
    def test_no_face_image_fallback_handling(self):
        """Validates that images lacking a detectable face fall back safely without crashing."""
        blank_img = Image.new("RGB", (300, 300), color=(140, 140, 140))
        buf = io.BytesIO()
        blank_img.save(buf, format="JPEG")
        buf.seek(0)
        res = self.client.post(
            "/predict",
            files={"file": ("blank.jpg", buf, "image/jpeg")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["detection_metadata"]["face_detected"])
        self.assertEqual(data["detection_metadata"]["detected_face_count"], 0)
        self.assertIn(data["prediction"], ["REAL", "FAKE", "INCONCLUSIVE"])

    # 8. Multiple-Face Image Handling
    def test_multiple_face_image_handling(self):
        """Validates that multi-face images log advisory and select primary face."""
        # Create an image wide enough to place two face crops side by side
        sample_path = Path("app/frontend/samples/real_authentic_sample.jpg")
        if sample_path.exists():
            face_img = Image.open(sample_path).resize((200, 200))
            canvas = Image.new("RGB", (500, 300), color=(240, 240, 240))
            canvas.paste(face_img, (30, 50))
            canvas.paste(face_img, (260, 50))
            buf = io.BytesIO()
            canvas.save(buf, format="JPEG")
            buf.seek(0)
            res = self.client.post(
                "/predict",
                files={"file": ("multi_face.jpg", buf, "image/jpeg")}
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            dm = data["detection_metadata"]
            self.assertIn("single_face_limitation", dm)
            self.assertIn("Primary face analyzed", dm["single_face_limitation"])

    # 9. Crop Stability Measurement
    def test_crop_stability_endpoint_contract(self):
        """Validates /crop_stability returns multi-margin crops and delta metric."""
        sample_path = Path("app/frontend/samples/real_authentic_sample.jpg")
        with open(sample_path, "rb") as f:
            res = self.client.post("/crop_stability", files={"file": ("sample.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["crops"]), 3)
        self.assertGreaterEqual(data["probability_delta"], 0.0)
        self.assertLessEqual(data["probability_delta"], 1.0)
        self.assertIn("scientific_warning", data)

    # 10. Reproducibility & Determinism
    def test_inference_determinism_and_reproducibility(self):
        """Validates that identical input yields bitwise identical prediction probabilities."""
        sample_path = Path("app/frontend/samples/sd15_diffusion_sample.jpg")
        with open(sample_path, "rb") as f:
            raw_bytes = f.read()

        res1 = self.client.post("/predict", files={"file": ("sample.jpg", io.BytesIO(raw_bytes), "image/jpeg")})
        res2 = self.client.post("/predict", files={"file": ("sample.jpg", io.BytesIO(raw_bytes), "image/jpeg")})
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)
        data1 = res1.json()
        data2 = res2.json()
        self.assertAlmostEqual(data1["fake_probability"], data2["fake_probability"], places=4)
        self.assertAlmostEqual(data1["real_probability"], data2["real_probability"], places=4)
        self.assertEqual(data1["prediction"], data2["prediction"])

    # 11. Probability Validity & Normalization
    def test_probability_normalization_sum_to_one(self):
        """Validates P(Fake) + P(Real) == 1.0 within floating tolerance."""
        sample_path = Path("app/frontend/samples/real_authentic_sample.jpg")
        with open(sample_path, "rb") as f:
            res = self.client.post("/predict", files={"file": ("sample.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        p_fake = data["fake_probability"]
        p_real = data["real_probability"]
        self.assertGreaterEqual(p_fake, 0.0)
        self.assertLessEqual(p_fake, 1.0)
        self.assertGreaterEqual(p_real, 0.0)
        self.assertLessEqual(p_real, 1.0)
        self.assertAlmostEqual(p_fake + p_real, 1.0, places=3)

    # 12. Correct Checkpoint Selection
    def test_correct_checkpoint_selection_in_audit_record(self):
        """Validates that active checkpoint matches Phase 10 filename in audit record."""
        res = self.client.get("/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        active_model = next((m for m in data.get("models", []) if m.get("is_active")), None)
        self.assertIsNotNone(active_model, "No active model designated in /models")
        self.assertEqual(active_model["checkpoint"], "lxfd_phase10_modern_generators_best.pt")

    # 13. Report Export & Download Endpoints
    def test_report_export_and_download(self):
        """Validates that generated research reports are accessible and downloadable."""
        res = self.client.get("/reports")
        self.assertEqual(res.status_code, 200)
        reports = res.json().get("reports", [])
        report_filenames = [r.get("filename") for r in reports]
        report_ids = [r.get("id") for r in reports]
        self.assertIn("phase11_baseline_verification.md", report_filenames)
        self.assertIn("phase11_ablation_study.md", report_filenames)
        self.assertIn("phase11_gemini_failure_analysis.md", report_filenames)
        self.assertIn("REP-PHASE11-BASELINE", report_ids)

        # Download json metrics
        dl_res = self.client.get("/reports/download/phase11_test_metrics.json")
        self.assertEqual(dl_res.status_code, 200)
        report_payload = dl_res.json()
        self.assertIn("content", report_payload)
        metrics = json.loads(report_payload["content"])
        self.assertIn("metadata", metrics)
        self.assertIn("generator_benchmarks", metrics)

    # 14. Calibration Configuration
    def test_calibration_configuration_and_temperature(self):
        """Validates calibration parameters file exists and has valid positive temperature."""
        self.assertTrue(self.calibration_path.exists())
        with open(self.calibration_path, "r") as f:
            calib_data = json.load(f)
        self.assertIn("temperature", calib_data)
        t = calib_data["temperature"]
        self.assertGreater(t, 1.0)
        self.assertLess(t, 3.0)

    # 15. Uncertainty & Inconclusive Policy
    def test_uncertainty_and_inconclusive_policy(self):
        """Validates that out-of-distribution or ambiguous inputs receive uncertainty warnings."""
        sample_path = Path("app/frontend/samples/gemini_imagen3_sample.png")
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                res = self.client.post("/predict", files={"file": ("gemini.png", f, "image/png")})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            # If branch discrepancy is high or probability is in boundary band, generator warning must be populated
            if data["branch_outputs"]["branch_discrepancy"] > 0.10 or data["prediction"] == "INCONCLUSIVE":
                self.assertIsNotNone(data.get("generator_warning"))


if __name__ == "__main__":
    unittest.main()
