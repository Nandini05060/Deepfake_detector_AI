import unittest
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.backend.main import app

class TestResearchFeaturesEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_shortcut_detector_invalid_file(self):
        """Rejects non-image files with 400 Bad Request."""
        res = self.client.post(
            "/shortcut_detector",
            files={"file": ("test.txt", b"plain text payload", "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("must be an image", res.json()["detail"].lower())

    def test_shortcut_detector_valid_image(self):
        """Runs shortcut diagnostic suite and verifies output schema and diagnostic findings."""
        sample_path = "app/frontend/samples/real_authentic_sample.jpg"
        with open(sample_path, "rb") as f:
            res = self.client.post(
                "/shortcut_detector",
                files={"file": ("sample.jpg", f, "image/jpeg")}
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("shortcut_risk_level", data)
        self.assertIn(data["shortcut_risk_level"], ["LOW", "MODERATE", "ELEVATED"])
        self.assertIn("findings", data)
        self.assertGreaterEqual(len(data["findings"]), 1)
        self.assertIn("tests", data)
        self.assertEqual(len(data["tests"]), 6)
        self.assertIn("disclaimer", data)

        # Check each diagnostic test payload
        test_ids = [t["id"] for t in data["tests"]]
        self.assertIn("baseline", test_ids)
        self.assertIn("background_masked", test_ids)
        self.assertIn("facial_interior", test_ids)
        self.assertIn("border_context_only", test_ids)
        self.assertIn("resolution_normalized", test_ids)
        self.assertIn("jpeg_q70", test_ids)

        for t in data["tests"]:
            self.assertIn("fake_probability", t)
            self.assertIn("delta_from_baseline", t)
            self.assertTrue(t["thumbnail_b64"].startswith("data:image/jpeg;base64,"))

    def test_counterfactual_robustness_invalid_file(self):
        """Rejects non-image files with 400 Bad Request."""
        res = self.client.post(
            "/counterfactual_robustness",
            files={"file": ("notes.pdf", b"%PDF-1.4...", "application/pdf")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("must be an image", res.json()["detail"].lower())

    def test_counterfactual_robustness_valid_image(self):
        """Runs counterfactual perturbation tests and verifies resilience metrics."""
        sample_path = "app/frontend/samples/stylegan2_sample.jpg"
        with open(sample_path, "rb") as f:
            res = self.client.post(
                "/counterfactual_robustness",
                files={"file": ("sample.jpg", f, "image/jpeg")}
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("baseline_prediction", data)
        self.assertIn("resilience_rate_percent", data)
        self.assertGreaterEqual(data["resilience_rate_percent"], 0.0)
        self.assertLessEqual(data["resilience_rate_percent"], 100.0)
        self.assertIn("mean_absolute_delta", data)
        self.assertIn("perturbation_results", data)
        self.assertEqual(len(data["perturbation_results"]), 9)
        self.assertIn("disclaimer", data)

        # Verify perturbation conditions
        pert_ids = [p["id"] for p in data["perturbation_results"]]
        self.assertIn("unperturbed", pert_ids)
        self.assertIn("jpeg_q95", pert_ids)
        self.assertIn("jpeg_q75", pert_ids)
        self.assertIn("jpeg_q40", pert_ids)
        self.assertIn("gaussian_blur", pert_ids)
        self.assertIn("gaussian_noise", pert_ids)
        self.assertIn("rescale_50", pert_ids)
        self.assertIn("grayscale", pert_ids)
        self.assertIn("brightness_plus20", pert_ids)

if __name__ == "__main__":
    unittest.main()
