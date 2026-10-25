import unittest
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.backend.main import app

class TestCropStabilityEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_invalid_file_type(self):
        """Rejects non-image files with 400 Bad Request."""
        res = self.client.post(
            "/crop_stability",
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("must be an image", res.json()["detail"].lower())

    def test_missing_face_fallback_handling(self):
        """A solid color image with no face should gracefully fall back without crashing."""
        blank_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        buf = io.BytesIO()
        blank_img.save(buf, format="JPEG")
        buf.seek(0)

        res = self.client.post(
            "/crop_stability",
            files={"file": ("blank.jpg", buf, "image/jpeg")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["crops"]), 3)
        self.assertIn("stability_classification", data)
        self.assertIn("scientific_warning", data)
        # Verify fallback detection flag
        for c in data["crops"]:
            self.assertFalse(c["face_detected"])

    def test_valid_image_stability_metrics(self):
        """Tests that a sample image returns valid delta and stability classification."""
        sample_path = "app/frontend/samples/real_authentic_sample.jpg"
        with open(sample_path, "rb") as f:
            res = self.client.post(
                "/crop_stability",
                files={"file": ("sample.jpg", f, "image/jpeg")}
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["crops"]), 3)
        self.assertIn(data["stability_classification"], [
            "Low prediction variation",
            "Moderate prediction variation",
            "High prediction variation"
        ])
        self.assertGreaterEqual(data["probability_delta"], 0.0)
        self.assertLessEqual(data["probability_delta"], 1.0)
        self.assertIn("scientific_warning", data)

if __name__ == "__main__":
    unittest.main()
