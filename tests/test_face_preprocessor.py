import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.face_preprocessor import YuNetFacePreprocessor

class TestYuNetFacePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = YuNetFacePreprocessor(
            model_path="yunet.onnx",
            target_size=(224, 224),
            margin=0.25,
            score_threshold=0.60
        )

    def test_synthetic_blank_image_fallback(self):
        """A solid color image has no face and should deterministically trigger center-square fallback."""
        blank_img = Image.new("RGB", (400, 600), color=(128, 128, 128))
        out_img, meta = self.preprocessor.process_image(blank_img)

        self.assertEqual(out_img.size, (224, 224))
        self.assertEqual(meta["face_detection_status"], "no_face_detected")
        self.assertEqual(meta["fallback_used"], "center_square_crop")
        self.assertEqual(meta["original_width"], 400)
        self.assertEqual(meta["original_height"], 600)
        # Cropped coords must be a 400x400 square from the center of 400x600
        x1, y1, x2, y2 = meta["crop_coords"]
        self.assertEqual(x2 - x1, 400)
        self.assertEqual(y2 - y1, 400)
        self.assertEqual(y1, 100) # (600 - 400) // 2

    def test_square_image_without_face(self):
        """A square non-face image should trigger square_direct fallback."""
        square_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        out_img, meta = self.preprocessor.process_image(square_img)

        self.assertEqual(out_img.size, (224, 224))
        self.assertEqual(meta["face_detection_status"], "no_face_detected")
        self.assertEqual(meta["fallback_used"], "square_direct")

    def test_real_dataset_image_detection(self):
        """Test on an actual dataset image with a human face."""
        candidates = list(Path("data/unseen_generator_test/real").glob("*.jpg"))
        if not candidates:
            candidates = list(Path("Final Dataset/Real").glob("*.jpg"))
        
        if candidates:
            sample_path = candidates[0]
            out_img, meta = self.preprocessor.process_image(sample_path)
            self.assertEqual(out_img.size, (224, 224))
            self.assertIn(meta["face_detection_status"], ["yunet_detected", "no_face_detected"])
            if meta["face_detection_status"] == "yunet_detected":
                self.assertIsNotNone(meta["bbox_raw"])
                self.assertGreater(meta["confidence_score"], 0.5)

if __name__ == "__main__":
    unittest.main()
