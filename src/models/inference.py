import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
from src.models.lxfd_model import LXDFDModel
from src.evaluation.calibration import TemperatureScaler
from src.utils.device import get_device

class LXDFDPredictor:
    """
    Clean, reproducible inference API for LX-DFD deepfake detection.
    Supports temperature calibration and configurable uncertainty thresholding.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/lxfd_attention_standard_best.pt",
        calibration_path: str = "checkpoints/calibration_params.json",
        fusion_type: str = "attention",
        device: torch.device = None,
        uncertainty_band: float = 0.15,
        decision_threshold: float = 0.5
    ):
        self.device = device or get_device()
        self.uncertainty_band = uncertainty_band
        self.decision_threshold = decision_threshold

        self.model = LXDFDModel(fusion_type=fusion_type, pretrained=False).to(self.device)
        ckpt_file = Path(checkpoint_path)
        if ckpt_file.exists():
            checkpoint = torch.load(ckpt_file, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[PREDICTOR] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[PREDICTOR] Warning: Checkpoint {checkpoint_path} not found.")

        self.model.eval()

        self.scaler = TemperatureScaler()
        self.calibration_enabled = False
        if calibration_path and Path(calibration_path).exists():
            self.scaler.load(calibration_path)
            self.calibration_enabled = True

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image) -> dict:
        """
        Given a PIL Image, returns structured prediction output dictionary.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError("Input image must be a PIL.Image or path string.")

        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor_img)
            logit = out[0].item() if isinstance(out, tuple) else out.item()

        raw_fake_prob = 1.0 / (1.0 + np.exp(-logit))

        if self.calibration_enabled:
            cal_fake_prob = float(self.scaler.calibrate_probs(np.array([raw_fake_prob]))[0])
        else:
            cal_fake_prob = float(raw_fake_prob)

        real_prob = float(1.0 - cal_fake_prob)
        fake_prob = float(cal_fake_prob)

        # Check uncertainty band
        lower_bound = self.decision_threshold - self.uncertainty_band
        upper_bound = self.decision_threshold + self.uncertainty_band

        is_uncertain = (lower_bound <= fake_prob <= upper_bound)

        if is_uncertain:
            prediction = "UNCERTAIN"
            confidence_level = "LOW"
            confidence = float(max(fake_prob, real_prob))
        elif fake_prob >= self.decision_threshold:
            prediction = "FAKE"
            confidence_level = "HIGH" if fake_prob > 0.8 else "MODERATE"
            confidence = fake_prob
        else:
            prediction = "REAL"
            confidence_level = "HIGH" if real_prob > 0.8 else "MODERATE"
            confidence = real_prob

        return {
            "prediction": prediction,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level,
            "is_uncertain": is_uncertain,
            "calibrated": self.calibration_enabled,
            "temperature": float(self.scaler.temperature),
            "raw_fake_probability": round(float(raw_fake_prob), 4)
        }
