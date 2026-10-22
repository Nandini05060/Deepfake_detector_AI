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
    Supports temperature calibration, branch-level explainability,
    and empirical reliability/uncertainty thresholding.
    """
    def __init__(
        self,
        checkpoint_path: str = None,
        calibration_path: str = "checkpoints/calibration_params.json",
        fusion_type: str = "attention",
        device: torch.device = None,
        uncertainty_band: float = 0.15,
        decision_threshold: float = 0.5
    ):
        self.device = device or get_device()
        self.uncertainty_band = uncertainty_band
        self.decision_threshold = decision_threshold

        # Default to the most robust and generalizable checkpoint
        if checkpoint_path is None:
            candidate_paths = [
                Path("checkpoints/lxfd_multigen_diverse_best.pt"),
                Path("checkpoints/lxfd_attention_robustness_v2_best.pt"),
                Path("checkpoints/lxfd_attention_standard_best.pt")
            ]
            checkpoint_path = "checkpoints/lxfd_multigen_diverse_best.pt"
            for cp in candidate_paths:
                if cp.exists():
                    checkpoint_path = str(cp)
                    break

        self.checkpoint_path = checkpoint_path
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
        Given a PIL Image, returns structured prediction output dictionary
        including individual branch probabilities, attention weights,
        and empirical reliability warnings.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError("Input image must be a PIL.Image or path string.")

        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out, meta = self.model(tensor_img)
            logit = out[0].item() if out.dim() > 0 else out.item()

            # Extract branch-independent predictions
            spatial_prob = meta['spatial_fake_prob'].item() if 'spatial_fake_prob' in meta else 0.5
            freq_prob = meta['frequency_fake_prob'].item() if 'frequency_fake_prob' in meta else 0.5
            w_s = meta.get('w_s', 0.5)
            w_f = meta.get('w_f', 0.5)

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
        in_uncertainty_band = (lower_bound <= fake_prob <= upper_bound)

        # Branch discrepancy check (empirical disagreement)
        branch_discrepancy = abs(spatial_prob - freq_prob)
        high_discrepancy = (branch_discrepancy > 0.10)

        # Empirical reliability assessment
        # Do not label HIGH confidence solely because sigmoid is near 0 or 1 on unvalidated external inputs!
        if in_uncertainty_band:
            prediction = "UNCERTAIN"
            confidence_level = "LOW"
            empirical_reliability = "LOW (Decision Boundary Region)"
            confidence = float(max(fake_prob, real_prob))
            is_uncertain = True
        elif fake_prob >= self.decision_threshold:
            prediction = "FAKE"
            confidence = fake_prob
            if fake_prob > 0.85 and not high_discrepancy:
                confidence_level = "HIGH"
                empirical_reliability = "HIGH (Strong Dual-Branch Agreement)"
            else:
                confidence_level = "MODERATE"
                empirical_reliability = "MODERATE (Branch Discrepancy Observed)"
            is_uncertain = False
        else:
            prediction = "REAL"
            confidence = real_prob
            # For REAL predictions, if spatial says 99.9% real but frequency suggests possible artifacts,
            # or if the image may be from an unobserved generator family, qualify the confidence:
            if real_prob > 0.90 and not high_discrepancy:
                confidence_level = "HIGH (In-Distribution)"
                empirical_reliability = "MODERATE (Subject to Generator Scope)"
            else:
                confidence_level = "MODERATE"
                empirical_reliability = "MODERATE (Artifact Absence vs True Authenticity)"
            is_uncertain = high_discrepancy

        # Out-of-Distribution Generator Advisory
        generator_warning = None
        if prediction == "REAL":
            generator_warning = (
                "Scope Advisory: Validated on StyleGAN2, ProGAN, Stable Diffusion v1.5, and InsightFace. "
                "Novel or unvalidated synthesis models (e.g., Google Gemini Imagen 3, Midjourney v6, FLUX.1) "
                "may omit legacy artifacts. A REAL result denotes absence of known signatures, not guaranteed authenticity."
            )
        elif high_discrepancy:
            generator_warning = (
                "Branch Divergence Warning: Spatial and frequency branches exhibit substantial disagreement (>10%). "
                "The sample may be corrupted, heavily compressed, or out-of-distribution."
            )

        return {
            "prediction": prediction,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level,
            "empirical_reliability": empirical_reliability,
            "is_uncertain": is_uncertain,
            "calibrated": self.calibration_enabled,
            "temperature": float(self.scaler.temperature),
            "raw_fake_probability": round(float(raw_fake_prob), 4),
            "checkpoint_used": Path(self.checkpoint_path).name,
            "branch_outputs": {
                "spatial_branch_fake_prob": round(float(spatial_prob), 4),
                "frequency_branch_fake_prob": round(float(freq_prob), 4),
                "spatial_attention_weight_ws": round(float(w_s), 4),
                "frequency_attention_weight_wf": round(float(w_f), 4),
                "branch_discrepancy": round(float(branch_discrepancy), 4)
            },
            "generator_warning": generator_warning
        }
