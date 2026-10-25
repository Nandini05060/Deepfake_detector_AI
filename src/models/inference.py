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
        decision_threshold: float = 0.5,
        min_floor: float = 0.25
    ):
        self.device = device or get_device()
        self.uncertainty_band = uncertainty_band
        self.decision_threshold = decision_threshold
        self.min_floor = min_floor

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
        self.model = LXDFDModel(fusion_type=fusion_type, min_floor=min_floor, pretrained=False).to(self.device)
        ckpt_file = Path(checkpoint_path)
        if ckpt_file.exists():
            checkpoint = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[PREDICTOR] Loaded checkpoint: {checkpoint_path} with min_floor={min_floor}")
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

        self.model.eval()

        self.scaler = TemperatureScaler()
        self.calibration_enabled = False
        if calibration_path and Path(calibration_path).exists():
            self.scaler.load(calibration_path)
            self.calibration_enabled = True

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, face_meta: dict = None) -> dict:
        """
        Given a PIL Image and optional face_meta dictionary, returns structured prediction output dictionary
        including individual branch probabilities, attention weights, 3-way final decision
        (LIKELY_REAL, LIKELY_FAKE, INCONCLUSIVE), and forensic reliability warnings.
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

        # Non-probability trigger: Face detection check
        face_detected = True
        if face_meta is not None:
            face_detected = face_meta.get("face_detected", True)

        # Empirical reliability assessment & 3-way decision routing
        if not face_detected:
            prediction = "INCONCLUSIVE"
            final_decision = "INCONCLUSIVE"
            confidence_level = "LOW"
            empirical_reliability = "LOW (Face Localization Unverified)"
            confidence = float(max(fake_prob, real_prob))
            is_uncertain = True
        elif in_uncertainty_band:
            prediction = "INCONCLUSIVE"
            final_decision = "INCONCLUSIVE"
            confidence_level = "LOW"
            empirical_reliability = "LOW (Decision Boundary Region: 0.40 - 0.60)"
            confidence = float(max(fake_prob, real_prob))
            is_uncertain = True
        elif high_discrepancy:
            # Check if branches agree on the same side of the decision threshold
            if spatial_prob >= self.decision_threshold and freq_prob >= self.decision_threshold:
                prediction = "FAKE"
                final_decision = "LIKELY_FAKE"
                confidence = fake_prob
                confidence_level = "MODERATE"
                empirical_reliability = "MODERATE (Dual-Branch Positive with Discrepancy)"
                is_uncertain = False
            elif spatial_prob < self.decision_threshold and freq_prob < self.decision_threshold:
                # Both branches lean real, but check for unadapted diffusion signature:
                # Frequency branch detects elevated spectral suspicion while spatial is near-zero
                if freq_prob >= 0.28 and spatial_prob < 0.10:
                    prediction = "INCONCLUSIVE"
                    final_decision = "INCONCLUSIVE"
                    confidence_level = "LOW"
                    empirical_reliability = "LOW (Modality Divergence: Spectral Suspicion with Spatial Blindness)"
                    confidence = float(max(fake_prob, real_prob))
                    is_uncertain = True
                else:
                    prediction = "REAL"
                    final_decision = "LIKELY_REAL"
                    confidence = real_prob
                    confidence_level = "MODERATE"
                    empirical_reliability = "MODERATE (Artifact Absence vs True Authenticity)"
                    is_uncertain = False
            elif fake_prob >= 0.80 and (spatial_prob >= 0.85 or freq_prob >= 0.85):
                prediction = "FAKE"
                final_decision = "LIKELY_FAKE"
                confidence = fake_prob
                confidence_level = "MODERATE"
                empirical_reliability = "MODERATE (Single Modality Dominant Manipulation Evidence)"
                is_uncertain = False
            else:
                # Modality contradiction: one branch predicts fake while the other predicts real without decisive evidence
                prediction = "INCONCLUSIVE"
                final_decision = "INCONCLUSIVE"
                confidence_level = "LOW"
                empirical_reliability = "LOW (Branch Divergence: Spatial vs Frequency Contradiction)"
                confidence = float(max(fake_prob, real_prob))
                is_uncertain = True
        elif fake_prob >= self.decision_threshold:
            prediction = "FAKE"
            final_decision = "LIKELY_FAKE"
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
            final_decision = "LIKELY_REAL"
            confidence = real_prob
            if real_prob > 0.90 and not high_discrepancy:
                confidence_level = "HIGH (In-Distribution)"
                empirical_reliability = "MODERATE (Subject to Generator Scope)"
            else:
                confidence_level = "MODERATE"
                empirical_reliability = "MODERATE (Artifact Absence vs True Authenticity)"
            is_uncertain = False

        # Out-of-Distribution Generator Advisory
        generator_warning = None
        if not face_detected:
            generator_warning = (
                "FACE LOCALIZATION WARNING: No face could be detected with primary confidence (>=0.50). "
                "Prediction is based on full frame context and must be treated as INCONCLUSIVE."
            )
        elif final_decision == "INCONCLUSIVE":
            generator_warning = (
                "INCONCLUSIVE VERDICT: This image exhibits features (decision boundary ambiguity or branch divergence) "
                "that place it outside the model's calibrated decisive operating range. "
                "Novel generators (e.g. Google Gemini Imagen 3, Midjourney v6) or subtle inpainting may produce ambiguous forensic signals."
            )
        elif final_decision == "LIKELY_REAL":
            generator_warning = (
                "Scope Advisory: Validated on StyleGAN2, ProGAN, Stable Diffusion v1.5, and InsightFace. "
                "Novel synthesis models may omit legacy artifacts. A LIKELY_REAL result denotes absence of known signatures, not guaranteed authenticity."
            )
        elif high_discrepancy:
            generator_warning = (
                "Branch Divergence Warning: Spatial and frequency branches exhibit substantial disagreement (>10%). "
                "The sample may be corrupted, heavily compressed, or out-of-distribution."
            )

        return {
            "prediction": prediction,
            "final_decision": final_decision,
            "verdict": final_decision,
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
            "generator_warning": generator_warning,
            "forensic_disclaimer": "Forensic Notice: LX-DFD prediction is an automated statistical model estimate, not definitive legal or biometric proof of authenticity or manipulation."
        }
