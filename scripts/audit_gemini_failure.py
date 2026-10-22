import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.backend.main import extract_face_crop
from src.models.lxfd_model import LXDFDModel
from src.evaluation.calibration import TemperatureScaler

def audit_gemini_sample():
    print("=" * 80)
    print("           STEP 1 & 2: REPRODUCE AND AUDIT GEMINI FAILURE SAMPLE          ")
    print("=" * 80)

    img_path = Path("data/test_samples/gemini/gemini_sample.png")
    if not img_path.exists():
        raise FileNotFoundError(f"Gemini test sample not found at: {img_path}")

    pil_img = Image.open(img_path)
    orig_w, orig_h = pil_img.size
    print(f"[*] Original Image: {img_path}")
    print(f"    Dimensions: {orig_w} x {orig_h} (Aspect Ratio: {orig_w/orig_h:.3f})")
    print(f"    Mode: {pil_img.mode}, Format: {pil_img.format}")

    # YuNet Face Detection
    orig_np = np.array(pil_img.convert("RGB"))
    h, w, _ = orig_np.shape
    yunet_path = Path("yunet.onnx")
    detector = cv2.FaceDetectorYN.create(
        model=str(yunet_path),
        config="",
        input_size=(w, h),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5
    )
    bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)

    face_bbox = None
    face_score = None
    if faces is not None and len(faces) > 0:
        f = max(faces, key=lambda x: x[2] * x[3])
        face_bbox = {
            "x": float(f[0]),
            "y": float(f[1]),
            "w": float(f[2]),
            "h": float(f[3])
        }
        face_score = float(f[-1])
        print(f"[*] YuNet Face Detected: Bounding Box={face_bbox}, Detection Score={face_score:.4f}")
    else:
        print("[!] No face detected by YuNet!")

    # Face Crop
    face_crop = extract_face_crop(pil_img.convert("RGB"))
    crop_w, crop_h = face_crop.size
    print(f"[*] Extracted Face Crop: {crop_w} x {crop_h}")
    crop_save_path = Path("data/test_samples/gemini/gemini_face_crop.png")
    face_crop.save(crop_save_path)

    # Tensor preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    tensor_img = transform(face_crop).unsqueeze(0)

    # Sanity checks on input tensor
    has_nan_input = torch.isnan(tensor_img).any().item()
    has_inf_input = torch.isinf(tensor_img).any().item()
    print(f"[*] Input Tensor Checks: Shape={list(tensor_img.shape)}, NaN={has_nan_input}, Inf={has_inf_input}")

    # Audit Checkpoints
    checkpoints_to_audit = [
        {"id": "lxfd_standard", "name": "LX-DFD Standard (Current Backend)", "path": "checkpoints/lxfd_attention_standard_best.pt"},
        {"id": "lxfd_robustness_v2", "name": "LX-DFD Robustness v2", "path": "checkpoints/lxfd_attention_robustness_v2_best.pt"},
        {"id": "lxfd_multigen", "name": "LX-DFD Multi-Generator (Phase 8)", "path": "checkpoints/lxfd_multigen_diverse_best.pt"}
    ]

    scaler = TemperatureScaler()
    scaler.load("checkpoints/calibration_params.json")
    T_val = float(scaler.temperature)
    print(f"[*] Temperature Scaler Loaded: T = {T_val:.4f}")

    audit_results = {
        "sample_metadata": {
            "source": "Google Gemini (Imagen 3)",
            "file_path": str(img_path),
            "original_dimensions": [orig_w, orig_h],
            "crop_dimensions": [crop_w, crop_h],
            "detected_face_bbox": face_bbox,
            "face_confidence": face_score
        },
        "models": {}
    }

    for ckpt_info in checkpoints_to_audit:
        m_id = ckpt_info["id"]
        m_name = ckpt_info["name"]
        ckpt_path = ckpt_info["path"]

        model = LXDFDModel(fusion_type="attention", pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        model.eval()

        with torch.no_grad():
            F_s = model.spatial_branch(tensor_img)
            F_f, dct_map = model.frequency_branch(tensor_img)
            fused, meta = model.fusion(F_s, F_f)
            logit = model.classifier(fused).squeeze(-1).item()

            # Separate branch projections
            c_s = model.classifier(F_s).squeeze(-1).item()
            c_f = model.classifier(F_f).squeeze(-1).item()

        raw_fake_prob = float(1.0 / (1.0 + np.exp(-logit)))
        cal_fake_prob = float(scaler.calibrate_probs(np.array([raw_fake_prob]))[0])
        raw_real_prob = float(1.0 - raw_fake_prob)
        cal_real_prob = float(1.0 - cal_fake_prob)

        w_s = float(meta["w_s"])
        w_f = float(meta["w_f"])

        # Check tensor anomalies
        anomalies = {
            "F_s_nan": torch.isnan(F_s).any().item(),
            "F_s_zero": (torch.norm(F_s).item() == 0.0),
            "F_f_nan": torch.isnan(F_f).any().item(),
            "F_f_zero": (torch.norm(F_f).item() == 0.0),
            "dct_nan": torch.isnan(dct_map).any().item(),
            "fused_nan": torch.isnan(fused).any().item()
        }

        print(f"\n--- Model: {m_name} ({ckpt_path}) ---")
        print(f"  Logit:               {logit:8.4f}")
        print(f"  Raw Fake Prob:       {raw_fake_prob * 100:.4f}%")
        print(f"  Calibrated Fake Prob:{cal_fake_prob * 100:.4f}%")
        print(f"  Calibrated Real Prob:{cal_real_prob * 100:.4f}%")
        print(f"  Prediction (tau=0.5):{'FAKE' if cal_fake_prob >= 0.5 else 'AUTHENTIC REAL'}")
        print(f"  Spatial Weight (w_s):{w_s:.6f}")
        print(f"  Freq Weight (w_f):   {w_f:.6f}")
        print(f"  Spatial Only Logit:  {c_s:8.4f} (Prob: {1.0 / (1.0 + np.exp(-c_s)):.6f})")
        print(f"  Freq Only Logit:     {c_f:8.4f} (Prob: {1.0 / (1.0 + np.exp(-c_f)):.6f})")
        print(f"  F_s Norm:            {torch.norm(F_s).item():.4f}")
        print(f"  F_f Norm:            {torch.norm(F_f).item():.4f}")
        print(f"  DCT Map Range:       [{dct_map.min().item():.4f}, {dct_map.max().item():.4f}]")
        print(f"  Tensor Anomalies:    {anomalies}")

        audit_results["models"][m_id] = {
            "name": m_name,
            "checkpoint": ckpt_path,
            "logit": round(logit, 4),
            "raw_fake_probability": round(raw_fake_prob, 6),
            "calibrated_fake_probability": round(cal_fake_prob, 6),
            "calibrated_real_probability": round(cal_real_prob, 6),
            "prediction": "FAKE" if cal_fake_prob >= 0.5 else "AUTHENTIC REAL",
            "attention_weights": {
                "spatial_weight_ws": round(w_s, 6),
                "frequency_weight_wf": round(w_f, 6)
            },
            "branch_independent_logits": {
                "spatial_branch_logit": round(c_s, 4),
                "frequency_branch_logit": round(c_f, 4)
            },
            "tensor_norms": {
                "F_s_norm": round(torch.norm(F_s).item(), 4),
                "F_f_norm": round(torch.norm(F_f).item(), 4),
                "dct_map_mean": round(dct_map.mean().item(), 4),
                "dct_map_std": round(dct_map.std().item(), 4)
            },
            "anomalies": anomalies
        }

    # Save to reports/gemini_failure_metrics.json
    metrics_path = Path("reports/gemini_failure_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[+] Audit metrics saved to: {metrics_path.resolve()}")

if __name__ == "__main__":
    audit_gemini_sample()
