"""
Automated Regression Test Suite for Gemini Failure Diagnostic Image
File: data/diagnostics/gemini_failure.png
Image: Gemini_Generated_Image_qykw28qykw28qykw.png
SHA-256: 944cb50e9187a22a46dc6232387aa6769d4513c4654b141832b1eec04cd377fa
"""

import sys
import hashlib
from pathlib import Path
from PIL import Image
import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.inference import LXDFDPredictor
from app.backend.main import extract_face_crop_with_meta
from src.utils.device import get_device

DIAG_PATH = repo_root / "data/diagnostics/gemini_failure.png"
EXPECTED_SHA256 = "944cb50e9187a22a46dc6232387aa6769d4513c4654b141832b1eec04cd377fa"

def test_diagnostic_image_integrity():
    assert DIAG_PATH.exists(), f"Diagnostic image not found at: {DIAG_PATH}"
    actual_hash = hashlib.sha256(DIAG_PATH.read_bytes()).hexdigest()
    assert actual_hash == EXPECTED_SHA256, f"Hash mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
    
    im = Image.open(DIAG_PATH)
    assert im.size == (2816, 1536), f"Unexpected size: {im.size}"
    assert im.format == "PNG", f"Unexpected format: {im.format}"

def test_reproduce_phase15_exact_metrics():
    p15_ckpt = repo_root / "experiments/phase15_gemini_generalization/lxfd_phase15_generalization_best.pt"
    p15_cal = repo_root / "experiments/phase15_gemini_generalization/calibration_params_phase15.json"
    
    device = get_device()
    predictor = LXDFDPredictor(str(p15_ckpt), str(p15_cal), device=device, min_floor=0.25)
    
    im = Image.open(DIAG_PATH).convert("RGB")
    face_img, meta = extract_face_crop_with_meta(im, margin=0.50)
    
    assert meta["face_detected"] is True
    res = predictor.predict(face_img, face_meta=meta)
    
    # Verify exact values reported in user screenshot:
    # Fake: ~43.5%, Real: ~56.5%, Spatial: ~43.6%, Freq: ~51.0%
    fake_p = res["fake_probability"] * 100
    real_p = res["real_probability"] * 100
    spatial_p = res["branch_outputs"]["spatial_branch_fake_prob"] * 100
    freq_p = res["branch_outputs"]["frequency_branch_fake_prob"] * 100
    
    assert 42.0 <= fake_p <= 45.0, f"Expected fake_p ~43.5%, got {fake_p:.2f}%"
    assert 55.0 <= real_p <= 58.0, f"Expected real_p ~56.5%, got {real_p:.2f}%"
    assert 42.0 <= spatial_p <= 45.0, f"Expected spatial_p ~43.6%, got {spatial_p:.2f}%"
    assert 49.0 <= freq_p <= 53.0, f"Expected freq_p ~51.0%, got {freq_p:.2f}%"
    assert res["final_decision"] == "INCONCLUSIVE", f"Expected INCONCLUSIVE, got {res['final_decision']}"

def test_phase16_unified_safe_routing():
    p16_ckpt = repo_root / "experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt"
    p16_cal = repo_root / "experiments/phase16_unified_generalization/calibration_params_phase16.json"
    
    device = get_device()
    predictor = LXDFDPredictor(str(p16_ckpt), str(p16_cal), device=device, min_floor=0.25)
    
    im = Image.open(DIAG_PATH).convert("RGB")
    face_img, meta = extract_face_crop_with_meta(im, margin=0.50)
    
    res = predictor.predict(face_img, face_meta=meta)
    # The system must not assert confident REAL on an AI generated face
    assert res["final_decision"] == "INCONCLUSIVE"
    assert res["is_uncertain"] is True
    assert abs(res["fake_probability"] + res["real_probability"] - 1.0) < 1e-4

def test_crop_context_sensitivity():
    p16_ckpt = repo_root / "experiments/phase16_unified_generalization/lxfd_phase16_unified_best.pt"
    p16_cal = repo_root / "experiments/phase16_unified_generalization/calibration_params_phase16.json"
    
    device = get_device()
    predictor = LXDFDPredictor(str(p16_ckpt), str(p16_cal), device=device, min_floor=0.25)
    
    im = Image.open(DIAG_PATH).convert("RGB")
    # Extended context includes diffusion boundary and hair
    face_img_75, meta_75 = extract_face_crop_with_meta(im, margin=0.75)
    res_75 = predictor.predict(face_img_75, face_meta=meta_75)
    
    # On extended margin, fake probability must be high
    assert res_75["fake_probability"] > 0.70, f"Expected high fake probability on extended crop, got {res_75['fake_probability']}"

if __name__ == "__main__":
    print("Running automated regression tests for gemini_failure.png...")
    test_diagnostic_image_integrity()
    print("  [PASS] test_diagnostic_image_integrity")
    test_reproduce_phase15_exact_metrics()
    print("  [PASS] test_reproduce_phase15_exact_metrics")
    test_phase16_unified_safe_routing()
    print("  [PASS] test_phase16_unified_safe_routing")
    test_crop_context_sensitivity()
    print("  [PASS] test_crop_context_sensitivity")
    print("All regression tests passed successfully!")
