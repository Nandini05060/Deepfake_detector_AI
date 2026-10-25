import unittest
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from app.backend.main import app
from src.models.fusion import SpatialFrequencyFusion
from src.models.lxfd_model import LXDFDModel
from src.models.lxfd_multiscale import LXDFDMultiScale12, MultiScaleSpatialFusion


class TestPhase12MultiScaleSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.multiscale_ckpt_path = Path("experiments/phase12_multiscale_v1/lxfd_multiscale12_best.pt")
        cls.expected_multiscale_sha256 = "a2b9989205f255248ca7c378e25164b88fbc47691ce8b9b9b815fcd00a72e773"
        cls.phase10_ckpt_path = Path("checkpoints/lxfd_phase10_modern_generators_best.pt")
        cls.expected_phase10_sha256 = "13dc56b8961f8d0c2c56a9ac947ef4292f4a2a5ef613df3aeb950f087fddcc05"

    # 1. MultiScale Model Instantiation & Forward Pass
    def test_multiscale_model_forward_pass_single_input(self):
        """Verifies LXDFDMultiScale12 accepts standard 224x224 tensor, handles dual-resolution internally."""
        model = LXDFDMultiScale12(embed_dim=256, pretrained=False)
        model.eval()
        dummy_224 = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logit, meta = model(dummy_224)
            
        self.assertEqual(logit.shape, (2,))
        self.assertIn("w_detail", meta)
        self.assertIn("w_global", meta)
        self.assertIn("w_freq", meta)
        self.assertIn("dct_map", meta)
        # Weights should sum to 1.0 (within float tolerance)
        total_weight = meta["w_detail"] + meta["w_global"] + meta["w_freq"]
        self.assertAlmostEqual(total_weight, 1.0, places=4)

    def test_multiscale_model_forward_pass_explicit_dual_inputs(self):
        """Verifies LXDFDMultiScale12 accepts explicit x_224 and x_112 inputs."""
        model = LXDFDMultiScale12(embed_dim=256, pretrained=False)
        model.eval()
        dummy_224 = torch.randn(1, 3, 224, 224)
        dummy_112 = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            logit, meta = model(dummy_224, dummy_112)
            
        self.assertEqual(logit.shape, (1,))
        self.assertIsInstance(logit.item(), float)

    # 2. Checkpoint Hash Authentication
    def test_candidate_checkpoint_sha256_hash(self):
        """Authenticates SHA-256 cryptographic hash of LXDFD-MultiScale-12 candidate checkpoint."""
        self.assertTrue(self.multiscale_ckpt_path.exists(), f"Candidate checkpoint missing at {self.multiscale_ckpt_path}")
        with open(self.multiscale_ckpt_path, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            computed_hash,
            self.expected_multiscale_sha256,
            f"MultiScale checkpoint hash mismatch! Expected {self.expected_multiscale_sha256}, got {computed_hash}"
        )

    def test_candidate_checkpoint_loads_strictly(self):
        """Verifies saved candidate state dictionary loads cleanly with strict=True."""
        model = LXDFDMultiScale12(embed_dim=256, pretrained=False)
        ckpt = torch.load(self.multiscale_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out, meta = model(dummy)
        self.assertEqual(out.shape, (1,))

    # 3. Constrained Attention Minimum Floor Enforcement
    def test_spatial_frequency_fusion_min_floor_enforcement(self):
        """Verifies that SpatialFrequencyFusion enforces min_floor (e.g. 0.25) across extreme inputs."""
        fusion = SpatialFrequencyFusion(embed_dim=256, fusion_type="attention", min_floor=0.25)
        # Create extreme adversarial inputs to force attention logits towards 0 or 1
        f_s = torch.ones(4, 256) * 100.0
        f_f = torch.zeros(4, 256)
        fused, weights = fusion(f_s, f_f)
        
        # When min_floor=0.25, neither w_s nor w_f can fall below 0.25 or exceed 0.75
        self.assertGreaterEqual(weights["w_s"], 0.2499)
        self.assertLessEqual(weights["w_s"], 0.7501)
        self.assertGreaterEqual(weights["w_f"], 0.2499)
        self.assertLessEqual(weights["w_f"], 0.7501)
        self.assertAlmostEqual(weights["w_s"] + weights["w_f"], 1.0, places=4)

    def test_lxfd_model_accepts_configurable_min_floor(self):
        """Verifies LXDFDModel accepts configurable min_floor parameter."""
        model = LXDFDModel(pretrained=False, min_floor=0.25)
        self.assertEqual(model.min_floor, 0.25)
        self.assertEqual(model.fusion.min_floor, 0.25)

    # 4. MultiScale Fusion Module Behavior
    def test_multiscale_spatial_fusion_modes(self):
        """Tests detail_only, global_only, equal_weight, and learned attention modes."""
        f_d = torch.randn(2, 256)
        f_g = torch.randn(2, 256)
        f_f = torch.randn(2, 256)
        
        fusion_detail = MultiScaleSpatialFusion(embed_dim=256, fusion_type="detail_only")
        out_d, w_d = fusion_detail(f_d, f_g, f_f)
        self.assertEqual(w_d["w_detail"], 1.0)
        self.assertEqual(w_d["w_global"], 0.0)
        
        fusion_global = MultiScaleSpatialFusion(embed_dim=256, fusion_type="global_only")
        out_g, w_g = fusion_global(f_d, f_g, f_f)
        self.assertEqual(w_g["w_global"], 1.0)
        self.assertEqual(w_g["w_detail"], 0.0)
        
        fusion_equal = MultiScaleSpatialFusion(embed_dim=256, fusion_type="equal_weight")
        out_e, w_e = fusion_equal(f_d, f_g, f_f)
        self.assertAlmostEqual(w_e["w_detail"], 0.5, places=3)
        self.assertAlmostEqual(w_e["w_global"], 0.5, places=3)
        self.assertEqual(w_e["w_freq"], 0.0)

    # 5. API Endpoint Parity & Registration
    def test_api_models_endpoint_registers_multiscale_candidate(self):
        """Verifies /models endpoint returns lxfd_multiscale12 candidate model."""
        res = self.client.get("/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        model_ids = [m["id"] for m in data.get("models", [])]
        self.assertIn("lxfd_attention", model_ids)
        self.assertIn("lxfd_multiscale12", model_ids)
        
        multiscale_info = next(m for m in data["models"] if m["id"] == "lxfd_multiscale12")
        self.assertEqual(multiscale_info["type"], "research_candidate")
        self.assertFalse(multiscale_info["is_active"])

    def test_api_research_phase12_metrics_endpoint(self):
        """Verifies /research/phase12/metrics returns valid Phase 12 benchmark data."""
        res = self.client.get("/research/phase12/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("comparison_benchmark", data)
        self.assertEqual(len(data["comparison_benchmark"]), 4)
        
        # Verify baseline entry
        baseline = data["comparison_benchmark"][0]
        self.assertIn("overall_metrics", baseline)
        self.assertEqual(baseline["gemini_performance"]["fraction"], "2/17")

    def test_api_reports_catalog_includes_phase12_reports(self):
        """Verifies /reports catalog contains Phase 12 report entries."""
        res = self.client.get("/reports")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        report_ids = [r["id"] for r in data.get("reports", [])]
        self.assertIn("REP-PHASE12-AUDIT", report_ids)
        self.assertIn("REP-PHASE12-RESOLUTION", report_ids)
        self.assertIn("REP-PHASE12-MULTISCALE", report_ids)
        self.assertIn("REP-PHASE12-FREQUENCY", report_ids)
        self.assertIn("REP-PHASE12-ROBUSTNESS", report_ids)
        self.assertIn("REP-PHASE12-UNCERTAINTY", report_ids)
        self.assertIn("REP-PHASE12-FINAL-COMPARE", report_ids)

    def test_api_phase12_report_download(self):
        """Verifies that Phase 12 markdown report is downloadable via /reports/download."""
        res = self.client.get("/reports/download/phase12_final_model_comparison.md")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["filename"], "phase12_final_model_comparison.md")
        self.assertIn("Model Comparison & Selection", payload["content"])


if __name__ == "__main__":
    unittest.main()
