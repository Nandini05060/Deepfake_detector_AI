"""
LXDFD-MultiScale-12: Multi-Scale Spatial-Frequency Deepfake Detection Architecture
Phase 12 Candidate Model.

Combines:
- Path A (Facial Detail, 224x224): Local texture, fine boundaries, pore microstructures.
- Path B (Global Structure, 112x112): Facial symmetry, macro-illumination, structural consistency.
- Frequency Branch (2D DCT, 224x224): Orthogonal frequency spectra for localized inpainting/face-swap seams.
- Constrained Multimodal Fusion: Balances all branches without modality starvation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from src.models.spatial_branch import SpatialBranch
from src.models.frequency_branch import FrequencyBranch

class MultiScaleSpatialFusion(nn.Module):
    """
    Combines Detail Spatial (224x224), Global Spatial (112x112), and Frequency (2D DCT)
    embeddings with configurable fusion strategies:
    - 'detail_only': Spatial 224 only
    - 'global_only': Spatial 112 only
    - 'concat': Concatenate all branches
    - 'equal_weight': 0.5 * Detail + 0.5 * Global
    - 'multiscale_frequency': Weighted combination of Detail, Global, and Frequency
    - 'learned_constrained': Learnable softmax weights bounded by minimum floor tau
    """
    def __init__(self, embed_dim: int = 256, fusion_type: str = 'multiscale_frequency', min_floor: float = 0.20):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.min_floor = min_floor
        
        if fusion_type in ('learned_constrained', 'multiscale_frequency'):
            # Computes attention across 3 modalities: Detail (s_d), Global (s_g), Frequency (f)
            self.attn_net = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 3)
            )

    def forward(
        self,
        F_detail: torch.Tensor,
        F_global: torch.Tensor,
        F_freq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, D = F_detail.shape
        
        if self.fusion_type == 'detail_only':
            return F_detail, {'w_detail': 1.0, 'w_global': 0.0, 'w_freq': 0.0}
            
        elif self.fusion_type == 'global_only':
            return F_global, {'w_detail': 0.0, 'w_global': 1.0, 'w_freq': 0.0}
            
        elif self.fusion_type == 'concat':
            if F_freq is not None:
                fused = torch.cat([F_detail, F_global, F_freq], dim=1) # (B, 3D)
            else:
                fused = torch.cat([F_detail, F_global], dim=1) # (B, 2D)
            return fused, {'w_detail': 0.3333, 'w_global': 0.3333, 'w_freq': 0.3333}
            
        elif self.fusion_type == 'equal_weight':
            fused = 0.5 * F_detail + 0.5 * F_global
            return fused, {'w_detail': 0.5, 'w_global': 0.5, 'w_freq': 0.0}
            
        elif self.fusion_type in ('learned_constrained', 'multiscale_frequency'):
            if F_freq is None:
                F_freq = torch.zeros_like(F_detail)
                
            combined = torch.cat([F_detail, F_global, F_freq], dim=1) # (B, 3D)
            raw_logits = self.attn_net(combined) # (B, 3)
            raw_w = F.softmax(raw_logits, dim=1) # (B, 3)
            
            # Constrained floor enforcement: w_i = floor + (1 - 3*floor) * raw_w_i
            # Prevents any modality from dropping below min_floor (e.g. 0.20)
            floor = self.min_floor
            bounded_w = floor + (1.0 - 3.0 * floor) * raw_w
            
            w_d = bounded_w[:, 0:1]
            w_g = bounded_w[:, 1:2]
            w_f = bounded_w[:, 2:3]
            
            fused = w_d * F_detail + w_g * F_global + w_f * F_freq
            
            return fused, {
                'w_detail': float(w_d.mean().item()),
                'w_global': float(w_g.mean().item()),
                'w_freq': float(w_f.mean().item())
            }
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

class LXDFDMultiScale12(nn.Module):
    """
    Phase 12 Candidate: Multi-Scale Spatial-Frequency Model.
    Architecture:
    - Path A (224x224): EfficientNet-B0 (F_detail in R^256)
    - Path B (112x112): Shared or dedicated EfficientNet-B0 (F_global in R^256)
    - Frequency Branch (224x224): 2D DCT ConvNet (F_freq in R^256)
    - Multimodal Fusion Head: Constrained fusion module
    - Classification Head: Linear(256, 128) -> GELU -> Dropout(0.3) -> Linear(128, 1)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        fusion_type: str = 'multiscale_frequency',
        shared_spatial_weights: bool = True,
        min_floor: float = 0.20,
        pretrained: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.shared_spatial_weights = shared_spatial_weights
        
        # Path A: Facial Detail (224x224)
        self.spatial_detail = SpatialBranch(embed_dim=embed_dim, pretrained=pretrained, dropout=dropout)
        
        # Path B: Global Structure (112x112)
        if shared_spatial_weights:
            self.spatial_global = self.spatial_detail
        else:
            self.spatial_global = SpatialBranch(embed_dim=embed_dim, pretrained=pretrained, dropout=dropout)
            
        # Frequency Branch: 2D DCT Log-Magnitude Spectrum
        self.frequency_branch = FrequencyBranch(in_size=224, embed_dim=embed_dim, dropout=dropout)
        
        # Multi-scale fusion
        self.fusion = MultiScaleSpatialFusion(embed_dim=embed_dim, fusion_type=fusion_type, min_floor=min_floor)
        
        in_dim = embed_dim * 3 if fusion_type == 'concat' else embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_224: torch.Tensor, x_112: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Inputs:
            x_224: Tensor of shape (B, 3, 224, 224)
            x_112: Optional tensor of shape (B, 3, 112, 112). If None, bilinear downsampled from x_224.
        Returns:
            logit: Tensor of shape (B,)
            meta: Dictionary of intermediate branch outputs, probabilities, and attention weights.
        """
        if x_112 is None:
            x_112 = F.interpolate(x_224, size=(112, 112), mode='bilinear', align_corners=False)
            
        F_detail = self.spatial_detail(x_224)
        F_global = self.spatial_global(x_112)
        F_freq, dct_map = self.frequency_branch(x_224)
        
        fused, w_dict = self.fusion(F_detail, F_global, F_freq)
        logit = self.classifier(fused).squeeze(-1)
        
        # Independent branch logits for explainability & uncertainty detection
        logit_detail = self.classifier(F_detail).squeeze(-1)
        logit_global = self.classifier(F_global).squeeze(-1)
        logit_freq = self.classifier(F_freq).squeeze(-1)
        
        meta = {
            'w_detail': w_dict['w_detail'],
            'w_global': w_dict['w_global'],
            'w_freq': w_dict['w_freq'],
            'F_detail': F_detail,
            'F_global': F_global,
            'F_freq': F_freq,
            'dct_map': dct_map,
            'detail_logit': logit_detail,
            'global_logit': logit_global,
            'freq_logit': logit_freq,
            'spatial_detail_prob': torch.sigmoid(logit_detail),
            'spatial_global_prob': torch.sigmoid(logit_global),
            'frequency_prob': torch.sigmoid(logit_freq)
        }
        return logit, meta
