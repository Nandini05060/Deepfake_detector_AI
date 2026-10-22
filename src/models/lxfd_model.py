import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.spatial_branch import SpatialBranch
from src.models.frequency_branch import FrequencyBranch
from src.models.fusion import SpatialFrequencyFusion

class SpatialOnlyModel(nn.Module):
    """
    Experiment 4: Spatial-Only Deepfake Detection Sub-Network.
    Extracts RGB spatial embeddings F_s via EfficientNet and maps to a binary logit.
    """
    def __init__(self, embed_dim: int = 256, pretrained: bool = True, dropout: float = 0.3, num_classes: int = 1):
        super().__init__()
        self.spatial_branch = SpatialBranch(embed_dim=embed_dim, pretrained=pretrained, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_s = self.spatial_branch(x)
        logit = self.classifier(F_s).squeeze(-1)
        return logit

class FrequencyOnlyModel(nn.Module):
    """
    Experiment 5: Frequency-Only Deepfake Detection Sub-Network.
    Computes 2D DCT Log-Magnitude Spectrum, extracts spectral embeddings F_f via 4-Stage ConvNet,
    and maps to a binary logit.
    """
    def __init__(self, in_size: int = 224, embed_dim: int = 256, dropout: float = 0.3, num_classes: int = 1):
        super().__init__()
        self.frequency_branch = FrequencyBranch(in_size=in_size, embed_dim=embed_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_f, _ = self.frequency_branch(x)
        logit = self.classifier(F_f).squeeze(-1)
        return logit

class LXDFDModel(nn.Module):
    """
    LX-DFD — Dual-Branch Spatial-Frequency Deepfake Detection Architecture.
    Combines Spatial RGB Features (Branch A) and Frequency 2D DCT Features (Branch B)
    via Attention / Weighted / Concatenation Fusion for Generalizable Deepfake Detection.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        fusion_type: str = 'attention',
        pretrained: bool = True,
        dropout: float = 0.3,
        spectral_dropout: bool = False,
        spectral_dropout_p: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        
        self.spatial_branch = SpatialBranch(embed_dim=embed_dim, pretrained=pretrained, dropout=dropout)
        self.frequency_branch = FrequencyBranch(
            in_size=224,
            embed_dim=embed_dim,
            dropout=dropout,
            spectral_dropout=spectral_dropout,
            spectral_dropout_p=spectral_dropout_p
        )
        self.fusion = SpatialFrequencyFusion(embed_dim=embed_dim, fusion_type=fusion_type)
        
        # Classifier Head
        in_classifier_dim = embed_dim * 2 if fusion_type == 'concat' else embed_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(in_classifier_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes) # Binary Logit
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Input: Image tensor (B, 3, H, W)
        Returns:
            logit (B,)
            dict containing ('w_s', 'w_f', 'dct_map', 'F_s', 'F_f')
        """
        F_s = self.spatial_branch(x) # (B, D)
        F_f, dct_map = self.frequency_branch(x) # (B, D), (B, 1, H, W)
        
        fused, weights_dict = self.fusion(F_s, F_f) # (B, D)
        logit = self.classifier(fused).squeeze(-1) # (B,)
        
        # Branch-independent projections for explainability and uncertainty detection
        spatial_logit = self.classifier(F_s).squeeze(-1) # (B,)
        frequency_logit = self.classifier(F_f).squeeze(-1) # (B,)
        
        meta = {
            'w_s': weights_dict['w_s'],
            'w_f': weights_dict['w_f'],
            'dct_map': dct_map,
            'F_s': F_s,
            'F_f': F_f,
            'spatial_logit': spatial_logit,
            'frequency_logit': frequency_logit,
            'spatial_fake_prob': torch.sigmoid(spatial_logit),
            'frequency_fake_prob': torch.sigmoid(frequency_logit)
        }
        return logit, meta

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes fake probability = sigmoid(logit)."""
        logit, _ = self.forward(x)
        return torch.sigmoid(logit)
