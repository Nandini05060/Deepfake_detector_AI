import torch
import torch.nn as nn
import torch.nn.functional as F
from src.frequency.dct import DiscreteCosineTransform2D

class SpectralDropout2D(nn.Module):
    """
    Phase 3 Experiment B: Spectral Dropout for 2D DCT Log-Magnitude Spectrum.
    Randomly zeroes out random rectangular blocks of the frequency spectrum during training,
    preventing the Frequency CNN from overfitting to brittle high-frequency harmonics.
    """
    def __init__(self, p: float = 0.2, block_size_range: tuple = (16, 48)):
        super().__init__()
        self.p = p
        self.block_size_range = block_size_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        
        B, C, H, W = x.shape
        mask = torch.ones_like(x)
        for b in range(B):
            if torch.rand(1).item() < self.p:
                bh = torch.randint(self.block_size_range[0], self.block_size_range[1] + 1, (1,)).item()
                bw = torch.randint(self.block_size_range[0], self.block_size_range[1] + 1, (1,)).item()
                y0 = torch.randint(0, max(1, H - bh), (1,)).item()
                x0 = torch.randint(0, max(1, W - bw), (1,)).item()
                mask[b, :, y0:y0+bh, x0:x0+bw] = 0.0
                
        return x * mask

class FrequencyBranch(nn.Module):
    """
    Branch B — Frequency Domain Feature Extractor.
    Computes 2D DCT log-magnitude spectrum and extracts frequency artifact embeddings F_f.
    """
    def __init__(
        self,
        in_size: int = 224,
        embed_dim: int = 256,
        dropout: float = 0.3,
        spectral_dropout: bool = False,
        spectral_dropout_p: float = 0.2
    ):
        super().__init__()
        self.dct_transform = DiscreteCosineTransform2D(size=in_size, log_scale=True)
        self.use_spectral_dropout = spectral_dropout
        self.spectral_dropout = SpectralDropout2D(p=spectral_dropout_p) if spectral_dropout else nn.Identity()
        
        # Frequency CNN (Specialized 4-Layer ConvNet for spectral patterns)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), # (112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Input: RGB Tensor (B, 3, H, W)
        Returns: Tuple of (F_f embedding, dct_map spectrum tensor)
        """
        dct_map = self.dct_transform(x) # (B, 1, H, W)
        spectral_in = self.spectral_dropout(dct_map)
        feat = self.features(spectral_in)
        flattened = torch.flatten(feat, 1)
        F_f = self.projection(flattened)
        return F_f, dct_map
