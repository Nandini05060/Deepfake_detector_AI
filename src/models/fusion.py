import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFrequencyFusion(nn.Module):
    """
    Implements and compares spatial-frequency feature fusion strategies:
    - spatial_only
    - frequency_only
    - concat (Simple Concatenation)
    - weighted (Learnable Static Scalar Fusion: w_s + w_f = 1)
    - attention (Adaptive Dynamic Attention Fusion: w_s + w_f = 1 per sample)
    """
    def __init__(self, embed_dim: int = 256, fusion_type: str = 'attention', min_floor: float = 0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type.lower()
        self.min_floor = min_floor
        
        if self.fusion_type == 'weighted':
            # Raw logit parameters for w_s and w_f
            self.weight_logits = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
            
        elif self.fusion_type == 'attention':
            # Lightweight attention block to compute dynamic sample-wise weights
            self.attention_net = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 2, 2) # Outputs 2 attention weights (w_s, w_f)
            )

    def forward(self, F_s: torch.Tensor, F_f: torch.Tensor) -> tuple:
        """
        Input: F_s (B, D), F_f (B, D)
        Output: Tuple of (F_fused, weights_dict)
        """
        B, D = F_s.shape
        
        if self.fusion_type == 'spatial_only':
            return F_s, {'w_s': 1.0, 'w_f': 0.0}
            
        elif self.fusion_type == 'frequency_only':
            return F_f, {'w_s': 0.0, 'w_f': 1.0}
            
        elif self.fusion_type == 'concat':
            fused = torch.cat([F_s, F_f], dim=1) # (B, 2D)
            return fused, {'w_s': 0.5, 'w_f': 0.5}
            
        elif self.fusion_type == 'weighted':
            weights = F.softmax(self.weight_logits, dim=0) # [w_s, w_f]
            w_s, w_f = weights[0], weights[1]
            fused = w_s * F_s + w_f * F_f # (B, D)
            return fused, {'w_s': w_s.item(), 'w_f': w_f.item()}
            
        elif self.fusion_type == 'attention':
            combined = torch.cat([F_s, F_f], dim=1) # (B, 2D)
            attn_logits = self.attention_net(combined) # (B, 2)
            raw_weights = F.softmax(attn_logits, dim=1) # (B, 2) -> [w_s, w_f] per sample
            
            # Prevent complete modality starvation by enforcing a minimum floor per branch
            # w_bounded = min_floor + (1 - 2*min_floor) * raw_weights
            attn_weights = self.min_floor + (1.0 - 2.0 * self.min_floor) * raw_weights
            
            w_s = attn_weights[:, 0:1] # (B, 1)
            w_f = attn_weights[:, 1:2] # (B, 1)
            
            fused = w_s * F_s + w_f * F_f # (B, D)
            return fused, {'w_s': w_s.mean().item(), 'w_f': w_f.mean().item()}
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
