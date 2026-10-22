import cv2
import numpy as np
import torch
import torch.nn.functional as F

class SimpleGradCAM:
    """
    Grad-CAM implementation for Spatial CNN feature maps.
    Generates class activation heatmaps highlighting regions contributing strongly to the model's prediction.
    """
    def __init__(self, model: torch.nn.Module, target_layer_name: str = "features"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        
        self.gradients = None
        self.activations = None
        
        # Locate target layer
        self.target_module = None
        
        # Look for spatial_branch.features or features module
        if hasattr(model, 'spatial_branch') and hasattr(model.spatial_branch, 'features'):
            self.target_module = model.spatial_branch.features[-1]
        elif hasattr(model, 'features'):
            self.target_module = model.features[-1]
        else:
            # Fallback to last conv module in model
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.target_module = module
                    
        if self.target_module is not None:
            self.target_module.register_forward_hook(self._save_activation)
            self.target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a single image tensor (1, 3, H, W).
        If target_class == 1 (default), highlights regions contributing to FAKE score.
        If target_class == 0, highlights regions contributing to REAL score.
        If target_class == -1 ('auto'), highlights regions contributing to the predicted class.
        Returns normalized heatmap uint8 array (H, W) in range [0, 255].
        """
        self.model.zero_grad()
        
        # Forward pass
        out = self.model(input_tensor)
        logit = out[0] if isinstance(out, tuple) else out
        
        if logit.dim() > 1:
            logit = logit.squeeze()
            
        # Target score for backpropagation
        if target_class == 1:
            score = logit
        elif target_class == 0:
            score = -logit
        else: # auto
            score = logit if logit.item() >= 0 else -logit
            
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            # Fallback dummy heatmap if hooks did not fire
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.uint8)
            
        gradients = self.gradients # (1, C, h, w)
        activations = self.activations # (1, C, h, w)
        
        # Global average pooling of gradients per channel
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (1, 1, h, w)
        cam = F.relu(cam) # Apply ReLU to keep positive contributions
        
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Min-Max Normalization to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        heatmap_uint8 = (cam * 255.0).astype(np.uint8)
        return heatmap_uint8

def overlay_heatmap_on_image(image_rgb_uint8: np.ndarray, heatmap_uint8: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap onto original RGB image using JET colormap.
    """
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image_rgb_uint8, 1.0 - alpha, color_heatmap_rgb, alpha, 0)
    return overlay
