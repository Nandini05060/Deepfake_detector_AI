import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class CustomJPEGCompression:
    """Simulates JPEG compression degradation."""
    def __init__(self, quality_range=(60, 95), p=0.35):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img

class CustomGaussianBlur:
    """Simulates Gaussian blur degradation."""
    def __init__(self, radius_range=(0.5, 1.5), p=0.25):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class CustomGaussianNoise:
    """Adds additive Gaussian noise to image tensor."""
    def __init__(self, std_range=(0.01, 0.04), p=0.30):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

def get_transforms(split: str = 'train', image_size: tuple = (224, 224), mode: str = 'standard'):
    """
    Constructs PyTorch transform pipelines for train, validation, and test splits.
    Supports 'standard', 'robustness', and 'robustness_v2' augmentation levels.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == 'train':
        if mode == 'robustness_v2':
            # Phase 3 Experiment A: Robustness-Aware Augmentation
            transform_list = [
                CustomJPEGCompression(quality_range=(60, 95), p=0.35),
                CustomGaussianBlur(radius_range=(0.5, 1.5), p=0.25),
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                CustomGaussianNoise(std_range=(0.01, 0.04), p=0.30),
                T.Normalize(mean=mean, std=std)
            ]
        elif mode == 'robustness':
            transform_list = [
                CustomJPEGCompression(quality_range=(60, 95), p=0.25),
                CustomGaussianBlur(radius_range=(0.5, 1.5), p=0.15),
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                CustomGaussianNoise(std_range=(0.01, 0.04), p=0.10),
                T.Normalize(mean=mean, std=std)
            ]
        else: # Standard Augmentation
            transform_list = [
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ]
    else: # Val / Test Split (Clean evaluation strictly untouched)
        transform_list = [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ]

    return T.Compose(transform_list)
