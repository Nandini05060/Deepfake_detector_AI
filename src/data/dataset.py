import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.data.augmentation import get_transforms

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for binary deepfake detection (REAL=0, FAKE=1).
    Supports loading from split CSVs (train.csv, val.csv, test.csv).
    """
    def __init__(
        self,
        csv_path: str,
        image_size: tuple = (224, 224),
        split: str = 'train',
        aug_mode: str = 'standard'
    ):
        self.csv_path = Path(csv_path)
        self.image_size = image_size
        self.split = split
        self.aug_mode = aug_mode
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV file not found: {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        
        # Label mapping: REAL -> 0.0, FAKE -> 1.0
        self.label_map = {'REAL': 0.0, 'FAKE': 1.0}
        
        self.transform = get_transforms(split=self.split, image_size=self.image_size, mode=self.aug_mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if 'path' in row and pd.notna(row['path']):
            img_path = str(row['path'])
        elif 'file_path' in row and pd.notna(row['file_path']):
            img_path = str(row['file_path'])
        elif 'processed_path' in row and pd.notna(row['processed_path']):
            img_path = str(row['processed_path'])
        elif 'relative_path' in row and pd.notna(row['relative_path']):
            img_path = str(row['relative_path'])
        elif 'filename' in row and pd.notna(row['filename']):
            img_path = str(Path(self.csv_path).parent.parent / "processed" / row['filename'])
        else:
            img_path = ""

        label_str = row['label'] if 'label' in row else row.get('class', 'REAL')
        label_val = self.label_map.get(str(label_str).upper(), 0.0)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for missing/corrupted images
            image = Image.new('RGB', self.image_size, color=(0, 0, 0))
            
        tensor_img = self.transform(image)
        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        
        sample = {
            'image': tensor_img,
            'label': label_tensor,
            'path': img_path,
            'source': row.get('source', 'unknown'),
            'generator': row.get('generator', 'unknown')
        }
        return sample
