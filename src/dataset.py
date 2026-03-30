import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_zipped_pickle
import cv2

class MitralValveDataset(Dataset):
    def __init__(self, data, transform=None, mode='train'):
        """
        Args:
            data: List of dictionaries containing video data.
            transform: Albumentations transform.
            mode: 'train' or 'test'.
        """
        self.data = data
        self.transform = transform
        self.mode = mode
        self.samples = []
        
        if self.mode == 'train':
            self._prepare_train_samples()
        else:
            # For test/inference, we might want to iterate differently, 
            # but for now let's assume we might want to validate on full videos 
            # or just load frames.
            # For simplicity in this dataset class, let's stick to frame-based loading.
            # If we need full video inference, we might handle it in the inference loop 
            # or a separate dataset mode.
            pass

    def _prepare_train_samples(self):
        for item in self.data:
            video = item['video'] # (H, W, T)
            label = item['label'] # (H, W, T)
            frames = item['frames'] # List of indices
            name = item['name']
            dataset_type = item['dataset']
            
            for frame_idx in frames:
                self.samples.append({
                    'video_name': name,
                    'frame_idx': frame_idx,
                    'image': video[:, :, frame_idx],
                    'mask': label[:, :, frame_idx],
                    'dataset': dataset_type
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        mask = sample['mask']
        
        # Image is uint8 (H, W). Mask is bool (H, W).
        # Convert mask to float32 for transform
        mask = mask.astype(np.float32)
        
        # Albumentations expects RGB images usually, but we have grayscale.
        # We can replicate channels or keep it grayscale. 
        # ResNet expects 3 channels usually. Let's replicate.
        image = np.stack([image, image, image], axis=-1) # (H, W, 3)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default to tensor conversion if no transform provided
            transform = A.Compose([ToTensorV2()])
            augmented = transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask

def get_transforms(img_size=256):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform
