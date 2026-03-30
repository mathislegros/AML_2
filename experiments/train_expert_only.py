"""
Optimized Training Script for Mitral Valve Segmentation
Based on deep analysis findings:
- Train ONLY on expert data (matches test distribution)
- Use higher pos_weight due to small mask ratio (~0.67%)
- CLAHE preprocessing for better contrast
- Strong augmentation for small dataset (19 videos, 57 frames)
- Leave-one-out cross-validation for expert data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import os
import argparse
import segmentation_models_pytorch as smp
import sys
from datetime import datetime
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gzip
import pickle

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class ExpertOnlyDataset(Dataset):
    """Dataset that only uses expert-labeled data with enhanced preprocessing"""

    def __init__(self, data, transform=None, use_clahe=True, use_box_crop=False):
        self.data = data
        self.transform = transform
        self.use_clahe = use_clahe
        self.use_box_crop = use_box_crop
        self.samples = []

        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._prepare_samples()

    def _prepare_samples(self):
        for item in self.data:
            video = item['video']
            label = item['label']
            frames = item['frames']
            name = item['name']
            box = item.get('box', None)

            for frame_idx in frames:
                self.samples.append({
                    'video_name': name,
                    'frame_idx': frame_idx,
                    'image': video[:, :, frame_idx],
                    'mask': label[:, :, frame_idx],
                    'box': box
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].copy()
        mask = sample['mask'].astype(np.float32)

        # Apply CLAHE for better contrast
        if self.use_clahe:
            image = self.clahe.apply(image)

        # Optionally crop to bounding box region
        if self.use_box_crop and sample['box'] is not None:
            box = sample['box']
            y_coords, x_coords = np.where(box)
            if len(y_coords) > 0:
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                # Add padding
                pad = 20
                H, W = image.shape
                y_min = max(0, y_min - pad)
                y_max = min(H, y_max + pad)
                x_min = max(0, x_min - pad)
                x_max = min(W, x_max + pad)
                image = image[y_min:y_max, x_min:x_max]
                mask = mask[y_min:y_max, x_min:x_max]

        # Convert to RGB (3 channels)
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def get_strong_transforms(img_size=256):
    """Strong augmentation for small dataset"""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_REFLECT),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.GridDistortion(p=0.3),
        # Intensity transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Normalization
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform


def get_model(arch='UnetPlusPlus', encoder_name='resnet34', encoder_weights='imagenet'):
    if arch == 'Unet':
        model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                         in_channels=3, classes=1, activation=None)
    elif arch == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                  in_channels=3, classes=1, activation=None)
    elif arch == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                   in_channels=3, classes=1, activation=None)
    elif arch == 'MAnet':
        model = smp.MAnet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                          in_channels=3, classes=1, activation=None)
    elif arch == 'FPN':
        model = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights,
                        in_channels=3, classes=1, activation=None)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


class CombinedLoss(nn.Module):
    """BCE + Dice + Focal Loss combination"""
    def __init__(self, pos_weight=50.0, bce_weight=0.3, dice_weight=0.4, focal_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss + self.focal_weight * focal_loss


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Calculate metrics
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # IoU
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = (preds + masks).sum(dim=(1, 2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.extend(iou.cpu().numpy())

            # Dice
            dice = (2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
            dice_scores.extend(dice.cpu().numpy())

    return running_loss / len(loader.dataset), np.mean(iou_scores), np.mean(dice_scores)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(args):
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("runs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_file = os.path.join("runs", f"expert_train_{timestamp}.txt")
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print("EXPERT-ONLY TRAINING FOR MITRAL VALVE SEGMENTATION")
    print("=" * 80)
    print(f"\nTimestamp: {timestamp}")
    print(f"Args: {args}")

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data - EXPERT ONLY
    print("\nLoading expert data only...")
    train_data = load_zipped_pickle("./data/train.pkl")
    expert_data = [d for d in train_data if d['dataset'] == 'expert']
    print(f"Expert videos: {len(expert_data)}")
    print(f"Total expert frames: {sum(len(d['frames']) for d in expert_data)}")

    # Transforms
    train_transform, val_transform = get_strong_transforms(args.img_size)

    # K-Fold on expert data
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    all_fold_ious = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(expert_data)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"{'='*60}")

        train_videos = [expert_data[i] for i in train_idx]
        val_videos = [expert_data[i] for i in val_idx]

        print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

        train_dataset = ExpertOnlyDataset(train_videos, transform=train_transform,
                                          use_clahe=args.use_clahe)
        val_dataset = ExpertOnlyDataset(val_videos, transform=val_transform,
                                        use_clahe=args.use_clahe)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)

        # Model
        model = get_model(arch=args.arch, encoder_name=args.encoder).to(device)

        # Loss with high pos_weight for class imbalance
        criterion = CombinedLoss(pos_weight=args.pos_weight).to(device)

        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs // 3, T_mult=2, eta_min=1e-6
        )

        # Mixed precision
        scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

        best_iou = 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")

            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                model_path = os.path.join("models", f"expert_model_fold{fold}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"  -> Saved best model! (IoU: {best_iou:.4f})")
            else:
                patience_counter += 1

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nFold {fold+1} Best IoU: {best_iou:.4f}")
        all_fold_ious.append(best_iou)

        if args.debug:
            break

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Fold IoUs: {all_fold_ious}")
    print(f"Mean IoU: {np.mean(all_fold_ious):.4f} +/- {np.std(all_fold_ious):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='efficientnet-b3')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus')
    parser.add_argument('--pos_weight', type=float, default=50.0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_clahe', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
