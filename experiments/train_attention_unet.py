"""
Improved Training Script with Architecture Upgrades
- Attention U-Net with decoder attention (SCSE)
- EfficientNet-B4 encoder (more capacity)
- Expert-only training (matches test distribution)
- BCE + Jaccard loss (proven)
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


class ExpertDataset(Dataset):
    """Dataset using ONLY expert data - matches test distribution"""

    def __init__(self, data, transform=None, use_clahe=True):
        self.data = data
        self.transform = transform
        self.use_clahe = use_clahe
        self.samples = []
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._prepare_samples()

    def _prepare_samples(self):
        for item in self.data:
            video = item['video']
            label = item['label']
            frames = item['frames']

            for frame_idx in frames:
                self.samples.append({
                    'image': video[:, :, frame_idx],
                    'mask': label[:, :, frame_idx]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].copy()
        mask = sample['mask'].astype(np.float32)

        if self.use_clahe:
            image = self.clahe.apply(image)

        # Convert to RGB
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def get_transforms(img_size=256):
    """Moderate augmentation - not too aggressive"""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform


class BCEJaccardLoss(nn.Module):
    """BCE + Jaccard loss - proven effective"""
    def __init__(self, bce_weight=0.25, jaccard_weight=0.75):
        super().__init__()
        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard = smp.losses.JaccardLoss(mode='binary', from_logits=True)

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.jaccard_weight * self.jaccard(pred, target)


def get_model(arch='UnetPlusPlus', encoder_name='efficientnet-b4', decoder_attention='scse'):
    """
    Get model with attention mechanisms
    - decoder_attention: 'scse' (channel + spatial attention)
    """
    if arch == 'Unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type=decoder_attention
        )
    elif arch == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type=decoder_attention
        )
    elif arch == 'MAnet':
        # MAnet has built-in attention
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    elif arch == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


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


def validate(model, loader, device, threshold=0.5):
    model.eval()
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > threshold).float()

            # IoU
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = (preds + masks).sum(dim=(1, 2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.extend(iou.cpu().numpy())

            # Dice
            dice = (2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
            dice_scores.extend(dice.cpu().numpy())

    return np.mean(iou_scores), np.mean(dice_scores)


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("runs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_file = os.path.join("runs", f"attention_unet_{timestamp}.txt")
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print("ATTENTION U-NET TRAINING (Architecture Upgrade)")
    print("=" * 80)
    print(f"\nTimestamp: {timestamp}")
    print(f"Args: {args}")

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load EXPERT-ONLY data
    print("\nLoading EXPERT-ONLY data (matches test distribution)...")
    train_data = load_zipped_pickle("./data/train.pkl")
    expert_data = [d for d in train_data if d.get('dataset') == 'expert']
    print(f"Expert videos: {len(expert_data)} ({sum(len(d['frames']) for d in expert_data)} frames)")

    train_transform, val_transform = get_transforms(args.img_size)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_fold_ious = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(expert_data)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"{'='*60}")

        train_videos = [expert_data[i] for i in train_idx]
        val_videos = [expert_data[i] for i in val_idx]

        print(f"Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")

        train_dataset = ExpertDataset(train_videos, transform=train_transform, use_clahe=args.use_clahe)
        val_dataset = ExpertDataset(val_videos, transform=val_transform, use_clahe=args.use_clahe)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)

        # Model with attention
        model = get_model(
            arch=args.arch,
            encoder_name=args.encoder,
            decoder_attention=args.decoder_attention
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {args.arch} with {args.encoder}")
        print(f"Decoder attention: {args.decoder_attention}")
        print(f"Parameters: {num_params:,}")

        criterion = BCEJaccardLoss()
        print(f"Loss: 0.25 * BCE + 0.75 * Jaccard")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

        best_iou = 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_iou, val_dice = validate(model, val_loader, device, threshold=args.threshold)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")

            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                model_path = os.path.join("models", f"attention_unet_fold{fold}.pth")
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
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='Encoder: efficientnet-b4, efficientnet-b5, resnet50')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus',
                        help='Architecture: Unet, UnetPlusPlus, MAnet')
    parser.add_argument('--decoder_attention', type=str, default='scse',
                        help='Decoder attention: scse, None')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_clahe', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
