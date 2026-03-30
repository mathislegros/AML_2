"""
Training Script - Teammate's Exact Setup
- ResNet34 encoder + UnetPlusPlus
- NO CLAHE preprocessing
- Simple augmentation (matching dataset.py)
- BCE + Jaccard loss
- All data with expert weighting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np
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


class AllDataDataset(Dataset):
    """Dataset using ALL data with expert weighting - NO CLAHE"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for item in self.data:
            video = item['video']
            label = item['label']
            frames = item['frames']
            dataset = item.get('dataset', 'amateur')

            for frame_idx in frames:
                self.samples.append({
                    'image': video[:, :, frame_idx],
                    'mask': label[:, :, frame_idx],
                    'dataset': dataset
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].copy()
        mask = sample['mask'].astype(np.float32)
        is_expert = 1.0 if sample['dataset'] == 'expert' else 0.0

        # NO CLAHE - just convert to RGB like teammate's dataset.py
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, torch.tensor(is_expert, dtype=torch.float32)


class ExpertOnlyDataset(Dataset):
    """Dataset for validation - expert only, NO CLAHE"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.samples = []
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

        # NO CLAHE
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def get_transforms(img_size=256):
    """EXACT transforms from teammate's dataset.py"""
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


class BCEJaccardLoss(nn.Module):
    """BCE + Jaccard loss with sample weighting"""
    def __init__(self, bce_weight=0.25, jaccard_weight=0.75):
        super().__init__()
        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.jaccard = smp.losses.JaccardLoss(mode='binary', from_logits=True)

    def forward(self, pred, target, sample_weights=None):
        bce_loss = self.bce(pred, target)

        if sample_weights is not None:
            weights = sample_weights.view(-1, 1, 1, 1)
            bce_loss = (bce_loss * weights).mean()
        else:
            bce_loss = bce_loss.mean()

        jaccard_loss = self.jaccard(pred, target)

        return self.bce_weight * bce_loss + self.jaccard_weight * jaccard_loss


def get_model(encoder_name='resnet34'):
    """Get model - matching teammate's setup"""
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type=None  # NO attention - like teammate
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, expert_weight=2.0, scaler=None):
    model.train()
    running_loss = 0.0

    for images, masks, is_expert in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        is_expert = is_expert.to(device)

        # Expert samples get higher weight
        sample_weights = torch.where(is_expert > 0.5,
                                     torch.tensor(expert_weight, device=device),
                                     torch.tensor(1.0, device=device))

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks, sample_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks, sample_weights)
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

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = (preds + masks).sum(dim=(1, 2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.extend(iou.cpu().numpy())

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

    log_file = os.path.join("runs", f"teammate_style_{timestamp}.txt")
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print("TEAMMATE STYLE TRAINING (NO CLAHE, Simple Augmentation)")
    print("=" * 80)
    print(f"\nTimestamp: {timestamp}")
    print(f"Args: {args}")

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load ALL data
    print("\nLoading ALL data (expert + amateur)...")
    train_data = load_zipped_pickle("data/train.pkl")
    expert_data = [d for d in train_data if d.get('dataset') == 'expert']
    amateur_data = [d for d in train_data if d.get('dataset') == 'amateur']

    print(f"Expert videos: {len(expert_data)} ({sum(len(d['frames']) for d in expert_data)} frames)")
    print(f"Amateur videos: {len(amateur_data)} ({sum(len(d['frames']) for d in amateur_data)} frames)")
    print(f"Total: {len(train_data)} videos")

    train_transform, val_transform = get_transforms(args.img_size)

    # K-Fold on EXPERT data only (validation matches test distribution)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_fold_ious = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(expert_data)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"{'='*60}")

        # Training: ALL amateur + train split of expert
        train_expert = [expert_data[i] for i in train_idx]
        val_expert = [expert_data[i] for i in val_idx]

        train_videos = amateur_data + train_expert
        print(f"Train: {len(train_videos)} videos ({len(amateur_data)} amateur + {len(train_expert)} expert)")
        print(f"Val: {len(val_expert)} expert videos (matches test distribution)")

        train_dataset = AllDataDataset(train_videos, transform=train_transform)
        val_dataset = ExpertOnlyDataset(val_expert, transform=val_transform)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)

        # Model - ResNet34 + UnetPlusPlus (teammate's setup)
        model = get_model(encoder_name=args.encoder).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: UnetPlusPlus with {args.encoder}")
        print(f"Parameters: {num_params:,}")

        criterion = BCEJaccardLoss()
        print(f"Loss: 0.25 * BCE + 0.75 * Jaccard")
        print(f"Expert weight: {args.expert_weight}x")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

        best_iou = 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                         device, args.expert_weight, scaler)
            val_iou, val_dice = validate(model, val_loader, device, threshold=args.threshold)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")

            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                model_path = os.path.join("models", f"teammate_style_fold{fold}.pth")
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--expert_weight', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
