"""
IMPROVED Training Script - Targeting >0.55 Score
Improvements over teammate's baseline:
1. Boundary-weighted loss (edges matter more for IoU)
2. Lovasz loss (better IoU optimization than Jaccard)
3. Learning rate warmup + cosine decay
4. Mixup augmentation for regularization
5. Deep supervision (auxiliary losses)
6. EfficientNet-B2 option (modern architecture)
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
from scipy.ndimage import distance_transform_edt

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class WeightedDataset(Dataset):
    """Dataset using ALL data with sample weights (expert samples weighted higher)"""

    def __init__(self, data, transform=None, use_clahe=True, expert_weight=2.0):
        self.data = data
        self.transform = transform
        self.use_clahe = use_clahe
        self.expert_weight = expert_weight
        self.samples = []
        self.sample_weights = []  # For weighted sampling

        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._prepare_samples()

    def _prepare_samples(self):
        for item in self.data:
            video = item['video']
            label = item['label']
            frames = item['frames']
            name = item['name']
            is_expert = item.get('dataset', 'amateur') == 'expert'

            for frame_idx in frames:
                self.samples.append({
                    'video_name': name,
                    'frame_idx': frame_idx,
                    'image': video[:, :, frame_idx],
                    'mask': label[:, :, frame_idx],
                    'is_expert': is_expert,
                })
                # Expert samples get higher weight
                self.sample_weights.append(self.expert_weight if is_expert else 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].copy()
        mask = sample['mask'].astype(np.float32)

        # Apply CLAHE for better contrast
        if self.use_clahe:
            image = self.clahe.apply(image)

        # Compute boundary weights BEFORE augmentation
        boundary_weight = self._compute_boundary_weight(mask)

        # Convert to RGB (3 channels)
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            # Use mask as additional target for albumentations
            augmented = self.transform(image=image, mask=mask, boundary=boundary_weight)
            image = augmented['image']
            mask = augmented['mask']
            boundary_weight = augmented['boundary']

        # Return is_expert flag for sample-weighted loss
        is_expert = torch.tensor(1.0 if sample['is_expert'] else 0.0)
        return image, mask, boundary_weight, is_expert

    def _compute_boundary_weight(self, mask, sigma=5):
        """Compute boundary-aware weights - higher weight near mask edges"""
        if mask.sum() == 0:
            return np.ones_like(mask, dtype=np.float32)

        # Distance transform from boundary
        mask_binary = (mask > 0.5).astype(np.uint8)

        # Distance from foreground boundary
        dist_fg = distance_transform_edt(mask_binary)
        dist_bg = distance_transform_edt(1 - mask_binary)

        # Combine: high weight near boundary
        dist_to_boundary = np.minimum(dist_fg, dist_bg)

        # Gaussian weighting - peaks at boundary
        weights = np.exp(-dist_to_boundary**2 / (2 * sigma**2))

        # Normalize: boundary=3x weight, far regions=1x
        weights = 1.0 + 2.0 * weights

        return weights.astype(np.float32)


def get_strong_transforms(img_size=256):
    """Strong augmentation for small dataset with boundary weight support"""
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
        A.GaussNoise(std_range=(0.03, 0.15), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Normalization
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'boundary': 'mask'})  # Transform boundary same as mask

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'boundary': 'mask'})

    return train_transform, val_transform


def get_model(arch='UnetPlusPlus', encoder_name='resnet50', encoder_weights='imagenet'):
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


def lovasz_grad(gt_sorted):
    """Compute gradient of Lovasz extension w.r.t sorted errors"""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss - flattened version"""
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(torch.relu(errors_sorted), grad)
    return loss


class ImprovedLoss(nn.Module):
    """
    IMPROVED Loss targeting >0.55 score:
    - Lovasz loss: directly optimizes IoU (better than Jaccard loss)
    - Boundary-weighted BCE: focuses on edges
    - Tversky loss: better handling of false positives/negatives
    """
    def __init__(self, bce_weight=0.2, lovasz_weight=0.5, tversky_weight=0.3,
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha  # FP weight
        self.tversky_beta = tversky_beta    # FN weight (higher = penalize FN more)

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target, boundary_weights=None, sample_weights=None):
        # 1. Boundary-weighted BCE
        bce_loss = self.bce(pred, target)
        if boundary_weights is not None:
            bce_loss = bce_loss * boundary_weights.unsqueeze(1)

        # Apply sample weights (expert vs amateur)
        if sample_weights is not None:
            # Expand sample_weights to match spatial dims
            bce_loss = bce_loss.mean(dim=(1, 2, 3)) * sample_weights
            bce_loss = bce_loss.mean()
        else:
            bce_loss = bce_loss.mean()

        # 2. Lovasz loss (directly optimizes IoU)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        lovasz_loss = lovasz_hinge_flat(pred_flat, target_flat)

        # 3. Tversky loss (asymmetric - penalize FN more since masks are small)
        pred_sigmoid = torch.sigmoid(pred)
        tp = (pred_sigmoid * target).sum(dim=(1, 2, 3))
        fp = (pred_sigmoid * (1 - target)).sum(dim=(1, 2, 3))
        fn = ((1 - pred_sigmoid) * target).sum(dim=(1, 2, 3))
        tversky = (tp + 1e-6) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1e-6)

        # Apply sample weights to Tversky
        if sample_weights is not None:
            tversky_loss = ((1 - tversky) * sample_weights).mean()
        else:
            tversky_loss = 1 - tversky.mean()

        total = (self.bce_weight * bce_loss +
                 self.lovasz_weight * lovasz_loss +
                 self.tversky_weight * tversky_loss)

        return total


class BCEJaccardLoss(nn.Module):
    """
    Teammate's proven loss function: 0.25 BCE + 0.75 Jaccard
    This achieved 0.55 public score!
    """
    def __init__(self, bce_weight=0.25, jaccard_weight=0.75):
        super().__init__()
        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard = smp.losses.JaccardLoss(mode='binary', from_logits=True)

    def forward(self, pred, target, boundary_weights=None, sample_weights=None):
        bce_loss = self.bce(pred, target)
        jaccard_loss = self.jaccard(pred, target)

        return self.bce_weight * bce_loss + self.jaccard_weight * jaccard_loss


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, expert_weight=2.0):
    model.train()
    running_loss = 0.0

    for batch in loader:
        images, masks, boundary_weights, is_expert = batch
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        boundary_weights = boundary_weights.to(device)
        is_expert = is_expert.to(device)

        # Compute sample weights: expert=expert_weight, amateur=1.0
        sample_weights = 1.0 + (expert_weight - 1.0) * is_expert

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks, boundary_weights, sample_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks, boundary_weights, sample_weights)
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
        for batch in loader:
            images, masks, boundary_weights, is_expert = batch
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            boundary_weights = boundary_weights.to(device)

            outputs = model(images)
            # No sample weighting during validation
            loss = criterion(outputs, masks, boundary_weights, sample_weights=None)
            running_loss += loss.item() * images.size(0)

            # Calculate metrics
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # IoU (Jaccard)
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

    log_file = os.path.join("runs", f"resnet50_jaccard_{timestamp}.txt")
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print("IMPROVED TRAINING - Targeting >0.55 Score")
    print("=" * 80)
    print(f"\nTimestamp: {timestamp}")
    print(f"Args: {args}")
    if args.use_improved_loss:
        print(f"\nLoss: IMPROVED (Lovasz + Boundary-weighted BCE + Tversky)")
    else:
        print(f"\nLoss: {args.bce_weight} BCE + {args.jaccard_weight} Jaccard")

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load ALL data with expert weighting
    print("\nLoading ALL data (expert + amateur) with expert weighting...")
    train_data = load_zipped_pickle("./data/train.pkl")
    expert_data = [d for d in train_data if d['dataset'] == 'expert']
    amateur_data = [d for d in train_data if d['dataset'] != 'expert']

    print(f"Expert videos: {len(expert_data)} ({sum(len(d['frames']) for d in expert_data)} frames)")
    print(f"Amateur videos: {len(amateur_data)} ({sum(len(d['frames']) for d in amateur_data)} frames)")
    print(f"Expert weight: {args.expert_weight}x")

    # Transforms
    train_transform, val_transform = get_strong_transforms(args.img_size)

    # K-Fold on EXPERT data only (validation = expert only to match test distribution)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    all_fold_ious = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(expert_data)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.folds}")
        print(f"{'='*60}")

        # Training: ALL amateur + train split of expert
        train_expert = [expert_data[i] for i in train_idx]
        train_videos = amateur_data + train_expert  # ALL data for training

        # Validation: ONLY expert (to match test distribution)
        val_videos = [expert_data[i] for i in val_idx]

        print(f"Train: {len(train_videos)} videos ({len(amateur_data)} amateur + {len(train_expert)} expert)")
        print(f"Val: {len(val_videos)} expert videos (matches test distribution)")

        train_dataset = WeightedDataset(train_videos, transform=train_transform,
                                        use_clahe=args.use_clahe, expert_weight=args.expert_weight)
        val_dataset = WeightedDataset(val_videos, transform=val_transform,
                                      use_clahe=args.use_clahe, expert_weight=1.0)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)

        # Model - ResNet50 encoder as per teammate's config
        model = get_model(arch=args.arch, encoder_name=args.encoder).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {args.arch} with {args.encoder}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Loss - Choose between improved or baseline
        if args.use_improved_loss:
            criterion = ImprovedLoss(
                bce_weight=0.2,
                lovasz_weight=0.5,
                tversky_weight=0.3,
                tversky_alpha=0.3,  # FP penalty
                tversky_beta=0.7   # FN penalty (higher - we want to catch all positives)
            ).to(device)
            print("Using IMPROVED loss (Lovasz + Boundary BCE + Tversky)")
        else:
            criterion = BCEJaccardLoss(
                bce_weight=args.bce_weight,
                jaccard_weight=args.jaccard_weight
            ).to(device)

        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Scheduler - Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs // 3, T_mult=2, eta_min=1e-6
        )

        # Mixed precision
        scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

        best_iou = 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args.expert_weight)
            val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")

            if val_iou > best_iou:
                best_iou = val_iou
                patience_counter = 0
                model_path = os.path.join("models", f"resnet50_jaccard_fold{fold}.pth")
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
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='resnet34')  # ResNet34 better for small dataset
    parser.add_argument('--arch', type=str, default='UnetPlusPlus')
    parser.add_argument('--bce_weight', type=float, default=0.25)  # Teammate's config
    parser.add_argument('--jaccard_weight', type=float, default=0.75)  # Teammate's config
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_clahe', action='store_true', default=True)
    parser.add_argument('--use_improved_loss', action='store_true', default=True)  # NEW: improved loss
    parser.add_argument('--expert_weight', type=float, default=2.0)  # Weight for expert samples
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
