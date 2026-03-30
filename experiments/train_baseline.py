import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import os
import argparse
import segmentation_models_pytorch as smp
import sys
from datetime import datetime

from dataset import MitralValveDataset, get_transforms
from model import get_model
from utils import load_zipped_pickle, seed_everything

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1) # (B, 1, H, W)
        
        optimizer.zero_grad()
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
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            
            # Calculate IoU
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = (preds + masks).sum(dim=(1, 2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.extend(iou.cpu().numpy())
            
    return running_loss / len(loader.dataset), np.mean(iou_scores)

    return running_loss / len(loader.dataset), np.mean(iou_scores)

# Tee output to log file
class Logger(object):
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
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("runs", f"train_log_{timestamp}.txt")
    sys.stdout = Logger(log_file)
    
    print(f"Starting training at {timestamp}")
    print(f"Args: {args}")

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_data = load_zipped_pickle("./data/train.pkl")
    
    # Prepare for Cross-Validation
    # We split by video, not by frame
    video_names = [item['name'] for item in train_data]
    dataset_types = [item['dataset'] for item in train_data] # Stratify by this
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    train_transform, val_transform = get_transforms(img_size=args.img_size)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(video_names, dataset_types)):
        print(f"\nFold {fold+1}/{args.folds}")
        
        train_videos = [train_data[i] for i in train_idx]
        val_videos = [train_data[i] for i in val_idx]
        
        # Filter outliers in training set only
        train_dataset = MitralValveDataset(train_videos, transform=train_transform, mode='train')
        val_dataset = MitralValveDataset(val_videos, transform=val_transform, mode='train')
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        model = get_model(arch=args.arch, encoder_name=args.encoder, classes=1).to(device)
        
        # Loss: Weighted BCE + DiceLoss
        # Calculate positive weight based on class imbalance in this fold (approximate)
        # Or just use a fixed high weight since MV is small.
        # Let's estimate: MV is maybe 1-5% of image? So weight ~20?
        # Let's try dynamic calculation or a fixed parameter.
        pos_weight = torch.tensor([10.0]).to(device) # Start with 10
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        
        def criterion(outputs, masks):
            return 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
            
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_iou = 0.0
        
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_iou = validate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}")
            
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), os.path.join("models", f"best_model_fold{fold}.pth"))
                print("Saved best model!")
                
        print(f"Fold {fold+1} finished. Best IoU: {best_iou:.4f}")
        
        if args.debug:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus', help='Unet, UnetPlusPlus, DeepLabV3Plus')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
