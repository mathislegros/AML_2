"""
Advanced Inference with:
- Test Time Augmentation (TTA)
- Model Ensemble
- Temporal smoothing
- Morphological post-processing
- CLAHE preprocessing (matching training)
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import cv2
import glob
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gzip
import pickle
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

def get_model(arch='UnetPlusPlus', encoder_name='resnet34'):
    if arch == 'Unet':
        model = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                         in_channels=3, classes=1, activation=None)
    elif arch == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None,
                                  in_channels=3, classes=1, activation=None)
    elif arch == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None,
                                   in_channels=3, classes=1, activation=None)
    elif arch == 'MAnet':
        model = smp.MAnet(encoder_name=encoder_name, encoder_weights=None,
                          in_channels=3, classes=1, activation=None)
    elif arch == 'FPN':
        model = smp.FPN(encoder_name=encoder_name, encoder_weights=None,
                        in_channels=3, classes=1, activation=None)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def get_sequences(arr):
    """RLE encoding"""
    first_indices, last_indices = [], []
    arr = np.array(arr, dtype=np.int32)
    arr = [0] + list(arr) + [0]
    for index, value in enumerate(arr[:-1]):
        if arr[index+1] - arr[index] == 1:
            first_indices.append(index)
        if arr[index+1] - arr[index] == -1:
            last_indices.append(index)
    lengths = list(np.array(last_indices) - np.array(first_indices))
    return first_indices, lengths


def apply_tta(model, frame_tensor, device):
    """Apply Test Time Augmentation and average predictions"""
    preds = []

    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(frame_tensor)).cpu().numpy()
    preds.append(pred)

    # Horizontal flip
    frame_flip = torch.flip(frame_tensor, dims=[3])
    with torch.no_grad():
        pred = torch.sigmoid(model(frame_flip)).cpu().numpy()
    pred = np.flip(pred, axis=3)
    preds.append(pred)

    # Vertical flip
    frame_flip = torch.flip(frame_tensor, dims=[2])
    with torch.no_grad():
        pred = torch.sigmoid(model(frame_flip)).cpu().numpy()
    pred = np.flip(pred, axis=2)
    preds.append(pred)

    # Both flips
    frame_flip = torch.flip(frame_tensor, dims=[2, 3])
    with torch.no_grad():
        pred = torch.sigmoid(model(frame_flip)).cpu().numpy()
    pred = np.flip(pred, axis=(2, 3))
    preds.append(pred)

    # Average all predictions
    return np.mean(preds, axis=0)


def post_process_mask(mask, min_size=100):
    """Apply morphological post-processing"""
    mask = mask.astype(np.uint8)

    # Fill holes
    mask = binary_fill_holes(mask).astype(np.uint8)

    # Remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean_mask[labels == i] = 1

    # Slight dilation followed by erosion (closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return clean_mask


def temporal_smooth(video_mask, kernel_size=3):
    """Apply temporal smoothing across frames"""
    T = video_mask.shape[2]
    smoothed = np.zeros_like(video_mask)

    half_k = kernel_size // 2

    for t in range(T):
        start = max(0, t - half_k)
        end = min(T, t + half_k + 1)
        # Average across temporal window
        avg = np.mean(video_mask[:, :, start:end], axis=2)
        smoothed[:, :, t] = (avg > 0.5).astype(bool)

    return smoothed


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    model_paths = sorted(glob.glob(args.model_pattern))
    if not model_paths:
        raise ValueError(f"No models found matching pattern: {args.model_pattern}")

    print(f"Found {len(model_paths)} models for ensemble")

    models = []
    for path in model_paths:
        print(f"Loading: {path}")
        model = get_model(arch=args.arch, encoder_name=args.encoder)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        models.append(model)

    # Load test data
    print("Loading test data...")
    test_data = load_zipped_pickle("./data/test.pkl")
    print(f"Test videos: {len(test_data)}")

    # CLAHE for preprocessing (same as training)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Transform
    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    ids = []
    rle_values = []

    print("Running inference...")
    for item in tqdm(test_data):
        video = item['video']
        name = item['name']
        H, W, T = video.shape

        video_probs = np.zeros((H, W, T), dtype=np.float32)

        for t in range(T):
            frame = video[:, :, t]

            # Apply CLAHE
            if args.use_clahe:
                frame = clahe.apply(frame)

            # Resize and prepare
            frame_resized = cv2.resize(frame, (args.img_size, args.img_size))
            frame_rgb = np.stack([frame_resized] * 3, axis=-1)
            augmented = transform(image=frame_rgb)
            frame_tensor = augmented['image'].unsqueeze(0).to(device)

            # Ensemble prediction
            frame_preds = []
            for model in models:
                if args.tta:
                    pred = apply_tta(model, frame_tensor, device)
                else:
                    with torch.no_grad():
                        pred = torch.sigmoid(model(frame_tensor)).cpu().numpy()
                frame_preds.append(pred)

            # Average ensemble
            ensemble_pred = np.mean(frame_preds, axis=0).squeeze()

            # Resize back to original size
            pred_original = cv2.resize(ensemble_pred, (W, H), interpolation=cv2.INTER_LINEAR)
            video_probs[:, :, t] = pred_original

        # Apply threshold
        video_mask = (video_probs > args.threshold).astype(bool)

        # Temporal smoothing
        if args.temporal_smooth:
            video_mask = temporal_smooth(video_mask, kernel_size=args.temporal_kernel)

        # Post-processing
        if args.post_process:
            for t in range(T):
                video_mask[:, :, t] = post_process_mask(video_mask[:, :, t], min_size=args.min_size)

        # RLE encoding
        flat_mask = video_mask.flatten()
        starts, lengths = get_sequences(flat_mask)

        for i, (start, length) in enumerate(zip(starts, lengths)):
            ids.append(f"{name}_{i}")
            rle_values.append([int(start), int(length)])

    # Save submission
    formatted_values = [str(v) for v in rle_values]
    df = pd.DataFrame({"id": ids, "value": formatted_values})

    os.makedirs("submissions", exist_ok=True)
    output_filename = os.path.join("submissions", f"submission_{args.output_name}.csv")
    df.to_csv(output_filename, index=False)
    print(f"\nSaved submission to {output_filename}")
    print(f"Total RLE entries: {len(ids)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pattern', type=str, default='models/expert_model_fold*.pth')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--encoder', type=str, default='efficientnet-b3')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true', default=True)
    parser.add_argument('--temporal_smooth', action='store_true', default=True)
    parser.add_argument('--temporal_kernel', type=int, default=3)
    parser.add_argument('--post_process', action='store_true', default=True)
    parser.add_argument('--min_size', type=int, default=50)
    parser.add_argument('--use_clahe', action='store_true', default=True)
    parser.add_argument('--output_name', type=str, default='expert_ensemble')
    args = parser.parse_args()
    inference(args)
