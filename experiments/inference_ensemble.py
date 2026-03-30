"""
Ensemble Inference - Combines Multiple Model Families
- Teammate's proven ResNet34 models (0.55 score)
- New Attention U-Net with EfficientNet models
- Weighted ensemble with optimal threshold search
- Multi-scale + TTA support
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gzip
import pickle
import cv2
import argparse
import os
from glob import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)


class ModelConfig:
    """Configuration for a model family"""
    def __init__(self, name, pattern, encoder, arch, decoder_attention=None,
                 img_size=256, weight=1.0):
        self.name = name
        self.pattern = pattern
        self.encoder = encoder
        self.arch = arch
        self.decoder_attention = decoder_attention
        self.img_size = img_size
        self.weight = weight


def get_model(arch='UnetPlusPlus', encoder_name='resnet34', decoder_attention=None):
    """Get model based on architecture and encoder"""
    if arch == 'Unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type=decoder_attention
        )
    elif arch == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type=decoder_attention
        )
    elif arch == 'MAnet':
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
    elif arch == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def get_transform(img_size):
    """Get transform for specific resolution"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def apply_tta(model, image, device):
    """Apply test-time augmentation"""
    predictions = []

    with torch.no_grad():
        # Original
        pred = torch.sigmoid(model(image))
        predictions.append(pred)

        # Horizontal flip
        pred = torch.sigmoid(model(torch.flip(image, dims=[3])))
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred)

        # Vertical flip
        pred = torch.sigmoid(model(torch.flip(image, dims=[2])))
        pred = torch.flip(pred, dims=[2])
        predictions.append(pred)

        # Both flips
        pred = torch.sigmoid(model(torch.flip(image, dims=[2, 3])))
        pred = torch.flip(pred, dims=[2, 3])
        predictions.append(pred)

    return torch.stack(predictions).mean(dim=0)


def load_model_family(config, device):
    """Load all models for a given configuration"""
    model_paths = sorted(glob(config.pattern))
    if not model_paths:
        print(f"  Warning: No models found for {config.name}: {config.pattern}")
        return []

    models = []
    for path in model_paths:
        try:
            model = get_model(
                arch=config.arch,
                encoder_name=config.encoder,
                decoder_attention=config.decoder_attention
            )
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"  Error loading {path}: {e}")

    print(f"  {config.name}: Loaded {len(models)} models (weight: {config.weight})")
    return models


def predict_with_family(models, config, frame, original_size, device, use_tta=True, use_clahe=True):
    """Get prediction from a model family"""
    if not models:
        return None

    # Preprocess
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame = clahe.apply(frame.copy())

    # Convert to RGB
    image_rgb = np.stack([frame, frame, frame], axis=-1)

    transform = get_transform(config.img_size)
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    preds = []
    for model in models:
        if use_tta:
            pred = apply_tta(model, image_tensor, device)
        else:
            with torch.no_grad():
                pred = torch.sigmoid(model(image_tensor))
        preds.append(pred)

    # Average across fold models
    family_pred = torch.stack(preds).mean(dim=0)

    # Resize to original
    family_pred = F.interpolate(
        family_pred,
        size=original_size,
        mode='bilinear',
        align_corners=False
    )

    return family_pred.squeeze().cpu().numpy()


def temporal_smoothing(predictions, window=3):
    """Apply temporal smoothing"""
    if len(predictions) <= window:
        return predictions

    smoothed = []
    half_window = window // 2

    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        smoothed.append(np.mean(predictions[start:end], axis=0))

    return smoothed


def post_process_mask(mask, min_area=50):
    """Post-process mask"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return cleaned.astype(np.float32)


def main(args):
    print("=" * 70)
    print("ENSEMBLE INFERENCE - MULTI-MODEL COMBINATION")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Define model families to ensemble
    model_configs = []

    # 1. Teammate's proven models (ResNet34 + UnetPlusPlus)
    if args.include_teammate:
        model_configs.append(ModelConfig(
            name="Teammate ResNet34",
            pattern="../models/teammate/best_model_fold*.pth",
            encoder="resnet34",
            arch="UnetPlusPlus",
            decoder_attention=None,
            img_size=256,
            weight=args.teammate_weight
        ))

    # 2. Our simple_jaccard models (ResNet34 + UnetPlusPlus)
    if args.include_simple:
        model_configs.append(ModelConfig(
            name="Simple Jaccard ResNet34",
            pattern="../models/simple_jaccard_fold*.pth",
            encoder="resnet34",
            arch="UnetPlusPlus",
            decoder_attention=None,
            img_size=256,
            weight=args.simple_weight
        ))

    # 3. New Attention U-Net models (EfficientNet-B4 + SCSE)
    if args.include_attention:
        model_configs.append(ModelConfig(
            name="Attention EfficientNet-B4",
            pattern="../models/attention_unet_fold*.pth",
            encoder="efficientnet-b4",
            arch="UnetPlusPlus",
            decoder_attention="scse",
            img_size=256,
            weight=args.attention_weight
        ))

    # Load all model families
    print("\nLoading model families...")
    all_families = []
    total_weight = 0

    for config in model_configs:
        models = load_model_family(config, device)
        if models:
            all_families.append((config, models))
            total_weight += config.weight

    if not all_families:
        print("ERROR: No models loaded!")
        return

    print(f"\nTotal families: {len(all_families)}, Total weight: {total_weight}")

    # Load test data
    print("\nLoading test data...")
    test_data = load_zipped_pickle("../data/test.pkl")
    print(f"Test videos: {len(test_data)}")

    # Run ensemble inference
    print("\nRunning ensemble inference...")
    predictions = {}

    for item in tqdm(test_data):
        video = item['video']
        name = item['name']
        n_frames = video.shape[2]
        original_size = (video.shape[0], video.shape[1])

        video_preds = []

        for frame_idx in range(n_frames):
            frame = video[:, :, frame_idx]

            # Collect predictions from all families
            weighted_preds = []
            weights = []

            for config, models in all_families:
                pred = predict_with_family(
                    models, config, frame, original_size, device,
                    use_tta=args.tta, use_clahe=args.use_clahe
                )
                if pred is not None:
                    weighted_preds.append(pred * config.weight)
                    weights.append(config.weight)

            # Weighted average
            if weighted_preds:
                ensemble_pred = np.sum(weighted_preds, axis=0) / sum(weights)
            else:
                ensemble_pred = np.zeros(original_size)

            video_preds.append(ensemble_pred)

        # Temporal smoothing
        if args.temporal_smooth:
            video_preds = temporal_smoothing(video_preds, window=3)

        predictions[name] = video_preds

    # Apply threshold
    print(f"\nApplying threshold: {args.threshold}")
    results = []

    for item in test_data:
        name = item['name']
        video = item['video']
        n_frames = video.shape[2]

        pred_video = np.zeros_like(video, dtype=np.float32)

        for frame_idx in range(n_frames):
            pred = predictions[name][frame_idx]
            pred_binary = (pred > args.threshold).astype(np.float32)

            if args.post_process:
                pred_binary = post_process_mask(pred_binary, min_area=args.min_area)

            pred_video[:, :, frame_idx] = pred_binary

        results.append({
            'name': name,
            'prediction': pred_video.astype(bool)
        })

    # Save submission
    os.makedirs("../submissions", exist_ok=True)
    output_path = f"../submissions/submission_{args.output_name}.csv"
    save_zipped_pickle(results, output_path.replace('.csv', '.pkl'))

    print(f"\nSaved: {output_path.replace('.csv', '.pkl')}")

    # Statistics
    total_positive = sum(r['prediction'].sum() for r in results)
    total_pixels = sum(r['prediction'].size for r in results)
    print(f"Positive ratio: {total_positive/total_pixels*100:.3f}%")

    # Print ensemble composition
    print("\n" + "=" * 70)
    print("ENSEMBLE COMPOSITION")
    print("=" * 70)
    for config, models in all_families:
        print(f"  {config.name}: {len(models)} models, weight={config.weight}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model inclusion flags
    parser.add_argument('--include_teammate', action='store_true', default=True)
    parser.add_argument('--include_simple', action='store_true', default=True)
    parser.add_argument('--include_attention', action='store_true', default=True)

    # Model weights
    parser.add_argument('--teammate_weight', type=float, default=1.0)
    parser.add_argument('--simple_weight', type=float, default=0.5)
    parser.add_argument('--attention_weight', type=float, default=1.0)

    # Inference settings
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true', default=True)
    parser.add_argument('--temporal_smooth', action='store_true', default=True)
    parser.add_argument('--post_process', action='store_true', default=True)
    parser.add_argument('--min_area', type=int, default=50)
    parser.add_argument('--use_clahe', action='store_true', default=True)
    parser.add_argument('--output_name', type=str, default='ensemble')

    args = parser.parse_args()
    main(args)
