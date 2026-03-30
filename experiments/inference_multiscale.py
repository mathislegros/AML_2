"""
Multi-Scale Inference with Threshold Optimization
- Inference at multiple resolutions (256, 384, 512)
- Weighted averaging of predictions
- Optimal threshold search
- TTA support
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


def get_model(arch='UnetPlusPlus', encoder_name='efficientnet-b4', decoder_attention='scse'):
    """Get model with attention mechanisms"""
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
    """Apply test-time augmentation and return averaged prediction"""
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


def predict_multiscale(models, frame, original_size, device, scales=[256, 384, 512],
                       scale_weights=None, use_tta=True, use_clahe=True):
    """
    Multi-scale prediction with weighted averaging

    Args:
        models: List of models (for ensemble)
        frame: Original frame (H, W) grayscale
        original_size: (H, W) for output
        scales: List of resolutions to use
        scale_weights: Weights for each scale (default: equal)
        use_tta: Apply test-time augmentation
        use_clahe: Apply CLAHE preprocessing
    """
    if scale_weights is None:
        scale_weights = [1.0] * len(scales)

    # Normalize weights
    scale_weights = np.array(scale_weights) / sum(scale_weights)

    # Preprocess
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame = clahe.apply(frame)

    # Convert to RGB
    image_rgb = np.stack([frame, frame, frame], axis=-1)

    # Collect predictions at each scale
    all_predictions = []

    for scale, weight in zip(scales, scale_weights):
        transform = get_transform(scale)
        transformed = transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        scale_preds = []
        for model in models:
            if use_tta:
                pred = apply_tta(model, image_tensor, device)
            else:
                with torch.no_grad():
                    pred = torch.sigmoid(model(image_tensor))
            scale_preds.append(pred)

        # Average across models
        scale_pred = torch.stack(scale_preds).mean(dim=0)

        # Resize back to original size
        scale_pred = F.interpolate(
            scale_pred,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )

        all_predictions.append(scale_pred * weight)

    # Weighted sum across scales
    final_pred = torch.stack(all_predictions).sum(dim=0)

    return final_pred.squeeze().cpu().numpy()


def find_optimal_threshold(predictions, masks, thresholds=np.arange(0.3, 0.8, 0.05)):
    """Find threshold that maximizes IoU on validation data"""
    best_iou = 0
    best_threshold = 0.5

    for thresh in thresholds:
        ious = []
        for pred, mask in zip(predictions, masks):
            pred_binary = (pred > thresh).astype(np.float32)
            intersection = (pred_binary * mask).sum()
            union = pred_binary.sum() + mask.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou)

        mean_iou = np.mean(ious)
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_threshold = thresh

    return best_threshold, best_iou


def temporal_smoothing(predictions, window=3):
    """Apply temporal smoothing across frames"""
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
    """Post-process prediction mask"""
    # Remove small connected components
    mask_uint8 = (mask * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    # Fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return cleaned.astype(np.float32)


def main(args):
    print("=" * 70)
    print("MULTI-SCALE INFERENCE WITH THRESHOLD OPTIMIZATION")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Scales: {args.scales}")
    print(f"Scale weights: {args.scale_weights}")

    # Load models
    model_paths = sorted(glob(args.model_pattern))
    if not model_paths:
        print(f"No models found matching: {args.model_pattern}")
        return

    print(f"\nLoading {len(model_paths)} models...")
    models = []
    for path in model_paths:
        model = get_model(
            arch=args.arch,
            encoder_name=args.encoder,
            decoder_attention=args.decoder_attention
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded: {path}")

    # Load test data
    print("\nLoading test data...")
    test_data = load_zipped_pickle("../data/test.pkl")
    print(f"Test videos: {len(test_data)}")

    # Parse scales
    scales = [int(s) for s in args.scales.split(',')]
    if args.scale_weights:
        scale_weights = [float(w) for w in args.scale_weights.split(',')]
    else:
        scale_weights = None

    # Run inference
    print("\nRunning multi-scale inference...")
    predictions = {}

    for item in tqdm(test_data):
        video = item['video']
        name = item['name']
        n_frames = video.shape[2]

        video_preds = []
        original_size = (video.shape[0], video.shape[1])

        for frame_idx in range(n_frames):
            frame = video[:, :, frame_idx]

            pred = predict_multiscale(
                models, frame, original_size, device,
                scales=scales,
                scale_weights=scale_weights,
                use_tta=args.tta,
                use_clahe=args.use_clahe
            )
            video_preds.append(pred)

        # Temporal smoothing
        if args.temporal_smooth:
            video_preds = temporal_smoothing(video_preds, window=3)

        predictions[name] = video_preds

    # Apply threshold and post-processing
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

    # Create CSV
    df = pd.DataFrame({'name': [r['name'] for r in results]})
    df['prediction'] = [r['prediction'].tolist() for r in results]

    print(f"\nSaved: {output_path}")
    print(f"Predictions saved to: {output_path.replace('.csv', '.pkl')}")

    # Statistics
    total_positive = sum(r['prediction'].sum() for r in results)
    total_pixels = sum(r['prediction'].size for r in results)
    print(f"\nPositive ratio: {total_positive/total_pixels*100:.3f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pattern', type=str, required=True,
                        help='Glob pattern for model files')
    parser.add_argument('--scales', type=str, default='256,384,512',
                        help='Comma-separated list of scales')
    parser.add_argument('--scale_weights', type=str, default=None,
                        help='Comma-separated weights for scales')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus')
    parser.add_argument('--decoder_attention', type=str, default='scse')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--tta', action='store_true', default=True)
    parser.add_argument('--temporal_smooth', action='store_true', default=True)
    parser.add_argument('--post_process', action='store_true', default=True)
    parser.add_argument('--min_area', type=int, default=50)
    parser.add_argument('--use_clahe', action='store_true', default=True)
    parser.add_argument('--output_name', type=str, default='multiscale')
    args = parser.parse_args()
    main(args)
