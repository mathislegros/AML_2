"""
TTA (Test-Time Augmentation) Inference
- Uses teammate's models with horizontal + vertical flips
- Averages predictions for more robust results
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gzip
import pickle
import os
from glob import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


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


def get_model(encoder_name='resnet34'):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type=None
    )
    return model


def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def predict_with_tta(models, image_rgb, original_size, device, transform):
    """Predict with TTA (original + hflip + vflip + hflip+vflip)"""
    H, W = original_size

    # Create augmented versions
    augmentations = [
        image_rgb,                          # original
        np.fliplr(image_rgb).copy(),        # horizontal flip
        np.flipud(image_rgb).copy(),        # vertical flip
        np.flipud(np.fliplr(image_rgb)).copy(),  # both flips
    ]

    all_preds = []

    for aug_idx, aug_image in enumerate(augmentations):
        transformed = transform(image=aug_image)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        # Average across all models
        model_preds = []
        with torch.no_grad():
            for model in models:
                pred = torch.sigmoid(model(image_tensor))
                model_preds.append(pred)

        avg_pred = torch.stack(model_preds).mean(dim=0)

        # Resize to original size
        avg_pred = F.interpolate(avg_pred, size=(H, W), mode='bilinear', align_corners=False)
        pred_np = avg_pred.squeeze().cpu().numpy()

        # Reverse the augmentation
        if aug_idx == 1:  # hflip
            pred_np = np.fliplr(pred_np)
        elif aug_idx == 2:  # vflip
            pred_np = np.flipud(pred_np)
        elif aug_idx == 3:  # both
            pred_np = np.flipud(np.fliplr(pred_np))

        all_preds.append(pred_np)

    # Average all TTA predictions
    final_pred = np.mean(all_preds, axis=0)
    return final_pred


def main():
    print("=" * 70)
    print("TTA INFERENCE (Teammate Models)")
    print("=" * 70)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test data
    print("\nLoading test data...")
    test_data = load_zipped_pickle("data/test.pkl")
    print(f"Test videos: {len(test_data)}")

    # Load teammate's models
    print("\nLoading teammate's models...")
    model_paths = sorted(glob("models/teammate/best_model_fold*.pth"))
    models = []
    for path in model_paths:
        model = get_model(encoder_name="resnet34")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} models")

    transform = get_transform(256)

    # Run TTA predictions
    print("\nRunning TTA inference (4 augmentations per frame)...")
    all_predictions = {}

    for item in tqdm(test_data, desc="Processing videos"):
        video = item['video']
        name = item['name']
        n_frames = video.shape[2]
        H, W = video.shape[0], video.shape[1]

        video_preds = []
        for frame_idx in range(n_frames):
            frame = video[:, :, frame_idx]
            image_rgb = np.stack([frame, frame, frame], axis=-1)

            pred = predict_with_tta(models, image_rgb, (H, W), device, transform)
            video_preds.append(pred)

        all_predictions[name] = video_preds

    # Create submission with threshold 0.5
    print("\nCreating submission...")
    os.makedirs("submissions", exist_ok=True)

    threshold = 0.5
    ids = []
    rle_values = []

    for item in test_data:
        name = item['name']
        video = item['video']
        n_frames = video.shape[2]

        pred_video = np.zeros_like(video, dtype=bool)
        for frame_idx in range(n_frames):
            pred = all_predictions[name][frame_idx]
            pred_binary = pred > threshold
            pred_video[:, :, frame_idx] = pred_binary

        flat_mask = pred_video.flatten()
        starts, lengths = get_sequences(flat_mask)

        for i, (start, length) in enumerate(zip(starts, lengths)):
            ids.append(f"{name}_{i}")
            rle_values.append([int(start), int(length)])

    formatted_values = [str(v) for v in rle_values]
    df = pd.DataFrame({"id": ids, "value": formatted_values})
    output_path = "submissions/submission_teammate_tta_simple.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Segments: {len(rle_values)}")
    print("\nDone!")


if __name__ == '__main__':
    main()
