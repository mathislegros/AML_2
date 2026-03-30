"""
Full TTA Inference (Multi-scale + 8 augmentations)
- Uses teammate's models
- 3 scales: 224, 256, 288
- 8 augmentations: original, hflip, vflip, both flips, rot90, rot180, rot270, rot90+hflip
- Total: 3 x 8 x 5 models = 120 predictions per frame
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


def apply_augmentation(image, aug_type):
    """Apply augmentation and return (augmented_image, reverse_function)"""
    if aug_type == 0:  # original
        return image, lambda x: x
    elif aug_type == 1:  # hflip
        return np.fliplr(image).copy(), lambda x: np.fliplr(x)
    elif aug_type == 2:  # vflip
        return np.flipud(image).copy(), lambda x: np.flipud(x)
    elif aug_type == 3:  # both flips
        return np.flipud(np.fliplr(image)).copy(), lambda x: np.flipud(np.fliplr(x))
    elif aug_type == 4:  # rot90
        return np.rot90(image, k=1).copy(), lambda x: np.rot90(x, k=-1)
    elif aug_type == 5:  # rot180
        return np.rot90(image, k=2).copy(), lambda x: np.rot90(x, k=-2)
    elif aug_type == 6:  # rot270
        return np.rot90(image, k=3).copy(), lambda x: np.rot90(x, k=-3)
    elif aug_type == 7:  # rot90 + hflip
        img = np.rot90(image, k=1).copy()
        img = np.fliplr(img).copy()
        return img, lambda x: np.rot90(np.fliplr(x), k=-1)


def predict_with_full_tta(models, image_rgb, original_size, device, scales=[224, 256, 288]):
    """Predict with full TTA (3 scales x 8 augmentations = 24 predictions per model)"""
    H, W = original_size
    all_preds = []

    for scale in scales:
        transform = get_transform(scale)

        for aug_type in range(8):
            aug_image, reverse_fn = apply_augmentation(image_rgb, aug_type)

            transformed = transform(image=aug_image)
            image_tensor = transformed['image'].unsqueeze(0).to(device)

            # Average across all models
            model_preds = []
            with torch.no_grad():
                for model in models:
                    pred = torch.sigmoid(model(image_tensor))
                    model_preds.append(pred)

            avg_pred = torch.stack(model_preds).mean(dim=0)

            # Resize to original size (might be different due to rotation)
            pred_np = avg_pred.squeeze().cpu().numpy()

            # Reverse the augmentation
            pred_np = reverse_fn(pred_np).copy()  # .copy() to fix negative strides

            # Resize to original dimensions if needed (rotations change dimensions)
            if pred_np.shape != (H, W):
                pred_tensor = torch.from_numpy(pred_np.copy()).unsqueeze(0).unsqueeze(0).float()
                pred_tensor = F.interpolate(pred_tensor, size=(H, W), mode='bilinear', align_corners=False)
                pred_np = pred_tensor.squeeze().numpy()

            all_preds.append(pred_np)

    # Average all predictions
    final_pred = np.mean(all_preds, axis=0)
    return final_pred


def main():
    print("=" * 70)
    print("FULL TTA INFERENCE (Multi-scale + 8 Augmentations)")
    print("Scales: 224, 256, 288")
    print("Augmentations: orig, hflip, vflip, both, rot90, rot180, rot270, rot90+hflip")
    print("Total predictions per frame: 3 x 8 x 5 models = 120")
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

    scales = [224, 256, 288]

    # Run full TTA predictions
    print(f"\nRunning full TTA inference...")
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

            pred = predict_with_full_tta(models, image_rgb, (H, W), device, scales)
            video_preds.append(pred)

        all_predictions[name] = video_preds

    # Save soft predictions for threshold tuning
    print("\nSaving soft predictions...")
    os.makedirs("predictions", exist_ok=True)
    with gzip.open("predictions/full_tta_soft.pkl.gz", 'wb') as f:
        pickle.dump(all_predictions, f)
    print("Saved: predictions/full_tta_soft.pkl.gz")

    # Create submissions at multiple thresholds
    print("\nCreating submissions at multiple thresholds...")
    os.makedirs("submissions", exist_ok=True)

    thresholds = [0.45, 0.48, 0.5, 0.52, 0.55]
    for threshold in thresholds:
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
        output_path = f"submissions/submission_full_tta_thresh{threshold:.2f}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(rle_values)} segments)")

    print("\nDone!")


if __name__ == '__main__':
    main()
