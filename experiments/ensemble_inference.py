import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import cv2
import glob

from model import get_model
from utils import load_zipped_pickle
from dataset import get_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_sequences(arr):
    first_indices, last_indices, lengths = [], [], []
    n, i = len(arr), 0
    # Ensure arr is int
    arr = np.array(arr, dtype=np.int32)
    arr = [0] + list(arr) + [0]
    for index, value in enumerate(arr[:-1]):
        if arr[index+1]-arr[index] == 1:
            first_indices.append(index)
        if arr[index+1]-arr[index] == -1:
            last_indices.append(index)
    lengths = list(np.array(last_indices)-np.array(first_indices))
    return first_indices, lengths

def ensemble_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all model paths
    model_paths = glob.glob(args.model_pattern)
    print(f"Found {len(model_paths)} models: {model_paths}")
    
    if len(model_paths) == 0:
        print("No models found!")
        return

    # Load all models
    models = []
    for path in model_paths:
        model = get_model(arch=args.arch, encoder_name=args.encoder, classes=1)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    
    # Load test data
    print("Loading test data...")
    test_data = load_zipped_pickle("./data/test.pkl")
    
    # Transform
    _, val_transform = get_transforms(img_size=args.img_size)
    
    ids = []
    rle_values = []
    
    print("Running ensemble inference...")
    with torch.no_grad():
        for item in tqdm(test_data):
            video = item['video'] # (H, W, T)
            name = item['name']
            H, W, T = video.shape
            
            video_mask = np.zeros((H, W, T), dtype=bool)
            
            for t in range(T):
                frame = video[:, :, t]
                
                # Preprocess
                frame_resized = cv2.resize(frame, (args.img_size, args.img_size))
                frame_rgb = np.stack([frame_resized]*3, axis=-1)
                
                augmented = val_transform(image=frame_rgb)['image']
                input_tensor = augmented.unsqueeze(0).to(device)
                
                # Ensemble prediction
                total_prob = torch.zeros((1, 1, args.img_size, args.img_size)).to(device)
                
                for model in models:
                    output = model(input_tensor)
                    prob = torch.sigmoid(output)
                    total_prob += prob
                
                avg_prob = total_prob / len(models)
                pred = avg_prob > 0.5
                pred = pred.squeeze().cpu().numpy() # (img_size, img_size)
                
                # Resize back
                pred_float = pred.astype(np.float32)
                pred_original = cv2.resize(pred_float, (W, H), interpolation=cv2.INTER_NEAREST)
                
                video_mask[:, :, t] = pred_original.astype(bool)
                
            # Flatten and RLE
            flat_mask = video_mask.flatten()
            starts, lengths = get_sequences(flat_mask)
            
            for i, (start, length) in enumerate(zip(starts, lengths)):
                ids.append(f"{name}_{i}")
                rle_values.append([int(start), int(length)])
                
    # Create DataFrame
    formatted_values = [str(v) for v in rle_values]
    
    df = pd.DataFrame({"id": ids, "value": formatted_values})
    df = pd.DataFrame({"id": ids, "value": formatted_values})
    output_filename = os.path.join("submissions", f"submission_ensemble_{len(models)}models.csv")
    df.to_csv(output_filename, index=False)
    print(f"Saved submission to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pattern', type=str, default='models/best_model_fold*.pth')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus', help='Unet, UnetPlusPlus, DeepLabV3Plus')
    args = parser.parse_args()
    ensemble_inference(args)
