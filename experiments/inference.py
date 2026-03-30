import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import cv2

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

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_model(arch=args.arch, encoder_name=args.encoder, classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_data = load_zipped_pickle("./data/test.pkl")
    
    # Transform
    _, val_transform = get_transforms(img_size=args.img_size)
    
    ids = []
    rle_values = []
    
    print("Running inference...")
    with torch.no_grad():
        for item in tqdm(test_data):
            video = item['video'] # (H, W, T)
            name = item['name']
            H, W, T = video.shape
            
            # Process frame by frame
            # Note: This is inefficient, batching would be better.
            # But for simplicity and correctness first.
            
            video_mask = np.zeros((H, W, T), dtype=bool)
            
            for t in range(T):
                frame = video[:, :, t]
                
                # Preprocess
                # Resize to model input size
                frame_resized = cv2.resize(frame, (args.img_size, args.img_size))
                frame_rgb = np.stack([frame_resized]*3, axis=-1)
                
                augmented = val_transform(image=frame_rgb)['image']
                input_tensor = augmented.unsqueeze(0).to(device)
                
                output = model(input_tensor)
                pred = torch.sigmoid(output) > 0.5
                pred = pred.squeeze().cpu().numpy() # (img_size, img_size)
                
                # Resize back to original size
                # Use nearest neighbor to keep it binary? Or linear then threshold?
                # Resize expects float usually.
                pred_float = pred.astype(np.float32)
                pred_original = cv2.resize(pred_float, (W, H), interpolation=cv2.INTER_NEAREST)
                
                video_mask[:, :, t] = pred_original.astype(bool)
                
            # Flatten and RLE
            flat_mask = video_mask.flatten()
            starts, lengths = get_sequences(flat_mask)
            
            # Format for submission
            # "name_i"
            # We need to generate unique IDs for each row?
            # Wait, the sample submission says:
            # id: name_i
            # value: [start, len]
            # But a video has multiple sequences.
            # "A value in id has the form name_i, where name is the name of a video and i is a unique natural number"
            
            for i, (start, length) in enumerate(zip(starts, lengths)):
                ids.append(f"{name}_{i}")
                rle_values.append([int(start), int(length)])
                
    # Create DataFrame
    # value column format: "[start, len]"
    formatted_values = [str(v) for v in rle_values]
    
    df = pd.DataFrame({"id": ids, "value": formatted_values})
    model_basename = os.path.basename(args.model_path).replace('.pth', '')
    output_filename = os.path.join("submissions", f"submission_{model_basename}.csv")
    df.to_csv(output_filename, index=False)
    print(f"Saved submission to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus', help='Unet, UnetPlusPlus, DeepLabV3Plus')
    args = parser.parse_args()
    inference(args)
