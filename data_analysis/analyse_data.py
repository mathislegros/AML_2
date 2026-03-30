import pickle
import gzip
import numpy as np
import pandas as pd

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

print("Loading train data...")
train_data = load_zipped_pickle("./data/train.pkl")
print(f"Loaded {len(train_data)} training videos.")

expert_count = 0
amateur_count = 0
shapes = set()
frame_counts = []

for item in train_data:
    if item['dataset'] == 'expert':
        expert_count += 1
    else:
        amateur_count += 1
    
    video = item['video']
    shapes.add(video.shape[:2])
    frame_counts.append(video.shape[2])

print(f"Expert videos: {expert_count}")
print(f"Amateur videos: {amateur_count}")
print(f"Unique resolutions (H, W): {shapes}")
print(f"Frame counts - Min: {min(frame_counts)}, Max: {max(frame_counts)}, Mean: {np.mean(frame_counts):.2f}")

print("\nLoading test data...")
test_data = load_zipped_pickle("./data/test.pkl")
print(f"Loaded {len(test_data)} test videos.")

test_shapes = set()
for item in test_data:
    video = item['video']
    test_shapes.add(video.shape[:2])

print(f"Test unique resolutions (H, W): {test_shapes}")
