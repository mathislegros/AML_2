"""
Comprehensive Data Analysis for Mitral Valve Segmentation
"""
import numpy as np
import pickle
import gzip
from collections import defaultdict
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

print("="*80)
print("COMPREHENSIVE MITRAL VALVE DATASET ANALYSIS")
print("="*80)

# Load data
print("\n[1] Loading data...")
train_data = load_zipped_pickle("./data/train.pkl")
test_data = load_zipped_pickle("./data/test.pkl")

print(f"Training videos: {len(train_data)}")
print(f"Test videos: {len(test_data)}")

# ============================================================================
# BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print("[2] BASIC STATISTICS")
print("="*80)

# Separate by dataset type
expert_data = [d for d in train_data if d['dataset'] == 'expert']
amateur_data = [d for d in train_data if d['dataset'] == 'amateur']

print(f"\nExpert-labeled videos: {len(expert_data)}")
print(f"Amateur-labeled videos: {len(amateur_data)}")

# Video dimensions
print("\n--- Video Dimensions ---")
for name, data in [("Expert", expert_data), ("Amateur", amateur_data), ("Test", test_data)]:
    heights = [d['video'].shape[0] for d in data]
    widths = [d['video'].shape[1] for d in data]
    frames = [d['video'].shape[2] for d in data]
    print(f"\n{name}:")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
    print(f"  Frames: min={min(frames)}, max={max(frames)}, mean={np.mean(frames):.1f}")

# ============================================================================
# MASK ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[3] MASK / LABEL ANALYSIS")
print("="*80)

def analyze_masks(data, name):
    """Analyze mask characteristics"""
    mask_areas = []
    mask_ratios = []
    mask_centroids_x = []
    mask_centroids_y = []
    mask_widths = []
    mask_heights = []

    for item in data:
        video = item['video']
        label = item['label']
        frames_idx = item['frames']
        H, W, T = video.shape

        for f_idx in frames_idx:
            mask = label[:, :, f_idx]
            area = np.sum(mask)
            ratio = area / (H * W)
            mask_areas.append(area)
            mask_ratios.append(ratio)

            # Find centroid and bounding box
            if area > 0:
                y_coords, x_coords = np.where(mask)
                mask_centroids_x.append(np.mean(x_coords) / W)
                mask_centroids_y.append(np.mean(y_coords) / H)
                mask_widths.append((np.max(x_coords) - np.min(x_coords)) / W)
                mask_heights.append((np.max(y_coords) - np.min(y_coords)) / H)

    print(f"\n{name} Mask Statistics:")
    print(f"  Number of labeled frames: {len(mask_areas)}")
    print(f"  Mask area (pixels): min={min(mask_areas)}, max={max(mask_areas)}, mean={np.mean(mask_areas):.1f}, std={np.std(mask_areas):.1f}")
    print(f"  Mask ratio (%): min={min(mask_ratios)*100:.3f}, max={max(mask_ratios)*100:.3f}, mean={np.mean(mask_ratios)*100:.3f}")
    print(f"  Centroid X (normalized): mean={np.mean(mask_centroids_x):.3f}, std={np.std(mask_centroids_x):.3f}")
    print(f"  Centroid Y (normalized): mean={np.mean(mask_centroids_y):.3f}, std={np.std(mask_centroids_y):.3f}")
    print(f"  Mask width (normalized): mean={np.mean(mask_widths):.3f}, std={np.std(mask_widths):.3f}")
    print(f"  Mask height (normalized): mean={np.mean(mask_heights):.3f}, std={np.std(mask_heights):.3f}")

    return {
        'areas': mask_areas,
        'ratios': mask_ratios,
        'centroids_x': mask_centroids_x,
        'centroids_y': mask_centroids_y,
        'widths': mask_widths,
        'heights': mask_heights
    }

expert_masks = analyze_masks(expert_data, "Expert")
amateur_masks = analyze_masks(amateur_data, "Amateur")

# ============================================================================
# PIXEL INTENSITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[4] PIXEL INTENSITY ANALYSIS")
print("="*80)

def analyze_intensities(data, name, sample_frames=5):
    """Analyze pixel intensities"""
    all_intensities = []
    mv_intensities = []  # Mitral valve region
    bg_intensities = []  # Background

    for item in data:
        video = item['video']
        H, W, T = video.shape

        # Sample some frames
        if 'label' in item:
            frames_idx = item['frames']
            label = item['label']
            for f_idx in frames_idx:
                frame = video[:, :, f_idx]
                mask = label[:, :, f_idx]

                all_intensities.extend(frame.flatten().tolist())
                if np.sum(mask) > 0:
                    mv_intensities.extend(frame[mask].tolist())
                    bg_intensities.extend(frame[~mask].tolist())
        else:
            # Test data - just sample frames
            sample_idx = np.linspace(0, T-1, min(sample_frames, T), dtype=int)
            for f_idx in sample_idx:
                frame = video[:, :, f_idx]
                all_intensities.extend(frame.flatten().tolist())

    print(f"\n{name} Intensity Statistics:")
    print(f"  Overall: mean={np.mean(all_intensities):.1f}, std={np.std(all_intensities):.1f}")
    if mv_intensities:
        print(f"  MV region: mean={np.mean(mv_intensities):.1f}, std={np.std(mv_intensities):.1f}")
        print(f"  Background: mean={np.mean(bg_intensities):.1f}, std={np.std(bg_intensities):.1f}")
        print(f"  MV vs BG contrast: {abs(np.mean(mv_intensities) - np.mean(bg_intensities)):.1f}")

    return all_intensities

expert_int = analyze_intensities(expert_data, "Expert")
amateur_int = analyze_intensities(amateur_data, "Amateur")
test_int = analyze_intensities(test_data, "Test")

# ============================================================================
# BOUNDING BOX ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[5] BOUNDING BOX ANALYSIS")
print("="*80)

def analyze_boxes(data, name):
    """Analyze bounding boxes"""
    box_ratios = []
    box_sizes = []

    for item in data:
        if 'box' not in item:
            continue
        box = item['box']
        H, W = box.shape

        box_area = np.sum(box)
        total_area = H * W
        box_ratios.append(box_area / total_area)

        # Find box dimensions
        y_coords, x_coords = np.where(box)
        if len(y_coords) > 0:
            box_w = (np.max(x_coords) - np.min(x_coords)) / W
            box_h = (np.max(y_coords) - np.min(y_coords)) / H
            box_sizes.append((box_w, box_h))

    if box_ratios:
        print(f"\n{name} Bounding Box Statistics:")
        print(f"  Box area ratio: mean={np.mean(box_ratios)*100:.2f}%, std={np.std(box_ratios)*100:.2f}%")
        widths = [s[0] for s in box_sizes]
        heights = [s[1] for s in box_sizes]
        print(f"  Box width (normalized): mean={np.mean(widths):.3f}")
        print(f"  Box height (normalized): mean={np.mean(heights):.3f}")

analyze_boxes(expert_data, "Expert")
analyze_boxes(amateur_data, "Amateur")

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[6] TEMPORAL ANALYSIS (Frame Distribution)")
print("="*80)

def analyze_temporal(data, name):
    """Analyze which frames are labeled"""
    relative_frames = []
    frame_gaps = []

    for item in data:
        T = item['video'].shape[2]
        frames = item['frames']

        # Relative position of labeled frames
        for f in frames:
            relative_frames.append(f / T)

        # Gaps between labeled frames
        sorted_frames = sorted(frames)
        for i in range(len(sorted_frames) - 1):
            frame_gaps.append(sorted_frames[i+1] - sorted_frames[i])

    print(f"\n{name} Temporal Distribution:")
    print(f"  Labeled frame positions (relative): mean={np.mean(relative_frames):.3f}, std={np.std(relative_frames):.3f}")
    print(f"  Labeled frame positions: min={min(relative_frames):.3f}, max={max(relative_frames):.3f}")
    if frame_gaps:
        print(f"  Gap between labeled frames: mean={np.mean(frame_gaps):.1f}, min={min(frame_gaps)}, max={max(frame_gaps)}")

analyze_temporal(expert_data, "Expert")
analyze_temporal(amateur_data, "Amateur")

# ============================================================================
# PCA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[7] PCA ANALYSIS")
print("="*80)

def extract_frame_features(data, resize_to=(64, 64), max_frames_per_video=3):
    """Extract features from frames for PCA"""
    features = []
    labels = []  # 'expert', 'amateur', or 'test'

    for item in data:
        video = item['video']
        H, W, T = video.shape
        dataset = item.get('dataset', 'test')

        # Sample frames
        if 'frames' in item:
            sample_idx = item['frames'][:max_frames_per_video]
        else:
            sample_idx = np.linspace(0, T-1, min(max_frames_per_video, T), dtype=int)

        for f_idx in sample_idx:
            frame = video[:, :, f_idx]
            # Resize for consistent feature size
            frame_resized = cv2.resize(frame, resize_to)
            features.append(frame_resized.flatten())
            labels.append(dataset)

    return np.array(features), labels

print("\nExtracting features for PCA...")
train_features, train_labels = extract_frame_features(train_data)
test_features, test_labels = extract_frame_features(test_data)

# Combine and perform PCA
all_features = np.vstack([train_features, test_features])
all_labels = train_labels + test_labels

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_features)

# PCA
pca = PCA(n_components=min(50, len(all_features)))
features_pca = pca.fit_transform(features_scaled)

print(f"\nPCA Results:")
print(f"  Variance explained by first 10 components: {sum(pca.explained_variance_ratio_[:10])*100:.1f}%")
print(f"  Variance explained by first 20 components: {sum(pca.explained_variance_ratio_[:20])*100:.1f}%")

# Analyze separation between groups
expert_idx = [i for i, l in enumerate(all_labels) if l == 'expert']
amateur_idx = [i for i, l in enumerate(all_labels) if l == 'amateur']
test_idx = [i for i, l in enumerate(all_labels) if l == 'test']

if expert_idx and amateur_idx and test_idx:
    expert_pca = features_pca[expert_idx]
    amateur_pca = features_pca[amateur_idx]
    test_pca = features_pca[test_idx]

    print(f"\nPCA Component Analysis (first 3 components):")
    for i in range(3):
        print(f"\n  PC{i+1}:")
        print(f"    Expert:  mean={np.mean(expert_pca[:, i]):.3f}, std={np.std(expert_pca[:, i]):.3f}")
        print(f"    Amateur: mean={np.mean(amateur_pca[:, i]):.3f}, std={np.std(amateur_pca[:, i]):.3f}")
        print(f"    Test:    mean={np.mean(test_pca[:, i]):.3f}, std={np.std(test_pca[:, i]):.3f}")

# ============================================================================
# MASK SHAPE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[8] MASK SHAPE / MORPHOLOGY ANALYSIS")
print("="*80)

def analyze_mask_shapes(data, name):
    """Analyze mask shapes and morphology"""
    aspect_ratios = []
    compactness = []  # 4*pi*area / perimeter^2
    solidity = []  # area / convex hull area

    for item in data:
        label = item['label']
        frames_idx = item['frames']

        for f_idx in frames_idx:
            mask = label[:, :, f_idx].astype(np.uint8)
            if np.sum(mask) == 0:
                continue

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Use largest contour
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < 10:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness.append(4 * np.pi * area / (perimeter ** 2))

            # Bounding rect aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratios.append(w / h if h > 0 else 1)

            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity.append(area / hull_area)

    print(f"\n{name} Mask Morphology:")
    if aspect_ratios:
        print(f"  Aspect ratio (w/h): mean={np.mean(aspect_ratios):.3f}, std={np.std(aspect_ratios):.3f}")
    if compactness:
        print(f"  Compactness: mean={np.mean(compactness):.3f}, std={np.std(compactness):.3f}")
    if solidity:
        print(f"  Solidity: mean={np.mean(solidity):.3f}, std={np.std(solidity):.3f}")

    return {'aspect_ratios': aspect_ratios, 'compactness': compactness, 'solidity': solidity}

expert_shapes = analyze_mask_shapes(expert_data, "Expert")
amateur_shapes = analyze_mask_shapes(amateur_data, "Amateur")

# ============================================================================
# KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[9] KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("""
Based on the analysis:

1. DATA IMBALANCE:
   - Expert vs Amateur ratio matters
   - Test set uses EXPERT labels only
   - Recommendation: Weight expert data higher or train only on expert data

2. MASK CHARACTERISTICS:
   - Very small mask area (<1% of image)
   - High class imbalance requires special handling
   - Recommendation: Use high pos_weight in BCE loss (20-50)

3. IMAGE QUALITY:
   - Expert data likely higher resolution
   - Different intensity distributions
   - Recommendation: Use CLAHE for contrast enhancement

4. SPATIAL CONSISTENCY:
   - Masks appear in consistent regions (use bounding box info)
   - Recommendation: Could crop to bounding box region to focus model

5. TEMPORAL INFORMATION:
   - Labels are sparse (3 frames per video)
   - But MV moves smoothly between frames
   - Recommendation: Use temporal consistency in post-processing

6. MASK MORPHOLOGY:
   - Consistent shape characteristics
   - Recommendation: Use morphological post-processing
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
