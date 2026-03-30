"""
Analyze mask characteristics to determine optimal loss function
"""
import gzip
import pickle
import numpy as np
from scipy import ndimage
import cv2

def load_data():
    with gzip.open('./data/train.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_masks(train_data):
    """Analyze mask properties to guide loss function selection"""

    results = {
        'expert': {'ratios': [], 'sizes': [], 'boundaries': [], 'compactness': []},
        'amateur': {'ratios': [], 'sizes': [], 'boundaries': [], 'compactness': []}
    }

    for item in train_data:
        label = item['label']
        frames = item['frames']
        dataset = item.get('dataset', 'amateur')

        for frame_idx in frames:
            mask = (label[:, :, frame_idx] > 0.5).astype(np.uint8)
            total_pixels = mask.size
            fg_pixels = mask.sum()

            if fg_pixels == 0:
                continue

            # 1. Foreground ratio (class imbalance indicator)
            ratio = fg_pixels / total_pixels
            results[dataset]['ratios'].append(ratio)
            results[dataset]['sizes'].append(fg_pixels)

            # 2. Boundary length (perimeter)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            results[dataset]['boundaries'].append(perimeter)

            # 3. Compactness = 4π × Area / Perimeter²
            # Circle = 1.0, thin/elongated shapes < 1
            if perimeter > 0:
                compactness = 4 * np.pi * fg_pixels / (perimeter ** 2)
                results[dataset]['compactness'].append(compactness)

    return results

def print_analysis(results):
    print("=" * 70)
    print("MASK ANALYSIS FOR LOSS FUNCTION SELECTION")
    print("=" * 70)

    for dataset in ['expert', 'amateur']:
        data = results[dataset]
        if not data['ratios']:
            continue

        ratios = np.array(data['ratios'])
        sizes = np.array(data['sizes'])
        boundaries = np.array(data['boundaries'])
        compactness = np.array(data['compactness'])

        print(f"\n{'='*30} {dataset.upper()} {'='*30}")

        print(f"\n1. CLASS IMBALANCE (foreground ratio):")
        print(f"   Mean:   {ratios.mean()*100:.3f}%")
        print(f"   Median: {np.median(ratios)*100:.3f}%")
        print(f"   Std:    {ratios.std()*100:.3f}%")
        print(f"   Range:  {ratios.min()*100:.3f}% - {ratios.max()*100:.3f}%")

        print(f"\n2. MASK SIZE (pixels):")
        print(f"   Mean:   {sizes.mean():.0f}")
        print(f"   Median: {np.median(sizes):.0f}")
        print(f"   Std:    {sizes.std():.0f}")
        print(f"   Very small (<300px): {(sizes < 300).sum()}/{len(sizes)} ({(sizes < 300).mean()*100:.1f}%)")
        print(f"   Small (300-1000px):  {((sizes >= 300) & (sizes < 1000)).sum()}/{len(sizes)}")
        print(f"   Medium (1000-3000):  {((sizes >= 1000) & (sizes < 3000)).sum()}/{len(sizes)}")
        print(f"   Large (>3000px):     {(sizes >= 3000).sum()}/{len(sizes)}")

        print(f"\n3. BOUNDARY COMPLEXITY (perimeter):")
        print(f"   Mean perimeter: {boundaries.mean():.1f} pixels")
        print(f"   Perimeter/√Area ratio: {(boundaries / np.sqrt(sizes)).mean():.2f}")
        print(f"   (Circle=3.54, higher=more complex)")

        print(f"\n4. SHAPE COMPACTNESS (4πA/P²):")
        print(f"   Mean:   {compactness.mean():.3f}")
        print(f"   Median: {np.median(compactness):.3f}")
        print(f"   (1.0=circle, <0.5=elongated/thin, <0.2=very thin)")

        # Classify shapes
        thin = (compactness < 0.3).sum()
        moderate = ((compactness >= 0.3) & (compactness < 0.6)).sum()
        compact = (compactness >= 0.6).sum()
        print(f"   Thin (<0.3):     {thin}/{len(compactness)} ({thin/len(compactness)*100:.1f}%)")
        print(f"   Moderate:        {moderate}/{len(compactness)} ({moderate/len(compactness)*100:.1f}%)")
        print(f"   Compact (>0.6):  {compact}/{len(compactness)} ({compact/len(compactness)*100:.1f}%)")

def recommend_loss(results):
    print("\n" + "=" * 70)
    print("LOSS FUNCTION RECOMMENDATIONS")
    print("=" * 70)

    # Use expert data since test matches expert distribution
    data = results['expert']
    ratios = np.array(data['ratios'])
    sizes = np.array(data['sizes'])
    compactness = np.array(data['compactness'])

    issues = []
    recommendations = []

    # 1. Class imbalance check
    mean_ratio = ratios.mean()
    if mean_ratio < 0.01:
        issues.append(f"SEVERE class imbalance ({mean_ratio*100:.2f}% foreground)")
        recommendations.append(("Focal Loss", "Downweights easy background pixels"))
        recommendations.append(("High pos_weight BCE", f"pos_weight={int(1/mean_ratio)} to balance classes"))
    elif mean_ratio < 0.05:
        issues.append(f"Moderate class imbalance ({mean_ratio*100:.2f}% foreground)")
        recommendations.append(("Dice/Jaccard Loss", "Naturally handles class imbalance"))

    # 2. Small mask check
    small_ratio = (sizes < 500).mean()
    if small_ratio > 0.3:
        issues.append(f"Many small masks ({small_ratio*100:.0f}% < 500px)")
        recommendations.append(("Tversky (α=0.3, β=0.7)", "Penalizes missing small regions (FN)"))

    # 3. Shape complexity check
    thin_ratio = (compactness < 0.3).mean()
    if thin_ratio > 0.3:
        issues.append(f"Thin/elongated shapes ({thin_ratio*100:.0f}% compactness < 0.3)")
        recommendations.append(("Lovasz Loss", "Better IoU optimization for complex boundaries"))
        recommendations.append(("Dice Loss", "Smoother gradients for thin structures"))

    # 4. High variance check
    if ratios.std() / ratios.mean() > 0.5:
        issues.append(f"High size variance (CV={ratios.std()/ratios.mean():.2f})")
        recommendations.append(("Combo Loss", "Mix of losses handles variability"))

    print("\nDETECTED ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. ⚠️  {issue}")

    print("\nRECOMMENDED LOSSES (in priority order):")

    # Always include proven baseline
    print(f"\n   🥇 BCE + Jaccard (0.25/0.75)")
    print(f"      Why: PROVEN at 0.55 by teammate")
    print(f"      Jaccard = IoU loss, directly optimizes competition metric")

    seen = set()
    rank = 2
    for loss, reason in recommendations:
        if loss not in seen:
            medal = "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"\n   {medal} {loss}")
            print(f"      Why: {reason}")
            seen.add(loss)
            rank += 1
            if rank > 4:
                break

    print("\n" + "-" * 70)
    print("SPECIFIC LOSS CONFIGURATIONS TO TRY:")
    print("-" * 70)

    print("""
    1. BASELINE (teammate's proven config):
       loss = 0.25 * BCEWithLogitsLoss() + 0.75 * JaccardLoss()

    2. FOCAL + DICE (for class imbalance):
       loss = 0.3 * FocalLoss(gamma=2.0) + 0.7 * DiceLoss()

    3. TVERSKY (if missing small regions):
       loss = TverskyLoss(alpha=0.3, beta=0.7)  # Penalize FN more

    4. COMBO (kitchen sink):
       loss = 0.2 * BCE + 0.3 * Dice + 0.5 * Jaccard
    """)

if __name__ == '__main__':
    print("Loading data...")
    train_data = load_data()
    print(f"Loaded {len(train_data)} videos")

    print("Analyzing masks...")
    results = analyze_masks(train_data)

    print_analysis(results)
    recommend_loss(results)
