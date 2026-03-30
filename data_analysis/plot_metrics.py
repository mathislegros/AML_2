import matplotlib.pyplot as plt
import re
import os

def parse_log(filename):
    folds_data = {}
    current_fold = None
    
    with open(filename, 'r') as f:
        for line in f:
            # Detect Fold
            fold_match = re.search(r"Fold (\d+)/(\d+)", line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                folds_data[current_fold] = {'train_loss': [], 'val_loss': [], 'val_iou': []}
                continue
            
            # Detect Metrics
            # Epoch 1/50 - Train Loss: 0.7071 - Val Loss: 0.6648 - Val IoU: 0.1941
            metrics_match = re.search(r"Epoch \d+/\d+ - Train Loss: ([\d\.]+) - Val Loss: ([\d\.]+) - Val IoU: ([\d\.]+)", line)
            if metrics_match and current_fold is not None:
                folds_data[current_fold]['train_loss'].append(float(metrics_match.group(1)))
                folds_data[current_fold]['val_loss'].append(float(metrics_match.group(2)))
                folds_data[current_fold]['val_iou'].append(float(metrics_match.group(3)))
                
    return folds_data

def plot_metrics(folds_data):
    num_folds = len(folds_data)
    fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5*num_folds))
    
    if num_folds == 1:
        axes = [axes]
    
    for i, fold in enumerate(sorted(folds_data.keys())):
        data = folds_data[fold]
        epochs = range(1, len(data['train_loss']) + 1)
        
        # Plot Loss
        ax_loss = axes[i][0]
        ax_loss.plot(epochs, data['train_loss'], label='Train Loss')
        ax_loss.plot(epochs, data['val_loss'], label='Val Loss')
        ax_loss.set_title(f'Fold {fold} - Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)
        
        # Plot IoU
        ax_iou = axes[i][1]
        ax_iou.plot(epochs, data['val_iou'], label='Val IoU', color='green')
        ax_iou.set_title(f'Fold {fold} - Validation IoU')
        ax_iou.set_xlabel('Epochs')
        ax_iou.set_ylabel('IoU')
        ax_iou.legend()
        ax_iou.grid(True)
        
    plt.tight_layout()
    output_path = os.path.join("runs", 'training_metrics.png')
    plt.savefig(output_path)
    print(f"Saved plots to {output_path}")

if __name__ == "__main__":
    # Find latest log file in runs/
    log_dir = "runs"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.startswith("train_log_") and f.endswith(".txt")]
        if log_files:
            # Sort by timestamp (filename)
            log_files.sort()
            latest_log = os.path.join(log_dir, log_files[-1])
            print(f"Parsing log file: {latest_log}")
            data = parse_log(latest_log)
            plot_metrics(data)
        else:
            print("No log files found in runs/")
    else:
        print("runs/ directory not found.")
