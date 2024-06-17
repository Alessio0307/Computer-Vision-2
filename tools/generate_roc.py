import os
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, roc_auc, title):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"{title}.png")
    plt.show()

def main(output_dir):
    with open(os.path.join(output_dir, 'roc_data.json'), 'r') as f:
        data = json.load(f)

    all_targets = np.array(data['targets'])
    all_losses = np.array(data['losses'])
    all_total_losses = np.array(data['total_losses'])

    # Compute ROC curve for HRNet with EDSR (total_loss)
    fpr, tpr, _ = roc_curve(all_targets, all_total_losses)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve - HRNet with EDSR (Total Loss)')

    # Compute ROC curve for HRNet without EDSR (loss)
    fpr, tpr, _ = roc_curve(all_targets, all_losses)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve - HRNet without EDSR (Loss)')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate ROC Curves')
    parser.add_argument('--output_dir', required=True, help='Output directory where roc_data.json is saved')
    args = parser.parse_args()

    main(args.output_dir)
