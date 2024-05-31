import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_metrics1(metric_sets, labels):
    epochs = range(1, len(metric_sets[0]) + 1)
    
    plt.figure(figsize=(18, 6))
    
    # Define colors
    colors = ['#aec7e8', '#1f77b4', '#98df8a', '#2ca02c', '#fdbf6f', '#ff7f0e']


    
    for i, metrics in enumerate(metric_sets):
        sentiment_acc = [metric[0] for metric in metrics]
        paraphrase_acc = [metric[1] for metric in metrics]
        sts_corr = [metric[2] for metric in metrics]
        
        K = 3
        # Create spline interpolations for smooth lines
        epochs_new = np.linspace(min(epochs), max(epochs), 300)  # Increase number of points for smoothness
        
        # Plot Sentiment Classification Accuracy (spline)
        plt.subplot(1, 3, 1)
        sentiment_spline = make_interp_spline(epochs, sentiment_acc, k=K)
        sentiment_smooth = sentiment_spline(epochs_new)
        plt.plot(epochs_new, sentiment_smooth, label=f'{labels[i]}', color=colors[i])
        plt.xlabel('Epochs')
        plt.ylabel('Sentiment Accuracy')
        plt.title('Sentiment Classification Accuracy')
        plt.legend()
        plt.xticks(epochs)
        plt.grid(False)
        
        # Plot Paraphrase Detection Accuracy (spline)
        plt.subplot(1, 3, 2)
        paraphrase_spline = make_interp_spline(epochs, paraphrase_acc, k=K)
        paraphrase_smooth = paraphrase_spline(epochs_new)
        plt.plot(epochs_new, paraphrase_smooth, label=f'{labels[i]}', color=colors[i+2])
        plt.xlabel('Epochs')
        plt.ylabel('Paraphrase Accuracy')
        plt.title('Paraphrase Detection Accuracy')
        plt.legend()
        plt.xticks(epochs)
        plt.grid(False)
        
        # Plot Semantic Textual Similarity Correlation (spline)
        plt.subplot(1, 3, 3)
        sts_spline = make_interp_spline(epochs, sts_corr, k=K)
        sts_smooth = sts_spline(epochs_new)
        plt.plot(epochs_new, sts_smooth, label=f'{labels[i]}', color=colors[i+4])
        plt.xlabel('Epochs')
        plt.ylabel('STS Correlation')
        plt.title('Semantic Textual Similarity Correlation')
        plt.legend()
        plt.xticks(epochs)
        plt.grid(False)
    
    plt.tight_layout()
    plt.savefig('plot_deg.png')

# Your metric sets and labels

# policy gradient, no smart
metrics1 = [
    (0.445, 0.752, 0.842),  # Epoch 0
    (0.460, 0.774, 0.852),  # Epoch 1
    (0.513, 0.791, 0.864),  # Epoch 2
    (0.496, 0.795, 0.860),  # Epoch 3
    (0.527, 0.819, 0.866),  # Epoch 4
    (0.473, 0.826, 0.868),  # Epoch 5
    (0.500, 0.829, 0.866),  # Epoch 6
    (0.528, 0.829, 0.866),  # Epoch 7 (BEST)
    (0.505, 0.837, 0.865),  # Epoch 8
]

# 0.5, BIG, lr=5e-6
scoresBig = [
    # Epoch 0
    (0.322, 0.820, 0.770),
    # Epoch 1
    (0.486, 0.842, 0.820),
    # Epoch 2
    (0.501, 0.847, 0.852),
    # Epoch 3
    (0.506, 0.859, 0.850),
    # Epoch 4
    (0.471, 0.867, 0.867),
    # Epoch 5
    (0.490, 0.870, 0.867),
    # Epoch 6
    (0.511, 0.873, 0.873),
    # Epoch 7
    (0.505, 0.873, 0.873),
    # Epoch 8
    (0.491, 0.874, 0.863),
    # # Epoch 9
    # (0.493, 0.878, 0.869)
]

metric_sets = [metrics1, scoresBig]
labels = ['PCGrad', "No Freeeze"]

plot_metrics1(metric_sets, labels)
