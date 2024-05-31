import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_metrics(metric_sets, labels):
    epochs = range(1, len(metric_sets[0]) + 1)
    
    plt.figure(figsize=(18, 6))
    
    for i, metrics in enumerate(metric_sets):
        sentiment_acc = [metric[0] for metric in metrics]
        paraphrase_acc = [metric[1] for metric in metrics]
        sts_corr = [metric[2] for metric in metrics]
        
        K = 3
        # Create spline interpolations for smooth lines
        epochs_new = np.linspace(min(epochs), max(epochs), 300)  # Increase number of points for smoothness
        sentiment_spline = make_interp_spline(epochs, sentiment_acc, k=K)
        paraphrase_spline = make_interp_spline(epochs, paraphrase_acc, k=K)
        sts_spline = make_interp_spline(epochs, sts_corr, k=K)
        
        sentiment_smooth = sentiment_spline(epochs_new)
        paraphrase_smooth = paraphrase_spline(epochs_new)
        sts_smooth = sts_spline(epochs_new)
        
        # Plot Sentiment Classification Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(epochs_new, sentiment_smooth, label=labels[i])
        plt.xlabel('Epochs')
        plt.ylabel('Sentiment Accuracy')
        plt.title('Sentiment Classification Accuracy')
        plt.legend()
        plt.xticks(epochs)
        plt.grid(False)
        
        # Plot Paraphrase Detection Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs_new, paraphrase_smooth, label=labels[i])
        plt.xlabel('Epochs')
        plt.ylabel('Paraphrase Accuracy')
        plt.title('Paraphrase Detection Accuracy')
        plt.legend()
        plt.xticks(epochs)
        plt.grid(False)
        
        # Plot Semantic Textual Similarity Correlation
        plt.subplot(1, 3, 3)
        plt.plot(epochs_new, sts_smooth, label=labels[i])
        plt.xlabel('Epochs')
        plt.ylabel('STS Correlation')
        plt.title('Semantic Textual Similarity Correlation')
        plt.legend()
        plt.xticks(epochs)
        plt.grid(False)
    
    plt.tight_layout()
    plt.savefig('plot2.png')



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
    # (0.505, 0.837, 0.865),  # Epoch 8
    # (0.502, 0.837, 0.872),  # Epoch 9
    # (0.493, 0.839, 0.866)   # Epoch 10
]

# # Penalty 1 metrics
# penalty_1 = [
#     (0.445, 0.751, 0.842),  # Epoch 0
#     (0.447, 0.790, 0.855),  # Epoch 1
#     (0.479, 0.807, 0.862),  # Epoch 2
#     (0.487, 0.813, 0.864),  # Epoch 3
#     (0.504, 0.820, 0.872),  # Epoch 4
#     (0.522, 0.827, 0.873),  # Epoch 5
#     (0.520, 0.831, 0.869),  # Epoch 6
#     (0.499, 0.829, 0.864),  # Epoch 7
#     (0.503, 0.835, 0.872),  # Epoch 8
#     # (0.499, 0.829, 0.864),  # Epoch 9
# ]

# do sentiment
metrics = [
    (0.442, 0.760, 0.845),  # Epoch 0
    (0.471, 0.790, 0.855),  # Epoch 1
    (0.514, 0.801, 0.868),  # Epoch 2
    (0.520, 0.813, 0.864),  # Epoch 3
    (0.505, 0.819, 0.865),  # Epoch 4
    (0.493, 0.827, 0.868),  # Epoch 5
    (0.506, 0.832, 0.867),  # Epoch 6
    (0.490, 0.833, 0.867),  # Epoch 7
    # (0.482, 0.837, 0.870),  # Epoch 8
    # (0.493, 0.837, 0.870),  # Epoch 9
]


# # Penalty 0.5 metrics
# penalty_half = [
#     (0.445, 0.751, 0.842),  # Epoch 0
#     (0.449, 0.791, 0.855),  # Epoch 1
#     (0.471, 0.807, 0.862),  # Epoch 2
#     (0.492, 0.814, 0.864),  # Epoch 3
#     (0.506, 0.820, 0.871),  # Epoch 4
#     (0.524, 0.827, 0.873),  # Epoch 5
#     (0.527, 0.830, 0.868),  # Epoch 6
#     (0.499, 0.829, 0.829),  # Epoch 7
    #   (0.510, 0.834, 0.871)   # Epoch 8 
# ]

metric_sets = [metrics1, metrics]
labels = ['PCGrad', 'Include Sentiment']

plot_metrics(metric_sets, labels)


# # penalty 0.5: 
# penalty_half = [
#     (0.445, 0.751, 0.842), # 0
#     (0.449, 0.791, 0.855), # 1
#     (0.471, 0.807, 0.862), # 2
#     (0.492, 0.814, 0.864), # 3
#     (0.506, 0.820, 0.871), # 4
#     (0.524, 0.827, 0.873), # 5 
#     (0.527, 0.830, 0.868), # 6 BEST! 
#     (0.499, 0.829, 0.829) # 7
# ]
    
