import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(predictions, ground_truth):
    # Calculate the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure
    plt.figure(figsize=(8, 6))

    # Plot the normalized confusion matrix
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Not Duplicate', 'Duplicate'], yticklabels=['Not Duplicate', 'Duplicate'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')

    plt.tight_layout()
    plt.savefig("plots/plot_para.png")
    # plt.show()

# Load the CSV files and perform the necessary data processing

# Load the CSV files
true_df = pd.read_csv('data/quora-dev.csv', delimiter='\t')
predicted_df = pd.read_csv('predictions/para-dev-output.csv', delimiter=',', header=None, names=['id', 'Predicted_Is_Paraphrase'])

# Clean up any extraneous spaces in the column names and data
true_df.columns = true_df.columns.str.strip()
predicted_df.columns = predicted_df.columns.str.strip()
true_df = true_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
predicted_df = predicted_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Display the first few rows of each dataframe to verify their structure
print(true_df.head())
print(predicted_df.head())

# Merge the dataframes on 'id'
merged_df = pd.merge(true_df, predicted_df, on='id')

# Extract relevant columns
results_df = merged_df[['id', 'is_duplicate', 'Predicted_Is_Paraphrase']]

# Rename columns for clarity
results_df.columns = ['id', 'true_label', 'predicted_label']

# Convert columns to appropriate types
results_df['true_label'] = results_df['true_label'].astype(float).astype(int)
results_df['predicted_label'] = results_df['predicted_label'].astype(float).astype(int)

# Call the function to plot the confusion matrix
plot_confusion_matrix(results_df['predicted_label'], results_df['true_label'])