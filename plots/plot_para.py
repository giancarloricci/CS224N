import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot the confusion matrix and the distribution of true categories
def plot_confusion_matrix_with_distribution(predictions, ground_truth):
    # Calculate the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)

    # Create a figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Duplicate', 'Duplicate'], yticklabels=['Not Duplicate', 'Duplicate'], ax=ax[0])
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    ax[0].set_title('Confusion Matrix')

    # Plot the distribution of true categories
    sns.countplot(x=ground_truth, palette='pastel', ax=ax[1])
    ax[1].set_xlabel('True Label')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Distribution of True Categories')
    ax[1].set_xticklabels(['Not Duplicate', 'Duplicate'])

    plt.tight_layout()
    # plt.show()
    plt.savefig("plot_para.png")

# Load the CSV files
true_df = pd.read_csv('data/quora-dev.csv', delimiter='\t')
predicted_df = pd.read_csv('predictions_check/para-dev-output.csv', delimiter=',', header=None, names=['id', 'Predicted_Is_Paraphrase'])

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

# Plot the confusion matrix and distribution with the actual data
plot_confusion_matrix_with_distribution(results_df['predicted_label'], results_df['true_label'])
