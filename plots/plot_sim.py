import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(predictions, ground_truth, epsilon1=0.5, epsilon2=1, epsilon3=2):
    # Calculate L2 distances
    distances = np.abs(predictions - ground_truth)

    # Split predictions and ground_truth into four regions based on thresholds
    close_mask = distances < epsilon1
    medium_mask = (distances >= epsilon1) & (distances < epsilon2)
    far_mask = (distances >= epsilon2) & (distances < epsilon3)
    very_far_mask = distances >= epsilon3

    close_predictions = predictions[close_mask]
    close_ground_truth = ground_truth[close_mask]

    medium_predictions = predictions[medium_mask]
    medium_ground_truth = ground_truth[medium_mask]

    far_predictions = predictions[far_mask]
    far_ground_truth = ground_truth[far_mask]

    very_far_predictions = predictions[very_far_mask]
    very_far_ground_truth = ground_truth[very_far_mask]

    # Calculate counts
    close_count = np.sum(close_mask)
    medium_count = np.sum(medium_mask)
    far_count = np.sum(far_mask)
    very_far_count = np.sum(very_far_mask)

    # Calculate metrics
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)

    # Print metrics
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R²) Score: {r2}')

    # Set Seaborn color palette
    sns.set_palette("husl")

    # Plot the scatter plot and count bar graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    color=sns.color_palette()[::-1]
    ax1.scatter(close_ground_truth, close_predictions, label='L1 within 0.5', color=color[0], alpha=0.5)
    ax1.scatter(medium_ground_truth, medium_predictions, label='L1 within 1', color=color[1], alpha=0.5)
    ax1.scatter(far_ground_truth, far_predictions, label='L1 within 2', color=color[2], alpha=0.5)
    ax1.scatter(very_far_ground_truth, very_far_predictions, label='L1 beyond 2', color=color[3], alpha=0.5)
    ax1.set_xlabel('True Similarity')
    ax1.set_ylabel('Predicted Similarity')
    ax1.set_title('True vs Predicted Similarity')
    ax1.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], color='black', linestyle='--')
    ax1.legend()

    ax2.bar(['Close', 'Medium', 'Far', 'Very Far'], [close_count, medium_count, far_count, very_far_count], color=sns.color_palette()[::-1], edgecolor='black')
    ax2.set_xlabel('Distance Categories')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Data')
    plt.savefig("sim_with_counts.png")

def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# Load the CSV files
true_df = pd.read_csv('data/sts-dev.csv', delimiter='\t')
predicted_df = pd.read_csv('predictions_check/sts-dev-output.csv', delimiter=',', header=None, names=['id', 'Predicted_Similarity'])

# Clean up dataframes
true_df = clean_dataframe(true_df)
predicted_df = clean_dataframe(predicted_df)
true_df.drop(columns=['Unnamed: 0'], inplace=True)

# Merge the dataframes on 'id'
merged_df = pd.merge(true_df, predicted_df, on='id')

# Extract relevant columns
results_df = merged_df[['id', 'similarity', 'Predicted_Similarity']]

# Rename columns for clarity
results_df.columns = ['id', 'true_similarity', 'predicted_similarity']

# Convert columns to appropriate types
results_df['true_similarity'] = results_df['true_similarity'].astype(float)
results_df['predicted_similarity'] = results_df['predicted_similarity'].astype(float)

# Check if the DataFrame is empty after merge
if results_df.empty:
    print("The merged DataFrame is empty. Please check the ID columns in your input files.")
else:
    # Evaluate predictions and plot
    evaluate_predictions(results_df['predicted_similarity'], results_df['true_similarity'])