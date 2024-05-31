import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def read_original_data(file_path):
    """Reads the original data file."""
    return pd.read_csv(file_path, sep="\t")

def read_predictions_data(file_path):
    """Reads the predictions data file."""
    predictions_data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 2:
                id_, predicted_sentiment = parts
                predictions_data.append({"id": id_.strip(), "predicted_sentiment": predicted_sentiment.strip()})
    return pd.DataFrame(predictions_data)

def merge_data(original_df, predictions_df):
    """Merges the original and predictions dataframes."""
    original_df_subset = original_df[['id', 'sentiment']]
    return pd.merge(original_df_subset, predictions_df, on="id", how="inner")

def save_merged_data(merged_df, file_path):
    """Saves the merged dataframe to a CSV file."""
    merged_df.to_csv(file_path, index=False)
    print("Merged file saved successfully.")

def plot_confusion_matrix(original_df, predictions_df, ax=None):
    """Plots the confusion matrix."""
    # Merge the dataframes
    merged_df = merge_data(original_df, predictions_df)
    merged_df['sentiment'] = merged_df['sentiment'].astype(int)
    merged_df['predicted_sentiment'] = merged_df['predicted_sentiment'].astype(int)
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(merged_df['sentiment'], merged_df['predicted_sentiment'])

    # Plot the heatmap
    if ax is None:
        plt.figure(figsize=(8, 6))
    else:
        plt.sca(ax)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=True)
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.title('Confusion Matrix')


def plot_sentiment_frequency(csv_file, ax=None):
    df = pd.read_csv(csv_file, sep="\t")   
    # Count the frequency of each category
    category_counts = df['sentiment'].value_counts().sort_index()
    
    # Plot the bar graph
    if ax is None:
        plt.figure(figsize=(8, 6))
    else:
        plt.sca(ax)
    category_counts.plot(kind='bar', color=(1.0, 0.5, 0.5))
    
    # Add labels and title
    plt.xlabel('Sentiment Category')
    plt.ylabel('Frequency')
    plt.title('Frequency of Sentiment Categories')



def main():
    # Example usage
    original_df = read_original_data("data/ids-sst-dev.csv")
    predictions_df = read_predictions_data("predictions_check/sst-dev-output.csv")
    merged_df = merge_data(original_df, predictions_df)
    save_merged_data(merged_df, "merged_file.csv")
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    # Plot confusion matrix
    plot_confusion_matrix(original_df, predictions_df, ax=axes[0])
    
    # Plot sentiment frequency
    plot_sentiment_frequency("data/ids-sst-dev.csv", ax=axes[1])
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("combined_plots.png")


if __name__ == "__main__":
    main()