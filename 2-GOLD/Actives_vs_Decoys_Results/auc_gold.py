import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

def calculate_ef(sorted_scores, n_total, n_positive, fraction):
    """Calculate the enrichment factor at a given fraction of the top results."""
    n_top = int(n_total * fraction)
    if n_top == 0:
        return 0
    n_top_positive = np.sum(sorted_scores[:n_top] == 1)
    ef = (n_top_positive / n_top) / (n_positive / n_total)
    return ef

def calculate_roc_auc(true_labels, scores):
    """Calculate the Area Under the Receiver Operating Characteristic Curve (AU-ROC)."""
    auc = roc_auc_score(true_labels, scores)
    return auc * 100  # Convert AUC from 0-1 to 0-100

def plot_roc_curve(true_labels, scores):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(true_labels, scores)
    
    plt.figure()
    auc = roc_auc_score(true_labels, scores) * 100  # Convert AUC for the plot legend
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def main(input_file):
    # Define the correct header for 12 columns
    header = ['Score', 'S(PLP)', 'S(hbond)', 'S(cho)', 'S(metal)', 'DE(clash)', 'DE(tors)', 'intcor', 'time', 'File_name', 'Ligand_name', 'Extra']

    # Read the data from the file without headers
    df = pd.read_csv(input_file, delim_whitespace=True, comment='#', header=None)

    # Print the number of columns and the first few rows for debugging
    print("Number of columns:", len(df.columns))
    print("First few rows of the data:\n", df.head())

    # Ensure header length matches the number of columns
    if len(df.columns) == len(header):
        df.columns = header
    else:
        raise ValueError(f"Header length ({len(header)}) does not match number of columns ({len(df.columns)})")

    # Drop the 'Extra' column if it exists
    df = df.drop(columns=['Extra'], errors='ignore')

    # Sort by Score in descending order
    df_sorted = df.sort_values(by='Score', ascending=False)

    # Determine the total number of entries and the number of positive (Library) entries
    n_total = len(df_sorted)
    n_positive = np.sum(df_sorted['File_name'].str.contains('Library'))
    
    # True labels for ROC calculation: 1 for Library and 0 for Decoys
    df_sorted['True Label'] = df_sorted['File_name'].apply(lambda x: 1 if 'Library' in x else 0)
    true_labels = df_sorted['True Label']
    scores = df_sorted['Score']

    # Calculate AU-ROC
    auc = calculate_roc_auc(true_labels, scores)
    print(f"AU-ROC: {auc:.2f}")

    # Calculate EF 1% and EF 20%
    ef_1 = calculate_ef(true_labels, n_total, n_positive, 0.01)
    ef_20 = calculate_ef(true_labels, n_total, n_positive, 0.20)
    print(f"EF 1%: {ef_1:.4f}")
    print(f"EF 20%: {ef_20:.4f}")

    # Plot ROC Curve
    plot_roc_curve(true_labels, scores)

# Input file name
input_file = 'bestranking.txt'  # Replace 'bestranking.txt' with your actual file name
main(input_file)
