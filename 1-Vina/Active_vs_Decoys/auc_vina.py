import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import re

def preprocess_line(line):
    """Extract filename, score, and label from a line of raw data."""
    # Extract the score using regex
    match = re.search(r'(-\d+\.\d+)', line)
    score = float(match.group(1)) if match else None
    
    # Determine if the line is an Active or Decoy
    if 'Actives' in line:
        label = 1
    else:
        label = 0

    # Extract Filename
    filename = line.split(':')[0].strip()
    
    return [filename, score, label]

def read_and_preprocess(file_path):
    """Read the raw data file and preprocess it."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process lines into a DataFrame
    data = [preprocess_line(line) for line in lines]
    df = pd.DataFrame(data, columns=['Filename', 'Score', 'True Label'])
    
    return df

def calculate_ef(sorted_labels, n_total, n_positive, fraction):
    """Calculate the enrichment factor at a given fraction of the top results."""
    n_top = int(n_total * fraction)
    if n_top == 0:
        return 0
    n_top_positive = np.sum(sorted_labels[:n_top] == 1)
    ef = (n_top_positive / n_top) / (n_positive / n_total)
    return ef

def calculate_roc_auc(true_labels, scores):
    """Calculate the Area Under the Receiver Operating Characteristic Curve (AU-ROC)."""
    return roc_auc_score(true_labels, scores)

def plot_roc_curve(true_labels, scores):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(true_labels, scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % (roc_auc_score(true_labels, scores) * 100))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def main(input_file):
    # Read and preprocess the raw data file
    df = read_and_preprocess(input_file)
    
    # Sort by Score (ascending, as lower scores are better)
    df_sorted = df.sort_values(by='Score')
    
    # Determine the total number of entries and the number of positive (Actives) entries
    n_total = len(df_sorted)
    n_positive = np.sum(df_sorted['True Label'])
    
    true_labels = df_sorted['True Label']
    scores = df_sorted['Score']
    
    # Calculate AU-ROC and convert to percentage
    auc = calculate_roc_auc(true_labels, -scores)  # Negate scores as lower is better
    auc_percentage = auc * 100
    print(f"AU-ROC: {auc_percentage:.2f}%")
    
    # Calculate EF 1% and EF 20%
    ef_1 = calculate_ef(true_labels, n_total, n_positive, 0.01)
    ef_20 = calculate_ef(true_labels, n_total, n_positive, 0.20)
    print(f"EF 1%: {ef_1:.4f}")
    print(f"EF 20%: {ef_20:.4f}")
    
    # Plot ROC Curve
    plot_roc_curve(true_labels, -scores)  # Negate scores to match ROC expectation

# Input file name
input_file = 'best_scores.txt'  # Replace with the name of your actual file if different
main(input_file)
