import numpy as np
import pandas as pd

# Sample normalized data for pre-trained models
data = np.array([
    [0.75, 0.85, 0.90, 0.80],  
    [0.80, 0.90, 0.85, 0.75], 
    [0.90, 0.75, 0.80, 0.85],  
    [0.96, 0.34, 0.40, 0.83],  
])

weights = np.array([0.25, 0.30, 0.20, 0.25])

# Check if the length of weights matches the number of columns in data
if len(weights) != data.shape[1]:
    raise ValueError("Length of weights must match the number of columns in data.")

# Save the data and weights to a CSV file
columns = ["Perplexity", "BLEU Score", "Inference Speed", "Model Size"]
models_data = pd.DataFrame(data, columns=columns)

csv_file_path = "result.csv"
models_data.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")

def topsis(data, weights):
    squared_data = data ** 2
    weighted_normalized_matrix = data / np.sqrt(np.sum(squared_data, axis=0))
    weighted_normalized_matrix *= weights
    positive_ideal = np.max(weighted_normalized_matrix, axis=0)
    negative_ideal = np.min(weighted_normalized_matrix, axis=0)
    distance_positive = np.sqrt(np.sum((weighted_normalized_matrix - positive_ideal) ** 2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted_normalized_matrix - negative_ideal) ** 2, axis=1))
    topsis_score = distance_negative / (distance_positive + distance_negative)
    return topsis_score

topsis_scores = topsis(data, weights) # TOPSIS scores

ranked_models = np.argsort(topsis_scores)[::-1] # Ranking

print("Ranked Models:", ranked_models)