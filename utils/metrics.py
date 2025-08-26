"""
Evaluation Metrics for ML Models
"""

import numpy as np

# Text Generation Metrics
def perplexity_score(y_true, y_pred):
    """Calculate perplexity for language models"""
    # Convert y_true to one-hot
    batch_size, seq_length, vocab_size = y_pred.shape
    y_true_one_hot = np.zeros_like(y_pred)
    
    for i in range(batch_size):
        for j in range(seq_length):
            y_true_one_hot[i, j, y_true[i, j]] = 1
    
    # Clip probabilities to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1.0)
    
    # Calculate cross-entropy
    cross_entropy = -np.sum(y_true_one_hot * np.log(y_pred)) / (batch_size * seq_length)
    
    # Perplexity is exp(cross_entropy)
    perplexity = np.exp(cross_entropy)
    return perplexity

def accuracy_score(y_true, y_pred):
    """Calculate accuracy for text generation"""
    batch_size, seq_length = y_true.shape
    correct = 0
    total = 0
    
    for i in range(batch_size):
        for j in range(seq_length):
            predicted_char = np.argmax(y_pred[i, j, :])
            if predicted_char == y_true[i, j]:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0

# Regression Metrics (kept for compatibility)
def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """Calculate R-squared score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def explained_variance_score(y_true, y_pred):
    """Calculate explained variance score"""
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    
    return 1 - np.var(y_true - y_pred) / var_y

def print_metrics(y_true, y_pred, title="Model Performance"):
    """Print all metrics in a formatted way"""
    print(f"\n{title}")
    print("=" * 50)
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"Explained Variance: {explained_variance_score(y_true, y_pred):.4f}")
    print("=" * 50)

def print_text_metrics(y_true, y_pred, title="Text Generation Performance"):
    """Print text generation metrics"""
    print(f"\n{title}")
    print("=" * 50)
    print(f"Perplexity: {perplexity_score(y_true, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("=" * 50)
