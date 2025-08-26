"""
Text Preprocessing Utilities for Character-Level Language Models
"""

import numpy as np

class TextPreprocessor:
    """Preprocess text for character-level language modeling"""
    
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def build_vocabulary(self, text):
        """Build character vocabulary from text"""
        unique_chars = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        return self.char_to_idx, self.idx_to_char
    
    def text_to_sequences(self, text):
        """Convert text to sequences of character indices"""
        # Convert text to indices
        indices = [self.char_to_idx.get(char, 0) for char in text]
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(indices) - self.sequence_length):
            sequence = indices[i:i + self.sequence_length]
            target = indices[i + 1:i + self.sequence_length + 1]
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, text):
        """Prepare text data for training"""
        # Build vocabulary
        self.build_vocabulary(text)
        
        # Convert to sequences
        X, y = self.text_to_sequences(text)
        
        return X, y, self.char_to_idx, self.idx_to_char
    
    def encode_text(self, text):
        """Encode text to character indices"""
        return [self.char_to_idx.get(char, 0) for char in text]
    
    def decode_text(self, indices):
        """Decode character indices back to text"""
        return ''.join([self.idx_to_char.get(idx, '?') for idx in indices])

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance"""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Compute the mean and standard deviation for later scaling"""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Scale features using the computed mean and standard deviation"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before transform")
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit to data, then transform it"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Scale back the data to the original representation"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before inverse_transform")
        return X * self.scale_ + self.mean_

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split arrays into random train and test subsets"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    return (X[train_indices], X[test_indices], 
            y[train_indices], y[test_indices])
