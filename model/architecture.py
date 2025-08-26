"""
Text Generation Model Implementation - Character-Level Language Model
GPU Accelerated Version
"""

import numpy as np
from utils.gpu_utils import (
    get_array_module, to_gpu_array, to_cpu_array, 
    dot, sum, mean, exp, log, random_randn, zeros, zeros_like,
    synchronize_gpu, print_gpu_info, GPU_AVAILABLE
)

class TextGenerationModel:
    """Character-level text generation model using a simple neural network"""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, learning_rate=0.01, sequence_length=50, device='auto'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.device = device
        
        # Get appropriate array module
        self.xp = get_array_module(device)
        
        # Model parameters
        self.embedding_weights = None
        self.hidden_weights = None
        self.output_weights = None
        self.hidden_bias = None
        self.output_bias = None
        
        # Training history
        self.history = {'loss': []}
        
        # Print device info
        if device == 'auto' or device == 'gpu':
            print_gpu_info()
    
    def initialize_parameters(self):
        """Initialize model parameters"""
        # Xavier/Glorot initialization
        self.embedding_weights = random_randn(self.vocab_size, self.embedding_dim, device=self.device) * self.xp.sqrt(2.0 / self.vocab_size)
        self.hidden_weights = random_randn(self.embedding_dim, self.hidden_dim, device=self.device) * self.xp.sqrt(2.0 / self.embedding_dim)
        self.output_weights = random_randn(self.hidden_dim, self.vocab_size, device=self.device) * self.xp.sqrt(2.0 / self.hidden_dim)
        
        self.hidden_bias = zeros(self.hidden_dim, device=self.device)
        self.output_bias = zeros(self.vocab_size, device=self.device)
    
    def one_hot_encode(self, sequence):
        """Convert sequence of indices to one-hot encoding"""
        batch_size, seq_length = sequence.shape
        one_hot = zeros((batch_size, seq_length, self.vocab_size), device=self.device)
        
        for i in range(batch_size):
            for j in range(seq_length):
                one_hot[i, j, sequence[i, j]] = 1
        
        return one_hot
    
    def softmax(self, x):
        """Compute softmax function"""
        exp_x = exp(x - self.xp.max(x, axis=-1, keepdims=True), device=self.device)
        return exp_x / sum(exp_x, axis=-1, keepdims=True, device=self.device)
    
    def forward(self, X):
        """Forward pass through the network"""
        batch_size, seq_length = X.shape
        
        # One-hot encode input
        X_one_hot = self.one_hot_encode(X)
        
        # Embedding layer
        embedded = dot(X_one_hot, self.embedding_weights, device=self.device)  # (batch_size, seq_length, embedding_dim)
        
        # Hidden layer with ReLU activation
        hidden = dot(embedded, self.hidden_weights, device=self.device) + self.hidden_bias
        hidden = self.xp.maximum(0, hidden)  # ReLU activation
        
        # Output layer
        output = dot(hidden, self.output_weights, device=self.device) + self.output_bias
        
        # Apply softmax to get probabilities
        probabilities = self.softmax(output)
        
        return probabilities, embedded, hidden
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss"""
        # Convert y_true to one-hot
        batch_size, seq_length = y_true.shape
        y_true_one_hot = self.one_hot_encode(y_true)
        
        # Clip probabilities to avoid log(0)
        y_pred = self.xp.clip(y_pred, 1e-15, 1.0)
        
        # Cross-entropy loss
        loss = -sum(y_true_one_hot * log(y_pred, device=self.device), device=self.device) / (batch_size * seq_length)
        return loss
    
    def compute_gradients(self, X, y_true, y_pred, embedded, hidden):
        """Compute gradients for all parameters"""
        batch_size, seq_length = X.shape
        
        # Convert y_true to one-hot
        y_true_one_hot = self.one_hot_encode(y_true)
        
        # Gradient of loss with respect to output
        d_output = (y_pred - y_true_one_hot) / (batch_size * seq_length)
        
        # Gradient of loss with respect to hidden layer
        d_hidden = dot(d_output, self.output_weights.T, device=self.device)
        
        # Gradient of loss with respect to embedding
        d_embedding = dot(d_hidden, self.hidden_weights.T, device=self.device)
        
        # Gradients for weights and biases - reshape for proper matrix multiplication
        d_output_weights = dot(hidden.reshape(-1, self.hidden_dim).T, 
                              d_output.reshape(-1, self.vocab_size), device=self.device)
        d_output_bias = sum(d_output, axis=(0, 1), device=self.device)
        
        d_hidden_weights = dot(embedded.reshape(-1, self.embedding_dim).T, 
                              d_hidden.reshape(-1, self.hidden_dim), device=self.device)
        d_hidden_bias = sum(d_hidden, axis=(0, 1), device=self.device)
        
        # Gradient for embedding weights
        d_embedding_weights = zeros_like(self.embedding_weights, device=self.device)
        X_one_hot = self.one_hot_encode(X)
        for i in range(batch_size):
            for j in range(seq_length):
                d_embedding_weights += self.xp.outer(X_one_hot[i, j], d_embedding[i, j])
        
        return d_embedding_weights, d_hidden_weights, d_output_weights, d_hidden_bias, d_output_bias
    
    def update_parameters(self, gradients):
        """Update model parameters using gradients"""
        d_embedding_weights, d_hidden_weights, d_output_weights, d_hidden_bias, d_output_bias = gradients
        
        self.embedding_weights -= self.learning_rate * d_embedding_weights
        self.hidden_weights -= self.learning_rate * d_hidden_weights
        self.output_weights -= self.learning_rate * d_output_weights
        self.hidden_bias -= self.learning_rate * d_hidden_bias
        self.output_bias -= self.learning_rate * d_output_bias
    
    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Train the model using mini-batch gradient descent"""
        if self.embedding_weights is None:
            self.initialize_parameters()
        
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = self.xp.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred, embedded, hidden = self.forward(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Backward pass
                gradients = self.compute_gradients(X_batch, y_batch, y_pred, embedded, hidden)
                
                # Update parameters
                self.update_parameters(gradients)
            
            # Synchronize GPU if needed
            if self.device == 'gpu':
                synchronize_gpu()
            
            # Average loss for the epoch
            avg_loss = epoch_loss / n_batches
            self.history['loss'].append(float(avg_loss))  # Convert to float for JSON serialization
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        y_pred, _, _ = self.forward(X)
        return y_pred
    
    def generate_text(self, seed_text, char_to_idx, idx_to_char, length=100, temperature=1.0):
        """Generate text given a seed text"""
        generated_text = seed_text
        
        # Initialize sequence with the last characters from seed text
        if len(seed_text) >= self.sequence_length:
            current_sequence = np.array([[char_to_idx.get(c, 0) for c in seed_text[-self.sequence_length:]]])
        else:
            # Pad with zeros if seed text is shorter than sequence length
            padding = [0] * (self.sequence_length - len(seed_text))
            current_sequence = np.array([padding + [char_to_idx.get(c, 0) for c in seed_text]])
        
        for _ in range(length):
            # Get predictions
            y_pred, _, _ = self.forward(current_sequence)
            
            # Get the last prediction
            last_pred = y_pred[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                last_pred = np.log(last_pred) / temperature
                last_pred = self.softmax(last_pred.reshape(1, -1)).flatten()
            
            # Sample next character
            next_char_idx = np.random.choice(len(last_pred), p=last_pred)
            next_char = idx_to_char[next_char_idx]
            
            generated_text += next_char
            
            # Update sequence (shift and add new character)
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = next_char_idx
        
        return generated_text
    
    def save(self, filepath):
        """Save model parameters"""
        # Convert to CPU arrays for saving
        embedding_weights = to_cpu_array(self.embedding_weights)
        hidden_weights = to_cpu_array(self.hidden_weights)
        output_weights = to_cpu_array(self.output_weights)
        hidden_bias = to_cpu_array(self.hidden_bias)
        output_bias = to_cpu_array(self.output_bias)
        
        np.savez(filepath, 
                 embedding_weights=embedding_weights,
                 hidden_weights=hidden_weights,
                 output_weights=output_weights,
                 hidden_bias=hidden_bias,
                 output_bias=output_bias,
                 vocab_size=self.vocab_size,
                 embedding_dim=self.embedding_dim,
                 hidden_dim=self.hidden_dim)
    
    def load(self, filepath):
        """Load model parameters"""
        data = np.load(filepath)
        
        # Load and convert to appropriate device
        self.embedding_weights = to_gpu_array(data['embedding_weights'], device=self.device).data
        self.hidden_weights = to_gpu_array(data['hidden_weights'], device=self.device).data
        self.output_weights = to_gpu_array(data['output_weights'], device=self.device).data
        self.hidden_bias = to_gpu_array(data['hidden_bias'], device=self.device).data
        self.output_bias = to_gpu_array(data['output_bias'], device=self.device).data
        
        self.vocab_size = int(data['vocab_size'])
        self.embedding_dim = int(data['embedding_dim'])
        self.hidden_dim = int(data['hidden_dim'])
