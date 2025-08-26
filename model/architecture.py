"""
Text Generation Model Implementation - Character-Level Language Model
"""

import numpy as np

class TextGenerationModel:
    """Character-level text generation model using a simple neural network"""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, learning_rate=0.01, sequence_length=50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        
        # Model parameters
        self.embedding_weights = None
        self.hidden_weights = None
        self.output_weights = None
        self.hidden_bias = None
        self.output_bias = None
        
        # Training history
        self.history = {'loss': []}
    
    def initialize_parameters(self):
        """Initialize model parameters"""
        # Xavier/Glorot initialization
        self.embedding_weights = np.random.randn(self.vocab_size, self.embedding_dim) * np.sqrt(2.0 / self.vocab_size)
        self.hidden_weights = np.random.randn(self.embedding_dim, self.hidden_dim) * np.sqrt(2.0 / self.embedding_dim)
        self.output_weights = np.random.randn(self.hidden_dim, self.vocab_size) * np.sqrt(2.0 / self.hidden_dim)
        
        self.hidden_bias = np.zeros(self.hidden_dim)
        self.output_bias = np.zeros(self.vocab_size)
    
    def one_hot_encode(self, sequence):
        """Convert sequence of indices to one-hot encoding"""
        batch_size, seq_length = sequence.shape
        one_hot = np.zeros((batch_size, seq_length, self.vocab_size))
        
        for i in range(batch_size):
            for j in range(seq_length):
                one_hot[i, j, sequence[i, j]] = 1
        
        return one_hot
    
    def softmax(self, x):
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        """Forward pass through the network"""
        batch_size, seq_length = X.shape
        
        # One-hot encode input
        X_one_hot = self.one_hot_encode(X)
        
        # Embedding layer
        embedded = np.dot(X_one_hot, self.embedding_weights)  # (batch_size, seq_length, embedding_dim)
        
        # Hidden layer with ReLU activation
        hidden = np.dot(embedded, self.hidden_weights) + self.hidden_bias
        hidden = np.maximum(0, hidden)  # ReLU activation
        
        # Output layer
        output = np.dot(hidden, self.output_weights) + self.output_bias
        
        # Apply softmax to get probabilities
        probabilities = self.softmax(output)
        
        return probabilities, embedded, hidden
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss"""
        # Convert y_true to one-hot
        batch_size, seq_length = y_true.shape
        y_true_one_hot = self.one_hot_encode(y_true)
        
        # Clip probabilities to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1.0)
        
        # Cross-entropy loss
        loss = -np.sum(y_true_one_hot * np.log(y_pred)) / (batch_size * seq_length)
        return loss
    
    def compute_gradients(self, X, y_true, y_pred, embedded, hidden):
        """Compute gradients for all parameters"""
        batch_size, seq_length = X.shape
        
        # Convert y_true to one-hot
        y_true_one_hot = self.one_hot_encode(y_true)
        
        # Gradient of loss with respect to output
        d_output = (y_pred - y_true_one_hot) / (batch_size * seq_length)
        
        # Gradient of loss with respect to hidden layer
        d_hidden = np.dot(d_output, self.output_weights.T)
        
        # Gradient of loss with respect to embedding
        d_embedding = np.dot(d_hidden, self.hidden_weights.T)
        
        # Gradients for weights and biases - reshape for proper matrix multiplication
        d_output_weights = np.dot(hidden.reshape(-1, self.hidden_dim).T, 
                                 d_output.reshape(-1, self.vocab_size))
        d_output_bias = np.sum(d_output, axis=(0, 1))
        
        d_hidden_weights = np.dot(embedded.reshape(-1, self.embedding_dim).T, 
                                 d_hidden.reshape(-1, self.hidden_dim))
        d_hidden_bias = np.sum(d_hidden, axis=(0, 1))
        
        # Gradient for embedding weights
        d_embedding_weights = np.zeros_like(self.embedding_weights)
        X_one_hot = self.one_hot_encode(X)
        for i in range(batch_size):
            for j in range(seq_length):
                d_embedding_weights += np.outer(X_one_hot[i, j], d_embedding[i, j])
        
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
            indices = np.random.permutation(n_samples)
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
            
            # Average loss for the epoch
            avg_loss = epoch_loss / n_batches
            self.history['loss'].append(avg_loss)
            
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
        np.savez(filepath, 
                 embedding_weights=self.embedding_weights,
                 hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_bias=self.hidden_bias,
                 output_bias=self.output_bias,
                 vocab_size=self.vocab_size,
                 embedding_dim=self.embedding_dim,
                 hidden_dim=self.hidden_dim)
    
    def load(self, filepath):
        """Load model parameters"""
        data = np.load(filepath)
        self.embedding_weights = data['embedding_weights']
        self.hidden_weights = data['hidden_weights']
        self.output_weights = data['output_weights']
        self.hidden_bias = data['hidden_bias']
        self.output_bias = data['output_bias']
        self.vocab_size = int(data['vocab_size'])
        self.embedding_dim = int(data['embedding_dim'])
        self.hidden_dim = int(data['hidden_dim'])
