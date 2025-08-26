#!/usr/bin/env python3
"""
Text Generation Model - Character-Level Language Model
"""

import numpy as np
import os
import sys
from model.architecture import TextGenerationModel
from utils.preprocessing import TextPreprocessor
from utils.metrics import perplexity_score
from utils.logger import Logger
from configs.default_config import Config

def load_sample_text():
    """Load sample text for training"""
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text that will be used 
    to train our character-level language model. The model will learn patterns in the text 
    and be able to generate new text that follows similar patterns.
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn and make predictions from data. Deep learning is a subset of machine 
    learning that uses neural networks with multiple layers to model complex patterns.
    
    Natural language processing is a field of artificial intelligence that helps computers 
    understand, interpret, and manipulate human language. Text generation is one of the 
    many applications of natural language processing.
    
    The future of artificial intelligence is bright and full of possibilities. As we 
    continue to develop more sophisticated models and algorithms, we will unlock new 
    capabilities and applications that we can only imagine today.
    
    Artificial intelligence has the potential to revolutionize many industries including 
    healthcare, finance, transportation, and education. Machine learning algorithms can 
    analyze vast amounts of data to identify patterns and make predictions that would be 
    impossible for humans to detect manually.
    
    Deep learning models have achieved remarkable success in image recognition, natural 
    language processing, and speech recognition. These models use multiple layers of 
    artificial neurons to learn hierarchical representations of data.
    
    The development of large language models has opened new possibilities for text 
    generation, translation, and understanding. These models can generate human-like text 
    and perform various language tasks with impressive accuracy.
    
    As we move forward, the integration of artificial intelligence into everyday 
    applications will become more seamless and intuitive. Smart assistants, autonomous 
    vehicles, and intelligent systems will become commonplace in our daily lives.
    
    The ethical implications of artificial intelligence must be carefully considered as 
    these technologies become more powerful and widespread. We must ensure that AI systems 
    are developed and deployed responsibly for the benefit of humanity.
    """
    return sample_text

def main():
    """Main training and text generation function"""
    logger = Logger()
    logger.log("Starting Text Generation Model Training")
    
    # Load configuration
    config = Config()
    
    # Load and preprocess text data
    logger.log("Loading and preprocessing text data...")
    text_data = load_sample_text()
    
    # Initialize text preprocessor
    preprocessor = TextPreprocessor(sequence_length=config.sequence_length)
    X, y, char_to_idx, idx_to_char = preprocessor.prepare_data(text_data)
    
    logger.log(f"Vocabulary size: {len(char_to_idx)} characters")
    logger.log(f"Total sequences: {len(X)}")
    logger.log(f"Sequence length: {config.sequence_length}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.log(f"Training set: {len(X_train)} sequences")
    logger.log(f"Test set: {len(X_test)} sequences")
    
    # Initialize and train model
    model = TextGenerationModel(
        vocab_size=len(char_to_idx),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        sequence_length=config.sequence_length
    )
    
    logger.log("Training text generation model...")
    history = model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    perplexity = perplexity_score(y_test, y_pred)
    
    logger.log(f"Test Perplexity: {perplexity:.4f}")
    
    # Save model and mappings
    model.save("model/text_generation_model.npz")
    np.savez("model/char_mappings.npz", 
             char_to_idx=char_to_idx, 
             idx_to_char=idx_to_char)
    logger.log("Model and character mappings saved successfully")
    
    # Generate sample text
    logger.log("Generating sample text...")
    seed_text = "The future of"
    generated_text = model.generate_text(
        seed_text, 
        char_to_idx, 
        idx_to_char, 
        length=200,
        temperature=0.8
    )
    
    logger.log(f"Seed text: '{seed_text}'")
    logger.log(f"Generated text: '{generated_text}'")
    
    # Generate multiple samples with different temperatures
    logger.log("Generating text samples with different creativity levels...")
    
    temperatures = [0.5, 0.8, 1.2]
    for temp in temperatures:
        sample = model.generate_text(
            "AI will", 
            char_to_idx, 
            idx_to_char, 
            length=100,
            temperature=temp
        )
        logger.log(f"Temperature {temp}: '{sample}'")
    
    logger.log("Text generation training completed successfully!")

if __name__ == "__main__":
    main()
