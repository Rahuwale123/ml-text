#!/usr/bin/env python3
"""
Train Text Generation Model on JSON Question-Answer Dataset
"""

import numpy as np
import json
import os
import sys
from model.architecture import TextGenerationModel
from utils.preprocessing import TextPreprocessor
from utils.metrics import perplexity_score, accuracy_score
from utils.logger import Logger
from configs.default_config import Config

def load_qa_dataset(filepath):
    """Load and format the QA dataset"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to text format for character-level training
        text_data = ""
        for i, item in enumerate(data):
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            # Format as JSON-like text
            qa_text = f'{{"question": "{question}", "answer": "{answer}"}}'
            text_data += qa_text + "\n"
            
            # Also add just the question-answer pairs for variety
            qa_simple = f"Q: {question}\nA: {answer}\n"
            text_data += qa_simple + "\n"
        
        return text_data, len(data)
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {filepath}")
        return None, 0
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in: {filepath}")
        return None, 0

def evaluate_model_performance(model, X_test, y_test, char_to_idx, idx_to_char):
    """Evaluate model performance with multiple metrics"""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate perplexity
    perplexity = perplexity_score(y_test, y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate sample completions
    sample_questions = [
        '{"question": "What is',
        'Q: How does',
        '{"question": "What are the',
        'Q: What is the difference'
    ]
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    print(f"📊 Test Perplexity: {perplexity:.4f}")
    print(f"📊 Test Accuracy: {accuracy:.4f}")
    print(f"📊 Vocabulary Size: {len(char_to_idx)}")
    print(f"📊 Test Sequences: {len(X_test)}")
    
    print("\n🎯 Sample Text Generation:")
    print("-" * 40)
    
    for i, seed in enumerate(sample_questions, 1):
        try:
            generated = model.generate_text(
                seed, 
                char_to_idx, 
                idx_to_char, 
                length=100,
                temperature=0.8
            )
            print(f"{i}. Seed: '{seed}'")
            print(f"   Generated: '{generated}'")
            print()
        except Exception as e:
            print(f"{i}. Error generating for '{seed}': {e}")
    
    return {
        'perplexity': perplexity,
        'accuracy': accuracy,
        'vocab_size': len(char_to_idx),
        'test_sequences': len(X_test)
    }

def main():
    """Main training function for QA dataset"""
    logger = Logger()
    logger.log("Starting QA Text Generation Model Training")
    
    # Load configuration
    config = Config()
    
    # Load QA dataset
    dataset_path = "data/sample_qa_dataset.json"
    logger.log(f"Loading QA dataset from: {dataset_path}")
    
    text_data, num_qa_pairs = load_qa_dataset(dataset_path)
    if text_data is None:
        logger.log("❌ Failed to load dataset. Exiting.")
        return
    
    logger.log(f"Loaded {num_qa_pairs} QA pairs")
    logger.log(f"Total text length: {len(text_data)} characters")
    
    # Initialize text preprocessor
    preprocessor = TextPreprocessor(sequence_length=config.sequence_length)
    X, y, char_to_idx, idx_to_char = preprocessor.prepare_data(text_data)
    
    logger.log(f"Vocabulary size: {len(char_to_idx)} characters")
    logger.log(f"Total sequences: {len(X)}")
    logger.log(f"Sequence length: {config.sequence_length}")
    
    # Show vocabulary
    logger.log(f"Vocabulary: {sorted(char_to_idx.keys())}")
    
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
        sequence_length=config.sequence_length,
        device='auto'  # Will use GPU if available, otherwise CPU
    )
    
    logger.log("Training QA text generation model...")
    history = model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size)
    
    # Evaluate model performance
    performance = evaluate_model_performance(model, X_test, y_test, char_to_idx, idx_to_char)
    
    # Save model and mappings
    model.save("model/qa_text_generation_model.npz")
    np.savez("model/qa_char_mappings.npz", 
             char_to_idx=char_to_idx, 
             idx_to_char=idx_to_char)
    logger.log("QA model and character mappings saved successfully")
    
    # Save performance metrics
    with open("model/qa_performance.json", 'w') as f:
        json.dump(performance, f, indent=2)
    logger.log("Performance metrics saved to model/qa_performance.json")
    
    # Generate more detailed samples
    logger.log("\n🎲 Generating Detailed Samples:")
    logger.log("-" * 40)
    
    test_prompts = [
        '{"question": "What is machine learning?"',
        'Q: How does deep learning work?',
        '{"question": "What is the difference between',
        'Q: What are the main types of',
        '{"question": "How do neural networks',
        'Q: What is natural language'
    ]
    
    for prompt in test_prompts:
        try:
            generated = model.generate_text(
                prompt, 
                char_to_idx, 
                idx_to_char, 
                length=150,
                temperature=0.7
            )
            logger.log(f"Prompt: '{prompt}'")
            logger.log(f"Generated: '{generated}'")
            logger.log("-" * 30)
        except Exception as e:
            logger.log(f"Error with prompt '{prompt}': {e}")
    
    logger.log("QA text generation training completed successfully!")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Dataset: {num_qa_pairs} QA pairs")
    print(f"✅ Training sequences: {len(X_train)}")
    print(f"✅ Test sequences: {len(X_test)}")
    print(f"✅ Vocabulary size: {performance['vocab_size']}")
    print(f"✅ Final perplexity: {performance['perplexity']:.4f}")
    print(f"✅ Final accuracy: {performance['accuracy']:.4f}")
    print(f"✅ Model saved: model/qa_text_generation_model.npz")
    print("=" * 60)

if __name__ == "__main__":
    main()
