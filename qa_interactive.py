#!/usr/bin/env python3
"""
Interactive QA Text Generation Script
"""

import numpy as np
import json
import os
from model.architecture import TextGenerationModel

def load_qa_model_and_mappings():
    """Load the trained QA model and character mappings"""
    try:
        # Load model
        model = TextGenerationModel(vocab_size=50, embedding_dim=64, hidden_dim=128, sequence_length=50, device='auto')
        model.load("model/qa_text_generation_model.npz")
        
        # Load character mappings
        mappings = np.load("model/qa_char_mappings.npz", allow_pickle=True)
        char_to_idx = mappings['char_to_idx'].item()
        idx_to_char = mappings['idx_to_char'].item()
        
        # Load performance metrics
        with open("model/qa_performance.json", 'r') as f:
            performance = json.load(f)
        
        return model, char_to_idx, idx_to_char, performance
    except FileNotFoundError:
        print("❌ QA model files not found. Please run 'python train_qa_model.py' first to train the model.")
        return None, None, None, None

def generate_qa_response(model, char_to_idx, idx_to_char, question, temperature=0.7, max_length=200):
    """Generate a QA response"""
    try:
        # Format question as JSON
        formatted_question = f'{{"question": "{question}", "answer": "'
        
        generated = model.generate_text(
            formatted_question, 
            char_to_idx, 
            idx_to_char, 
            length=max_length,
            temperature=temperature
        )
        
        # Extract just the answer part
        if '"answer": "' in generated:
            answer_start = generated.find('"answer": "') + len('"answer": "')
            answer_end = generated.find('"', answer_start)
            if answer_end != -1:
                answer = generated[answer_start:answer_end]
                return answer
            else:
                # If no closing quote, take everything after "answer": "
                answer = generated[answer_start:]
                return answer
        else:
            return generated
        
    except Exception as e:
        return f"Error generating response: {e}"

def qa_interactive_mode():
    """Interactive QA mode"""
    print("🤖 QA Text Generation Model - Interactive Mode")
    print("=" * 60)
    
    # Load model
    model, char_to_idx, idx_to_char, performance = load_qa_model_and_mappings()
    if model is None:
        return
    
    print("✅ QA Model loaded successfully!")
    print(f"📊 Vocabulary size: {len(char_to_idx)} characters")
    if performance:
        print(f"📊 Model perplexity: {performance['perplexity']:.4f}")
        print(f"📊 Model accuracy: {performance['accuracy']:.4f}")
    print()
    
    print("💡 Tips:")
    print("- Ask questions about machine learning, AI, and related topics")
    print("- Try different temperatures: 0.5 (conservative) to 1.2 (creative)")
    print("- Type 'quit' to exit")
    print()
    
    while True:
        print("\n" + "-" * 60)
        print("Ask a question:")
        question = input("> ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            print("⚠️  Please enter a question.")
            continue
        
        # Get generation parameters
        try:
            temperature = float(input("Temperature (0.5-1.2, default: 0.7): ") or "0.7")
            max_length = int(input("Max answer length (50-300, default: 150): ") or "150")
        except ValueError:
            print("⚠️  Using default values: temperature=0.7, max_length=150")
            temperature = 0.7
            max_length = 150
        
        # Generate response
        print(f"\n🎯 Generating answer with temperature {temperature}...")
        print(f"❓ Question: '{question}'")
        
        try:
            answer = generate_qa_response(model, char_to_idx, idx_to_char, question, temperature, max_length)
            print(f"🤖 Answer: '{answer}'")
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
        
        print("\n" + "=" * 60)

def qa_demo_mode():
    """Demo mode with predefined questions"""
    print("🎲 QA Model Demo Mode")
    print("=" * 60)
    
    # Load model
    model, char_to_idx, idx_to_char, performance = load_qa_model_and_mappings()
    if model is None:
        return
    
    print("✅ QA Model loaded successfully!")
    if performance:
        print(f"📊 Model performance: Perplexity={performance['perplexity']:.4f}, Accuracy={performance['accuracy']:.4f}")
    print()
    
    # Predefined questions
    demo_questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is the difference between AI and ML?",
        "What is neural network?",
        "What is overfitting?",
        "What is gradient descent?",
        "What is natural language processing?",
        "What is supervised learning?"
    ]
    
    temperatures = [0.5, 0.7, 1.0]
    
    for question in demo_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        for temp in temperatures:
            try:
                answer = generate_qa_response(model, char_to_idx, idx_to_char, question, temp, 120)
                print(f"Temperature {temp}: {answer}")
            except Exception as e:
                print(f"Temperature {temp}: Error - {e}")
        print()

def show_model_info():
    """Show detailed model information"""
    print("📊 QA Model Information")
    print("=" * 60)
    
    # Load model
    model, char_to_idx, idx_to_char, performance = load_qa_model_and_mappings()
    if model is None:
        return
    
    print(f"✅ Model Architecture:")
    print(f"   - Vocabulary size: {model.vocab_size}")
    print(f"   - Embedding dimension: {model.embedding_dim}")
    print(f"   - Hidden dimension: {model.hidden_dim}")
    print(f"   - Sequence length: {model.sequence_length}")
    print(f"   - Learning rate: {model.learning_rate}")
    
    if performance:
        print(f"\n📈 Performance Metrics:")
        print(f"   - Perplexity: {performance['perplexity']:.4f}")
        print(f"   - Accuracy: {performance['accuracy']:.4f}")
        print(f"   - Test sequences: {performance['test_sequences']}")
    
    print(f"\n🔤 Vocabulary (first 20 chars): {sorted(list(char_to_idx.keys()))[:20]}")
    print(f"📁 Model files:")
    print(f"   - Model: model/qa_text_generation_model.npz")
    print(f"   - Mappings: model/qa_char_mappings.npz")
    print(f"   - Performance: model/qa_performance.json")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive QA mode")
    print("2. Demo mode (predefined questions)")
    print("3. Show model information")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        qa_interactive_mode()
    elif choice == "2":
        qa_demo_mode()
    elif choice == "3":
        show_model_info()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Running interactive mode...")
        qa_interactive_mode()
