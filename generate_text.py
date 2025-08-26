#!/usr/bin/env python3
"""
Interactive Text Generation Script
"""

import numpy as np
import os
from model.architecture import TextGenerationModel
from utils.preprocessing import TextPreprocessor
from configs.default_config import Config

def load_model_and_mappings():
    """Load the trained model and character mappings"""
    try:
        # Load model
        model = TextGenerationModel(vocab_size=39, embedding_dim=64, hidden_dim=128, sequence_length=50)
        model.load("model/text_generation_model.npz")
        
        # Load character mappings
        mappings = np.load("model/char_mappings.npz", allow_pickle=True)
        char_to_idx = mappings['char_to_idx'].item()
        idx_to_char = mappings['idx_to_char'].item()
        
        return model, char_to_idx, idx_to_char
    except FileNotFoundError:
        print("❌ Model files not found. Please run 'python main.py' first to train the model.")
        return None, None, None

def generate_interactive():
    """Interactive text generation"""
    print("🤖 Text Generation Model - Interactive Mode")
    print("=" * 50)
    
    # Load model
    model, char_to_idx, idx_to_char = load_model_and_mappings()
    if model is None:
        return
    
    print("✅ Model loaded successfully!")
    print(f"📊 Vocabulary size: {len(char_to_idx)} characters")
    print()
    
    while True:
        print("\n" + "-" * 50)
        print("Enter your seed text (or 'quit' to exit):")
        seed_text = input("> ").strip()
        
        if seed_text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not seed_text:
            print("⚠️  Please enter some text to start generation.")
            continue
        
        # Get generation parameters
        try:
            length = int(input("How many characters to generate? (default: 100): ") or "100")
            temperature = float(input("Temperature (0.5-1.5, default: 0.8): ") or "0.8")
        except ValueError:
            print("⚠️  Using default values: length=100, temperature=0.8")
            length = 100
            temperature = 0.8
        
        # Generate text
        print(f"\n🎯 Generating {length} characters with temperature {temperature}...")
        print(f"📝 Seed: '{seed_text}'")
        
        try:
            generated_text = model.generate_text(
                seed_text, 
                char_to_idx, 
                idx_to_char, 
                length=length,
                temperature=temperature
            )
            
            print(f"✨ Generated: '{generated_text}'")
            
            # Show just the new part
            if len(generated_text) > len(seed_text):
                new_part = generated_text[len(seed_text):]
                print(f"🆕 New text: '{new_part}'")
            
        except Exception as e:
            print(f"❌ Error generating text: {e}")
        
        print("\n" + "=" * 50)

def generate_samples():
    """Generate sample texts with different temperatures"""
    print("🎲 Generating Sample Texts")
    print("=" * 50)
    
    # Load model
    model, char_to_idx, idx_to_char = load_model_and_mappings()
    if model is None:
        return
    
    seed_texts = [
        "The future of",
        "AI will",
        "Machine learning",
        "In the year 2030",
        "Artificial intelligence"
    ]
    
    temperatures = [0.5, 0.8, 1.2]
    
    for seed in seed_texts:
        print(f"\n🌱 Seed: '{seed}'")
        for temp in temperatures:
            try:
                generated = model.generate_text(
                    seed, 
                    char_to_idx, 
                    idx_to_char, 
                    length=80,
                    temperature=temp
                )
                new_part = generated[len(seed):]
                print(f"  Temperature {temp}: '{new_part}'")
            except Exception as e:
                print(f"  ❌ Error with temperature {temp}: {e}")
        print()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive text generation")
    print("2. Generate sample texts")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        generate_interactive()
    elif choice == "2":
        generate_samples()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Running interactive mode...")
        generate_interactive()
