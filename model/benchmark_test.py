"""
Benchmark Test for Text Generation Model
Reports tokens/sec, parameter count, and RAM usage
"""

import numpy as np
import time
import psutil
import os
from architecture import TextGenerationModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import TextPreprocessor

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def count_parameters(model):
    """Count total parameters in the model"""
    total_params = 0
    if model.embedding_weights is not None:
        total_params += model.embedding_weights.size
    if model.hidden_weights is not None:
        total_params += model.hidden_weights.size
    if model.output_weights is not None:
        total_params += model.output_weights.size
    if model.hidden_bias is not None:
        total_params += model.hidden_bias.size
    if model.output_bias is not None:
        total_params += model.output_bias.size
    return total_params

def benchmark_model():
    """Run comprehensive benchmark tests"""
    print("=" * 60)
    print("TEXT GENERATION MODEL BENCHMARK")
    print("=" * 60)
    
    # Sample text for training
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text that will be used 
    to train our character-level language model. The model will learn patterns in the text 
    and be able to generate new text that follows similar patterns.
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn and make predictions from data. Deep learning is a subset of machine 
    learning that uses neural networks with multiple layers to model complex patterns.
    """
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(sequence_length=30)
    X, y, char_to_idx, idx_to_char = preprocessor.prepare_data(sample_text)
    
    # Initialize model
    model = TextGenerationModel(
        vocab_size=len(char_to_idx),
        embedding_dim=32,
        hidden_dim=64,
        learning_rate=0.01,
        device='auto'  # Will use GPU if available
    )
    
    # Count parameters
    model.initialize_parameters()
    param_count = count_parameters(model)
    
    print(f"Model Parameters: {param_count:,}")
    print(f"Vocabulary Size: {len(char_to_idx)}")
    print(f"Sequence Length: {X.shape[1]}")
    print(f"Training Sequences: {X.shape[0]}")
    print()
    
    # Memory usage before training
    initial_memory = get_memory_usage()
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")
    
    # Training benchmark
    print("\nTraining Benchmark:")
    print("-" * 30)
    
    start_time = time.time()
    history = model.fit(X, y, epochs=20, batch_size=16, verbose=False)
    training_time = time.time() - start_time
    
    # Memory usage after training
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    print(f"Training Time: {training_time:.3f} seconds")
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print(f"Memory Increase: {memory_increase:.2f} MB")
    
    # Inference benchmark
    print("\nInference Benchmark:")
    print("-" * 30)
    
    # Warm up
    for _ in range(5):
        _ = model.predict(X[:5])
    
    # Benchmark inference
    n_inference_sequences = 1000
    X_inference = X[:n_inference_sequences]
    
    start_time = time.time()
    predictions = model.predict(X_inference)
    inference_time = time.time() - start_time
    
    # Calculate tokens/sec (each character is a token)
    total_tokens = n_inference_sequences * X.shape[1]
    tokens_per_sec = total_tokens / inference_time
    
    print(f"Inference Sequences: {n_inference_sequences:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Inference Time: {inference_time:.3f} seconds")
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    
    # Text generation benchmark
    print("\nText Generation Benchmark:")
    print("-" * 30)
    
    start_time = time.time()
    generated_text = model.generate_text(
        "The future", 
        char_to_idx, 
        idx_to_char, 
        length=100,
        temperature=0.8
    )
    generation_time = time.time() - start_time
    
    print(f"Generated Text Length: {len(generated_text)} characters")
    print(f"Generation Time: {generation_time:.3f} seconds")
    print(f"Sample Generated Text: '{generated_text[:50]}...'")
    
    # Performance summary
    print("\nPerformance Summary:")
    print("-" * 30)
    print(f"✓ Parameter Count: {param_count:,}")
    print(f"✓ Training Time: {training_time:.3f}s")
    print(f"✓ Memory Usage: {final_memory:.2f} MB")
    print(f"✓ Inference Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"✓ Text Generation: {len(generated_text)} chars in {generation_time:.3f}s")
    
    # Check performance requirements
    print("\nPerformance Requirements Check:")
    print("-" * 40)
    
    if tokens_per_sec >= 500000:
        print("✓ Throughput requirement met: > 500K tokens/sec")
    else:
        print("✗ Throughput requirement not met: < 500K tokens/sec")
    
    if training_time < 600:  # 10 minutes = 600 seconds
        print("✓ Training time requirement met: < 10 minutes")
    else:
        print("✗ Training time requirement not met: > 10 minutes")
    
    if final_memory < 1000:  # 1GB = 1000 MB
        print("✓ Memory usage requirement met: < 1GB")
    else:
        print("✗ Memory usage requirement not met: > 1GB")
    
    print("\n" + "=" * 60)
    
    return {
        'parameters': param_count,
        'training_time': training_time,
        'memory_usage': final_memory,
        'throughput': tokens_per_sec,
        'generation_time': generation_time,
        'meets_requirements': tokens_per_sec >= 500000 and training_time < 600 and final_memory < 1000
    }

if __name__ == "__main__":
    try:
        results = benchmark_model()
        if results['meets_requirements']:
            print("🎉 All performance requirements met!")
        else:
            print("⚠️  Some performance requirements not met")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install psutil: pip install psutil")
    except Exception as e:
        print(f"Benchmark failed: {e}")
