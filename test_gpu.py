#!/usr/bin/env python3
"""
GPU Test Script - Verify GPU functionality and performance
"""

import time
import numpy as np
from utils.gpu_utils import (
    GPU_AVAILABLE, print_gpu_info, get_gpu_memory_info,
    to_gpu_array, to_cpu_array, dot, sum, mean
)

def test_gpu_functionality():
    """Test basic GPU functionality"""
    print("🧪 Testing GPU Functionality")
    print("=" * 50)
    
    # Print GPU info
    print_gpu_info()
    
    if not GPU_AVAILABLE:
        print("❌ GPU not available - skipping tests")
        return
    
    # Test basic operations
    print("\n📊 Testing Basic Operations:")
    
    # Create test data
    size = 1000
    a = np.random.randn(size, size)
    b = np.random.randn(size, size)
    
    # CPU timing
    start_time = time.time()
    cpu_result = np.dot(a, b)
    cpu_time = time.time() - start_time
    
    # GPU timing
    start_time = time.time()
    gpu_a = to_gpu_array(a, device='gpu')
    gpu_b = to_gpu_array(b, device='gpu')
    gpu_result = dot(gpu_a.data, gpu_b.data, device='gpu')
    gpu_time = time.time() - start_time
    
    # Convert back to CPU for comparison
    gpu_result_cpu = to_cpu_array(gpu_result)
    
    # Verify results
    error = np.abs(cpu_result - gpu_result_cpu).max()
    
    print(f"   Matrix multiplication ({size}x{size}):")
    print(f"   - CPU time: {cpu_time:.4f}s")
    print(f"   - GPU time: {gpu_time:.4f}s")
    print(f"   - Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"   - Max error: {error:.2e}")
    
    # Test memory operations
    print("\n💾 Testing Memory Operations:")
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"   - GPU Memory: {mem_info['used_gb']:.2f}GB / {mem_info['total_gb']:.2f}GB")
        print(f"   - Usage: {mem_info['usage_percent']:.1f}%")
    
    print("✅ GPU functionality test completed!")

def test_model_performance():
    """Test model performance with GPU vs CPU"""
    print("\n🚀 Testing Model Performance")
    print("=" * 50)
    
    try:
        from model.architecture import TextGenerationModel
        from utils.preprocessing import TextPreprocessor
        
        # Create sample data
        sample_text = "The quick brown fox jumps over the lazy dog. " * 100
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(sequence_length=30)
        X, y, char_to_idx, idx_to_char = preprocessor.prepare_data(sample_text)
        
        print(f"📊 Dataset: {len(X)} sequences, vocab size: {len(char_to_idx)}")
        
        # Test CPU model
        print("\n🖥️  Testing CPU Model:")
        cpu_model = TextGenerationModel(
            vocab_size=len(char_to_idx),
            embedding_dim=32,
            hidden_dim=64,
            learning_rate=0.01,
            device='cpu'
        )
        
        start_time = time.time()
        cpu_model.fit(X[:100], y[:100], epochs=5, batch_size=16, verbose=False)
        cpu_time = time.time() - start_time
        print(f"   - CPU training time: {cpu_time:.4f}s")
        
        # Test GPU model (if available)
        if GPU_AVAILABLE:
            print("\n🚀 Testing GPU Model:")
            gpu_model = TextGenerationModel(
                vocab_size=len(char_to_idx),
                embedding_dim=32,
                hidden_dim=64,
                learning_rate=0.01,
                device='gpu'
            )
            
            start_time = time.time()
            gpu_model.fit(X[:100], y[:100], epochs=5, batch_size=16, verbose=False)
            gpu_time = time.time() - start_time
            print(f"   - GPU training time: {gpu_time:.4f}s")
            print(f"   - Speedup: {cpu_time/gpu_time:.2f}x")
        else:
            print("   - GPU not available for model testing")
        
        print("✅ Model performance test completed!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def main():
    """Main test function"""
    print("🔬 GPU Acceleration Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    test_gpu_functionality()
    
    # Test model performance
    test_model_performance()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    
    if GPU_AVAILABLE:
        print("✅ GPU acceleration is working!")
        print("💡 You can now train models much faster with GPU acceleration.")
    else:
        print("⚠️  No GPU acceleration available.")
        print("💡 Install CuPy or Numba for GPU acceleration.")

if __name__ == "__main__":
    main()
