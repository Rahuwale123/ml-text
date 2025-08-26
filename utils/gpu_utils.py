"""
GPU Utilities for Accelerated Computing
Provides GPU acceleration with automatic fallback to CPU
"""

import numpy as np
import os

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available (CuPy)")
except ImportError:
    try:
        # Try alternative GPU library
        import numba
        from numba import cuda
        GPU_AVAILABLE = cuda.is_available()
        if GPU_AVAILABLE:
            print("✅ GPU acceleration available (Numba)")
        else:
            print("⚠️  Numba available but no CUDA GPU detected")
    except ImportError:
        GPU_AVAILABLE = False
        print("⚠️  No GPU acceleration available - using CPU")

class GPUArray:
    """Wrapper for GPU/CPU arrays with automatic device management"""
    
    def __init__(self, data, device='auto'):
        self.device = device
        self._data = None
        
        if device == 'auto':
            self.device = 'gpu' if GPU_AVAILABLE else 'cpu'
        
        if self.device == 'gpu' and GPU_AVAILABLE:
            try:
                self._data = cp.asarray(data)
            except Exception as e:
                print(f"⚠️  GPU allocation failed, falling back to CPU: {e}")
                self.device = 'cpu'
                self._data = np.asarray(data)
        else:
            self.device = 'cpu'
            self._data = np.asarray(data)
    
    @property
    def data(self):
        """Get the underlying array data"""
        return self._data
    
    def to_cpu(self):
        """Convert to CPU array"""
        if self.device == 'gpu' and GPU_AVAILABLE:
            return GPUArray(cp.asnumpy(self._data), device='cpu')
        return self
    
    def to_gpu(self):
        """Convert to GPU array"""
        if GPU_AVAILABLE:
            if self.device == 'cpu':
                return GPUArray(cp.asarray(self._data), device='gpu')
            return self
        else:
            print("⚠️  GPU not available, keeping on CPU")
            return self
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying array"""
        return getattr(self._data, name)
    
    def __getitem__(self, key):
        """Array indexing"""
        return GPUArray(self._data[key], device=self.device)
    
    def __setitem__(self, key, value):
        """Array assignment"""
        if isinstance(value, GPUArray):
            self._data[key] = value.data
        else:
            self._data[key] = value
    
    def __len__(self):
        return len(self._data)
    
    def __str__(self):
        return f"GPUArray(device={self.device}, shape={self._data.shape}, dtype={self._data.dtype})"
    
    def __repr__(self):
        return self.__str__()

def get_array_module(device='auto'):
    """Get the appropriate array module (CuPy or NumPy)"""
    if device == 'auto':
        device = 'gpu' if GPU_AVAILABLE else 'cpu'
    
    if device == 'gpu' and GPU_AVAILABLE:
        return cp
    else:
        return np

def to_gpu_array(data, device='auto'):
    """Convert data to GPU array if available"""
    return GPUArray(data, device=device)

def to_cpu_array(data):
    """Convert data to CPU array"""
    if isinstance(data, GPUArray):
        return data.to_cpu().data
    elif GPU_AVAILABLE and hasattr(data, 'get'):  # CuPy array
        return cp.asnumpy(data)
    elif GPU_AVAILABLE and hasattr(data, 'device'):  # CuPy array with device attribute
        return cp.asnumpy(data)
    else:
        return np.asarray(data)

def synchronize_gpu():
    """Synchronize GPU operations"""
    if GPU_AVAILABLE:
        try:
            cp.cuda.Stream.null.synchronize()
        except:
            pass

def get_gpu_memory_info():
    """Get GPU memory information"""
    if GPU_AVAILABLE:
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            free_memory = meminfo[0] / 1024**3  # GB
            total_memory = meminfo[1] / 1024**3  # GB
            used_memory = total_memory - free_memory
            return {
                'free_gb': free_memory,
                'total_gb': total_memory,
                'used_gb': used_memory,
                'usage_percent': (used_memory / total_memory) * 100
            }
        except:
            return None
    return None

def print_gpu_info():
    """Print GPU information"""
    if GPU_AVAILABLE:
        try:
            print("🚀 GPU Information:")
            print(f"   - Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            mem_info = get_gpu_memory_info()
            if mem_info:
                print(f"   - Memory: {mem_info['used_gb']:.2f}GB / {mem_info['total_gb']:.2f}GB ({mem_info['usage_percent']:.1f}%)")
        except Exception as e:
            print(f"   - Error getting GPU info: {e}")
    else:
        print("⚠️  No GPU acceleration available")

# Convenience functions for common operations
def dot(a, b, device='auto'):
    """Matrix multiplication with GPU acceleration"""
    xp = get_array_module(device)
    return xp.dot(a, b)

def sum(array, axis=None, keepdims=False, device='auto'):
    """Sum with GPU acceleration"""
    xp = get_array_module(device)
    return xp.sum(array, axis=axis, keepdims=keepdims)

def mean(array, axis=None, device='auto'):
    """Mean with GPU acceleration"""
    xp = get_array_module(device)
    return xp.mean(array, axis=axis)

def exp(array, device='auto'):
    """Exponential with GPU acceleration"""
    xp = get_array_module(device)
    return xp.exp(array)

def log(array, device='auto'):
    """Logarithm with GPU acceleration"""
    xp = get_array_module(device)
    return xp.log(array)

def random_normal(size, device='auto'):
    """Random normal distribution with GPU acceleration"""
    xp = get_array_module(device)
    return xp.random.normal(size=size)

def random_randn(*args, device='auto'):
    """Random standard normal with GPU acceleration"""
    xp = get_array_module(device)
    return xp.random.randn(*args)

def zeros(shape, dtype=None, device='auto'):
    """Zeros array with GPU acceleration"""
    xp = get_array_module(device)
    return xp.zeros(shape, dtype=dtype)

def ones(shape, dtype=None, device='auto'):
    """Ones array with GPU acceleration"""
    xp = get_array_module(device)
    return xp.ones(shape, dtype=dtype)

def zeros_like(array, device='auto'):
    """Zeros array with same shape as input"""
    xp = get_array_module(device)
    return xp.zeros_like(array)
