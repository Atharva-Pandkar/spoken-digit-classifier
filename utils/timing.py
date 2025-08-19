import time
import functools
from typing import Callable, Any, Tuple

def time_function(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    return wrapper

class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = (self.end_time - self.start_time) * 1000  # milliseconds
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.elapsed_time is None:
            return 0.0
        return self.elapsed_time

def benchmark_function(func: Callable, *args, n_runs: int = 10, **kwargs) -> dict:
    """Benchmark a function over multiple runs"""
    times = []
    results = []
    
    for _ in range(n_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # milliseconds
        times.append(execution_time)
        results.append(result)
    
    import numpy as np
    
    return {
        'times_ms': times,
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'p95_time_ms': np.percentile(times, 95),
        'results': results
    }
