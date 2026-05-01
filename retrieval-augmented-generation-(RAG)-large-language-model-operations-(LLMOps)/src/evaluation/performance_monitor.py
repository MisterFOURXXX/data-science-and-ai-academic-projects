import time
import psutil
import torch
from typing import Dict, Any, Callable
from functools import wraps

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process()
    
    def get_cpu_usage(self) -> float:
        return self.process.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024
        }
    
    def get_gpu_usage(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {'gpu_memory_mb': 0, 'gpu_utilization': 0}
        
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        return {
            'gpu_memory_mb': gpu_memory,
            'gpu_reserved_mb': gpu_reserved,
            'gpu_free_mb': (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024) - gpu_memory
        }
    
    def monitor(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cpu_start = self.get_cpu_usage()
            mem_start = self.get_memory_usage()
            gpu_start = self.get_gpu_usage()
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            cpu_end = self.get_cpu_usage()
            mem_end = self.get_memory_usage()
            gpu_end = self.get_gpu_usage()
            
            metrics = {
                'execution_time_s': end_time - start_time,
                'cpu_usage_percent': cpu_end - cpu_start,
                'memory_delta_mb': mem_end['rss_mb'] - mem_start['rss_mb'],
                'peak_memory_mb': max(mem_start['rss_mb'], mem_end['rss_mb']),
                'gpu_memory_delta_mb': gpu_end['gpu_memory_mb'] - gpu_start['gpu_memory_mb'],
                'peak_gpu_memory_mb': gpu_end['gpu_memory_mb']
            }
            
            return result, metrics
        return wrapper
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{prefix}{key}: {value:.4f}")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    @monitor.monitor
    def sample_function():
        time.sleep(0.5)
        return "done"
    
    result, metrics = sample_function()
    monitor.log_metrics(metrics, "Sample: ")