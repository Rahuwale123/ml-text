"""
Simple Logging Utility
"""

import time
from datetime import datetime

class Logger:
    """Simple logger for ML training"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.start_time = time.time()
    
    def log(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
    
    def log_metrics(self, metrics_dict):
        """Log metrics in a formatted way"""
        self.log("Metrics:")
        for metric, value in metrics_dict.items():
            self.log(f"  {metric}: {value}")
    
    def log_training_start(self, model_name, epochs, batch_size):
        """Log training start information"""
        self.log(f"Starting training for {model_name}")
        self.log(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    def log_training_end(self):
        """Log training completion with elapsed time"""
        elapsed_time = time.time() - self.start_time
        self.log(f"Training completed in {elapsed_time:.2f} seconds")
    
    def log_epoch(self, epoch, total_epochs, loss):
        """Log epoch information"""
        self.log(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")
