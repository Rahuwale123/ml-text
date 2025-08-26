"""
Configuration Management
"""

import yaml
import os

class Config:
    """Configuration class for the ML model"""
    
    def __init__(self, config_path="configs/default_config.yaml"):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set attributes from config
            for key, value in config.items():
                setattr(self, key, value)
        else:
            # Default values if config file doesn't exist
            self.sequence_length = 50
            self.test_size = 0.2
            self.random_state = 42
            self.embedding_dim = 64
            self.hidden_dim = 128
            self.learning_rate = 0.01
            self.epochs = 100
            self.batch_size = 32
            self.temperature = 0.8
            self.generation_length = 200
            self.verbose = True
            self.save_model = True
            self.model_path = "model/text_generation_model.npz"
            self.log_file = "training.log"
    
    def save_config(self, config_path=None):
        """Save current configuration to YAML file"""
        if config_path is None:
            config_path = self.config_path
        
        config_dict = {
            'sequence_length': self.sequence_length,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'generation_length': self.generation_length,
            'verbose': self.verbose,
            'save_model': self.save_model,
            'model_path': self.model_path,
            'log_file': self.log_file
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
