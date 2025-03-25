import torch
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for LSTM Accident Classifier"""
    input_size: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seq_length: int = 5
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path: Optional[str] = 'models/accident_classifier.pth'
    scaler_path: Optional[str] = 'models/feature_scaler.joblib'

    def to_dict(self):
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)