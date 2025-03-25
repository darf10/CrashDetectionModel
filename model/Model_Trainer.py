import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from .LSTM_Classifier import LSTMClassifier
from .Model_Config import ModelConfig


class ModelTrainer:
    def __init__(self, config: ModelConfig = None):
        """
        Initialize model trainer with optional configuration

        Args:
            config: ModelConfig instance, uses default if not provided
        """
        self.config = config or ModelConfig()
        self.logger = self._setup_logger()

        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

        # Initialize model and move to device
        self.model = LSTMClassifier(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.config.device)

        # Prepare optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.criterion = nn.BCELoss()

    def _setup_logger(self):
        """Setup logging for model training"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler('model_training.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def train(self, X, y):
        """
        Train the LSTM model

        Args:
            X: Training features (n_samples, seq_length, features)
            y: Training labels (binary)

        Returns:
            Dictionary of training history
        """
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        joblib.dump(scaler, self.config.scaler_path)

        # Prepare data
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.config.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.config.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        history = {'train_loss': [], 'train_accuracy': []}

        # Training loop
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for batch_x, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Tracking metrics
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == batch_y).sum().item()
                total_predictions += batch_y.size(0)

            # Log epoch results
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions

            history['train_loss'].append(avg_loss)
            history['train_accuracy'].append(accuracy)

            self.logger.info(f'Epoch {epoch + 1}/{self.config.epochs}: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}')

        # Save model
        torch.save(self.model.state_dict(), self.config.model_path)

        return history

    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model.load_state_dict(torch.load(self.config.model_path))
            self.logger.info(f'Model loaded from {self.config.model_path}')
        except FileNotFoundError:
            self.logger.warning(f'No pre-trained model found at {self.config.model_path}')