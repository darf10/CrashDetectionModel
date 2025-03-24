import os
import numpy as np
import torch
import joblib
from collections import deque
from LSTM_Classifier import LSTMClassifier


class AccidentClassifier:
    """
    Wrapper class for accident classification using vehicle motion patterns with LSTM.
    """

    def __init__(self, model_path=None, input_size=10, hidden_size=64,
                 num_layers=2, dropout=0.2, seq_length=5, feature_scaler_path=None, device=None):
        """
        Initialize the Accident classifier.

        params:
            model_path: Path to saved model. If None, a new model will be initialized.
            input_size: Number of features per time step for LSTM
            hidden_size: Hidden size for LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            seq_length: Number of time steps to consider in sequence
            feature_scaler_path: Path to saved feature scaler (for normalization)
            device: Torch device to use (defaults to CUDA if available)
        """
        self.input_size = input_size
        self.seq_length = seq_length
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize LSTM model
        self.model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=num_layers, dropout=dropout)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        # Load feature scaler if provided
        self.feature_scaler = None
        if feature_scaler_path and os.path.exists(feature_scaler_path):
            self.feature_scaler = joblib.load(feature_scaler_path)

        # Store motion history for sequence-based predictions
        self.motion_history = deque(maxlen=seq_length)

    def predict(self, features):
        """
        Predict accident probability based on motion features.

        params:
            features: Numpy array of shape (n_vehicles, n_features) containing motion features
                     for each tracked vehicle.

        Returns:
            float: Probability of accident (0-1)
        """
        if features is None or len(features) == 0:
            return 0.0

        # Store in motion history
        self.motion_history.append(features)

        # Only make predictions when we have enough history
        if len(self.motion_history) >= self.seq_length:
            # Handle case where we have varying numbers of vehicles across frames
            # by computing statistics across vehicles
            sequence = []
            for frame_features in self.motion_history:
                # Compute statistics across all vehicles in the frame
                frame_stats = np.array([
                    np.mean(frame_features, axis=0),
                    np.max(frame_features, axis=0),
                    np.min(frame_features, axis=0),
                    np.std(frame_features, axis=0)
                ]).flatten()

                sequence.append(frame_stats)

            # Create a batch of size 1 (single prediction)
            # Shape: (1, seq_length, features_per_step)
            sequence = np.array([sequence[-self.seq_length:]])

            # Apply feature scaling if available
            if self.feature_scaler:
                # Reshape to apply scaler, then reshape back
                seq_shape = sequence.shape
                sequence = self.feature_scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(seq_shape)

            # Convert to tensor
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(self.device)

            # Get prediction
            with torch.no_grad():
                prediction = self.model(sequence_tensor)
                return prediction.item()
        else:
            # Not enough history for prediction
            return 0.0

    def train(self, X, y, epochs=50, batch_size=32, learning_rate=0.001, validation_split=0.2):
        """
        Train the LSTM classifier on provided data.

        params:
            X: Training features of shape (n_samples, seq_length, features)
            y: Training labels (0 for normal, 1 for accident)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary of training metrics
        """
        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device).unsqueeze(1)

        # Split into training and validation sets
        dataset_size = len(X_tensor)
        indices = list(range(dataset_size))
        val_split = int(np.floor(validation_split * dataset_size))

        # Shuffle indices
        np.random.shuffle(indices)
        train_indices, val_indices = indices[val_split:], indices[:val_split]

        # Create samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        # Create dataset and dataloaders
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler
        )
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=val_sampler
        )

        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        # For early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None

        # Training metrics
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Training phase
            total_train_loss = 0
            for inputs, targets in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            self.model.eval()
            total_val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()

                    # Calculate accuracy
                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct / total

            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Back to training mode
            self.model.train()

        # Load best model if early stopping occurred
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Set back to evaluation mode
        self.model.eval()

        return history

    def save(self, model_path, scaler_path=None):
        """
        Save the trained model and feature scaler.

        params:
            model_path: Path to save model
            scaler_path: Path to save feature scaler (if available)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), model_path)

        # Save scaler if available
        if self.feature_scaler and scaler_path:
            joblib.dump(self.feature_scaler, scaler_path)

    def load(self, model_path, scaler_path=None):
        """
        Load a trained model and feature scaler.

        params:
            model_path: Path to saved model
            scaler_path: Path to saved feature scaler
        """
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found")

        if scaler_path and os.path.exists(scaler_path):
            self.feature_scaler = joblib.load(scaler_path)
            print(f"Loaded feature scaler from {scaler_path}")
