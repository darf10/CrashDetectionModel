import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from collections import deque


class LSTMClassifier(nn.Module):
    """
    LSTM-based neural network for classifying car accidents.

    params:
        nn.Module - Base neural network from torch lib
    """


    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM-based accident classifier.

        params:
            input_size: size of input data
            hidden_size: size of hidden layer
            num_layers: number of LSTM layers
            dropout: dropout rate
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, seq_length, hidden_size*2)

        lstm_out = lstm_out[:, -1, :]
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x
