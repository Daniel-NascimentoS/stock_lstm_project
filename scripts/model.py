import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """
    LSTM model for stock price regression
    """
    def __init__(self, input_size = 1, hidden_size = 128, num_layers = 2, dropout = 0.2, output_size = 1):
        """
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of stack LSTM layers
            dropout (float): Dropout rate for regularization
            output_size (int): Number of output features (1 for regression)
        """
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out