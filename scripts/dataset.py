import torch
from torch.utils.data import Dataset, Subset
import polars as pl
import numpy as np

class StockDataset(Dataset):
    def __init__(self, parquet_files, window = 30, feature_cols=["('Close', 'AAPL')"]):
        self.data = pl.concat([pl.read_parquet(f) for f in sorted(parquet_files)])
        self.window = window
        self.feature_cols = feature_cols

        # Extract features as a NumPy array for easier indexing
        self.features = self.data.select(self.feature_cols).to_numpy()

    def __len__(self):
        return len(self.features) - self.window
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.window]
        y = self.features[idx + self.window, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
def create_time_series_splits(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/val/test splits for time series data manteining temporal order.

    Args =
        dataset (Dataset): PyTorch Dataset object.
        train_ratio (float): Proportion of data to use for training (default = 0.7).
        val_ratio (float): Proportion of data to use for validation (default = 0.15).

    Returns =
        dict: Dictionary with 'train', 'val', and 'test' Subset objects.
    """

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Sequential indices
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    splits ={
        'train': Subset(dataset, train_indices),
        'val': Subset(dataset, val_indices),
        'test': Subset(dataset, test_indices)
    }

    return splits