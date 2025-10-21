import torch
from torch.utils.data import Dataset, Subset
import polars as pl
import numpy as np
import sys
from pathlib import Path
from colorama import Fore, Style

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.scaler import TimeSeriesScaler

class StockDataset(Dataset):
    def __init__(self, parquet_files, window=30, feature_cols=['Close'], 
                 scaler=None, fit_scaler=True):
        """
        Stock dataset with normalization support
        
        Args:
            parquet_files: List of parquet files to load
            window: Lookback window size
            feature_cols: List of feature column names
            scaler: TimeSeriesScaler object (if None, creates new one)
            fit_scaler: Whether to fit the scaler on this data
        """
        self.data = pl.concat([pl.read_parquet(f) for f in sorted(parquet_files)])
        self.window = window
        self.feature_cols = feature_cols
        
        # Extract features as numpy array
        self.raw_features = self.data.select(feature_cols).to_numpy()
        
        # Initialize or use provided scaler
        if scaler is None:
            self.scaler = TimeSeriesScaler(scaler_type='minmax', feature_range=(0, 1))
            if fit_scaler:
                self.scaler.fit(self.raw_features)
        else:
            self.scaler = scaler
        
        # Normalize features
        self.features = self.scaler.transform(self.raw_features)
        
    def __len__(self):
        return len(self.features) - self.window
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.window]
        y = self.features[idx+self.window]
        
        # Handle multi-feature case
        if len(self.feature_cols) == 1:
            y = y[0]  # Extract single value for single feature
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def get_scaler(self):
        """Get the scaler used for normalization"""
        return self.scaler
    
    def inverse_transform(self, data):
        """Inverse transform normalized data back to original scale"""
        return self.scaler.inverse_transform(data)


def create_time_series_splits(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Creates train/val/test splits for time series data maintaining temporal order.
    
    Args:
        dataset: PyTorch Dataset object
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
    
    Returns:
        Dictionary with train, val, and test Subset objects
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Sequential indices (no shuffling!)
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    splits = {
        'train': Subset(dataset, train_indices),
        'val': Subset(dataset, val_indices),
        'test': Subset(dataset, test_indices)
    }
    
    return splits


def create_datasets_with_scaler(parquet_files, window=30, train_ratio=0.7, val_ratio=0.15,
                                feature_cols=['Close'], scaler_type='minmax'):
    """
    Create train/val/test datasets with proper scaler fitting
    
    Args:
        parquet_files: List of parquet files
        window: Lookback window
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        feature_cols: List of feature columns
        scaler_type: Type of scaler ('minmax' or 'standard')
    
    Returns:
        Dictionary with train, val, test datasets and fitted scaler
    """
    # Load all data
    all_data = pl.concat([pl.read_parquet(f) for f in sorted(parquet_files)])
    raw_features = all_data.select(feature_cols).to_numpy()
    
    # Calculate split points
    total_size = len(raw_features)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # IMPORTANT: Fit scaler ONLY on training data to avoid data leakage
    train_features = raw_features[:train_size]
    scaler = TimeSeriesScaler(scaler_type=scaler_type, feature_range=(0, 1))
    scaler.fit(train_features)
    
    # Create dataset with fitted scaler (don't refit)
    dataset = StockDataset(
        parquet_files, 
        window=window, 
        feature_cols=feature_cols,
        scaler=scaler,
        fit_scaler=False  # Already fitted on training data
    )
    
    # Create splits
    splits = create_time_series_splits(dataset, train_ratio, val_ratio)
    
    return {
        'train': splits['train'],
        'val': splits['val'],
        'test': splits['test'],
        'scaler': scaler,
        'dataset': dataset
    }
