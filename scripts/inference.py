import torch
import argparse
import glob
import polars as pl
from model import StockLSTM
from colorama import Fore, Style, init
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.scaler import TimeSeriesScaler

init(autoreset=True)

def load_model_and_scaler(checkpoint_path, device):
    """Load trained model and scaler from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model args
    model_args = checkpoint['args']
    
    # Initialize model
    model = StockLSTM(
        input_size=1,
        hidden_size=model_args['hidden_size'],
        num_layers=model_args['num_layers'],
        dropout=model_args['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    checkpoint_dir = Path(checkpoint_path).parent
    scaler_path = checkpoint_dir / 'scaler.pkl'
    
    if scaler_path.exists():
        scaler = TimeSeriesScaler()
        scaler.load(scaler_path)
    else:
        print(f"{Fore.YELLOW}⚠ Scaler not found, predictions will be in normalized scale{Style.RESET_ALL}")
        scaler = None
    
    return model, model_args, scaler

def predict(model, data, device):
    """Make prediction"""
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        prediction = model(data_tensor)
        return prediction.cpu().item()

def main():
    parser = argparse.ArgumentParser(description='LSTM Stock Prediction Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"LSTM Stock Prediction - Inference Mode")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    # Load model and scaler
    print(f"{Fore.YELLOW}Loading model from {args.checkpoint}...{Style.RESET_ALL}")
    model, model_args, scaler = load_model_and_scaler(args.checkpoint, args.device)
    window_size = model_args['window']
    print(f"{Fore.GREEN}✓ Model loaded successfully{Style.RESET_ALL}\n")
    
    # Load data
    parquet_files = sorted(glob.glob(f"{args.data_dir}/*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in parquet_files])
    prices = df['Close'].to_numpy().reshape(-1, 1)
    
    # Normalize data using loaded scaler
    if scaler is not None:
        prices_normalized = scaler.transform(prices).flatten()
        print(f"{Fore.GREEN}✓ Data normalized using saved scaler{Style.RESET_ALL}\n")
    else:
        prices_normalized = prices.flatten()
    
    # Generate predictions
    print(f"{Fore.CYAN}Generating {args.num_examples} predictions...{Style.RESET_ALL}\n")
    
    for i in range(args.num_examples):
        # Get random sequence
        idx = np.random.randint(0, len(prices_normalized) - window_size - 1)
        input_sequence = prices_normalized[idx:idx + window_size]
        actual_price_normalized = prices_normalized[idx + window_size]
        
        # Predict (in normalized space)
        predicted_price_normalized = predict(model, input_sequence, args.device)
        
        # Inverse transform to original scale
        if scaler is not None:
            predicted_price = scaler.inverse_transform(np.array([predicted_price_normalized]))[0]
            actual_price = scaler.inverse_transform(np.array([actual_price_normalized]))[0]
        else:
            predicted_price = predicted_price_normalized
            actual_price = actual_price_normalized
        
        # Calculate error
        error = abs(predicted_price - actual_price)
        error_pct = (error / actual_price) * 100
        
        # Display
        print(f"{Fore.YELLOW}Example {i+1}:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Predicted: ${predicted_price:.2f}{Style.RESET_ALL}")
        print(f"  {Fore.BLUE}Actual: ${actual_price:.2f}{Style.RESET_ALL}")
        print(f"  {Fore.MAGENTA}Error: ${error:.2f} ({error_pct:.2f}%){Style.RESET_ALL}\n")
    
    print(f"{Fore.GREEN}{'='*60}")
    print(f"Inference completed!")
    print(f"{'='*60}{Style.RESET_ALL}\n")

if __name__ == '__main__':
    main()
