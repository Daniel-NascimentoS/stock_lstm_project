import torch
import argparse
import glob
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.model import StockLSTM
from scripts.dataset import create_datasets_with_scaler
from torch.utils.data import DataLoader
from utils.visualizer import TrainingVisualizer
from utils.progress import ColoredProgress
from utils.scaler import TimeSeriesScaler

def evaluate_model(model, dataloader, device, scaler):
    """
    Evaluate model and collect predictes

    Returns:
        actual and predicted arrays
    """
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            if len(data.shape) == 2:
                data = data.unsqueeze(-1)

            output = model(data)

            # Store predictions and actuals
            all_predictions.extend(output.squeeze().cpu().numpy())
            all_actuals.extend(target.cpu().numpy())

    # Convert to numpy arrays 
    predicted = np.array(all_predictions)
    actual = np.array(all_actuals)

    # Inverse transform to original scale
    if scaler is not None:
        predicted = scaler.inverse_transform(predicted)
        actual = scaler.inverse_transform(actual)

    return actual, predicted


def main():
    parser = argparse.ArgumentParser(description='Evaluate LSTM model')
    parser.add_argument('--checkpoint', type=str, default='checkpoins/best_model.pth')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    ColoredProgress.print_header("Model Evaluation")

    # Load Checkpoint
    ColoredProgress.print_info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = checkpoint['args']

    # Load Scaler
    checkpoint_dir = Path(args.checkpoint).parent
    scaler_path = checkpoint_dir / 'scaler.pkl'
    scaler = TimeSeriesScaler()
    scaler.load(scaler_path)
    ColoredProgress.print_success("Scaler loaded")

    # Load data
    parquet_files = sorted(glob.glob(f"{args.data_dir}/*.parquet"))
    data_dict = create_datasets_with_scaler(
        parquet_files,
        window = model_args['window'],
        train_ratio=model_args['train_ratio'],
        val_ratio=model_args['val_ratio'],
        scaler_type=model_args.get('scaler_type', 'minmax')
    )

    # Use the scaler from checkpoint instead
    data_dict['scaler'] = scaler

    # Create dataloaders
    train_loader = DataLoader(data_dict['train'], batch_size = args.batch_size, shuffle = False)
    val_loader = DataLoader(data_dict['val'], batch_size = args.batch_size, shuffle = False)
    test_loader = DataLoader(data_dict['train'], batch_size = args.batch_size, shuffle = False)

    # Initialize model
    model = StockLSTM(
        input_size = 1,
        hidden_size = model_args['hidden_size'],
        num_layers = model_args['num_layers'],
        dropout= model_args['dropout']
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    ColoredProgress.print_success('Model Loaded')

    # Evaluate on all splits
    ColoredProgress.print_info("Evaluating on training set ...")
    train_actual, train_pred = evaluate_model(model, train_loader, args.device, scaler)

    ColoredProgress.print_info("Evaluating on validation set ...")
    val_actual, val_pred = evaluate_model(model, val_loader, args.device, scaler)

    ColoredProgress.print_info("Evaluating on test ser ...")
    test_actual, test_pred = evaluate_model(model, test_loader, args.device, scaler)

    # Create visualization
    visualizer = TrainingVisualizer(save_dir=args.output_dir)


    # Plot training losses if available
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        ColoredProgress.print_info('Plotting training history...')
        visualizer.plot_losses(
            checkpoint['train_losses'],
            checkpoint['val_losses'],
            title='Training History'
        )

    # Plot predictions for each split
    ColoredProgress.print_info('Plotting predictions')

    visualizer.plot_predictions(
        train_actual, train_pred,
        save_path=Path(args.output_dir) / 'train_predictions.png',
        title='Training Set: Predictions vs Actual'
    )

    visualizer.plot_predictions(
        val_actual, val_pred,
        save_path=Path(args.output_dir) / 'val_predictions.png',
        title='Validation Set: Predictions vs Actual'
    )

    visualizer.plot_predictions(
        test_actual, test_pred,
        save_path=Path(args.output_dir) / 'test_predictions.png',
        title='Test Set: Predictions vs Actual'
    )

    # Plot errors distributions
    ColoredProgress.print_info('Plotting error distributions...')
    visualizer.plot_prediction_errors(
        test_actual, test_pred,
        save_path=Path(args.output_dir) / 'test_erro.png'
    )

    # Create compreensive dashboard
    ColoredProgress.print_info('Creating evaluation dashboard...')
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        visualizer.plot_combined_dashboard(
            checkpoint['train_losses'],
            checkpoint['val_losses'],
            test_actual,
            test_pred,
            save_path=Path(args.output_dir) / 'evaluation_dashboard.png'
        )
    
    # Print metrics
    ColoredProgress.print_header('Evaluation Metrics')

    for split_name, actual, pred in [('Train', train_actual, train_pred),
                                     ('Val', val_actual, val_pred),
                                     ('Test', test_actual, test_pred)]:
        
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100

        print(f"\n{split_name} Set:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")

    ColoredProgress.print_success(f"\nAll plots saved  to {args.output_dir}/")

if __name__ == '__main__':
    main()