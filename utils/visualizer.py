# utils/visualizer.py (fix the plot_losses method)
"""
Visualization utilities for training results and predictions
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

class TrainingVisualizer:
    """Handles visualization of training metrics and predictions"""
    
    def __init__(self, save_dir='plots'):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_losses(self, train_losses, val_losses, save_path=None, title='Training History'):
        """
        Plot training and validation losses
        
        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            save_path: Path to save plot (if None, auto-generates)
            title: Plot title
        """
        epochs = range(1, len(train_losses) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot losses
        ax.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=6)
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add min loss annotations - FIXED
        min_train_idx = np.argmin(train_losses)
        min_val_idx = np.argmin(val_losses)
        
        ax.annotate(f'Min Train: {train_losses[min_train_idx]:.6f}',
                   xy=(min_train_idx + 1, train_losses[min_train_idx]),
                   xytext=(10, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))  # FIXED: connectionstyle not connections
        
        ax.annotate(f'Min Val: {val_losses[min_val_idx]:.6f}',
                   xy=(min_val_idx + 1, val_losses[min_val_idx]),
                   xytext=(10, -30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))  # FIXED: connectionstyle not connections
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / 'training_losses.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss plot saved to {save_path}")
        plt.close()
    
    def plot_predictions(self, actual, predicted, save_path=None, 
                        title='Predictions vs Actual', max_points=500):
        """
        Plot predicted vs actual values
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            save_path: Path to save plot
            title: Plot title
            max_points: Maximum number of points to plot (for readability)
        """
        # Limit points for visualization
        if len(actual) > max_points:
            indices = np.linspace(0, len(actual) - 1, max_points, dtype=int)
            actual = actual[indices]
            predicted = predicted[indices]
        else:
            indices = np.arange(len(actual))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Time series comparison
        ax1.plot(indices, actual, 'b-', label='Actual', linewidth=1.5, alpha=0.7)
        ax1.plot(indices, predicted, 'r--', label='Predicted', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot (predicted vs actual)
        ax2.scatter(actual, predicted, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', 
                label='Perfect Prediction', linewidth=2)
        
        ax2.set_xlabel('Actual Price', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Predicted Price', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Calculate and display metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%'
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / 'predictions.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Predictions plot saved to {save_path}")
        plt.close()
    
    def plot_prediction_errors(self, actual, predicted, save_path=None):
        """
        Plot prediction error distribution
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            save_path: Path to save plot
        """
        errors = predicted - actual
        error_pct = (errors / actual) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error distribution histogram
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Percentage error distribution
        ax2.hist(error_pct, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax2.set_xlabel('Prediction Error (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / 'prediction_errors.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error distribution plot saved to {save_path}")
        plt.close()
    
    def plot_combined_dashboard(self, train_losses, val_losses, 
                               actual, predicted, save_path=None):
        """
        Create a comprehensive dashboard with all metrics
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            actual: Array of actual values
            predicted: Array of predicted values
            save_path: Path to save plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, :])
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Training History', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Predictions vs Actual (time series)
        ax2 = fig.add_subplot(gs[1, :])
        max_points = 500
        if len(actual) > max_points:
            indices = np.linspace(0, len(actual) - 1, max_points, dtype=int)
            actual_plot = actual[indices]
            predicted_plot = predicted[indices]
        else:
            indices = np.arange(len(actual))
            actual_plot = actual
            predicted_plot = predicted
        
        ax2.plot(indices, actual_plot, 'b-', label='Actual', linewidth=1.5, alpha=0.7)
        ax2.plot(indices, predicted_plot, 'r--', label='Predicted', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time Step', fontweight='bold')
        ax2.set_ylabel('Price', fontweight='bold')
        ax2.set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.scatter(actual, predicted, alpha=0.5, s=20)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        ax3.set_xlabel('Actual', fontweight='bold')
        ax3.set_ylabel('Predicted', fontweight='bold')
        ax3.set_title('Prediction Scatter', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Error distribution
        ax4 = fig.add_subplot(gs[2, 1])
        errors = predicted - actual
        ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Prediction Error', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Add metrics text box
        metrics_text = (f'Metrics:\n'
                       f'MAE: {mae:.4f}\n'
                       f'RMSE: {rmse:.4f}\n'
                       f'MAPE: {mape:.2f}%\n'
                       f'Best Val Loss: {min(val_losses):.6f}')
        
        fig.text(0.98, 0.02, metrics_text, transform=fig.transFigure,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, verticalalignment='bottom', horizontalalignment='right')
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / 'training_dashboard.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved to {save_path}")
        plt.close()
    
    def load_and_plot_from_logs(self, log_dir):
        """
        Load training data from logs and create plots
        
        Args:
            log_dir: Directory containing training logs
        """
        log_dir = Path(log_dir)
        report_file = log_dir / 'training_report.json'
        
        if not report_file.exists():
            print(f"Error: {report_file} not found")
            return
        
        # Load training report
        with open(report_file, 'r') as f:
            reports = json.load(f)
        
        # Extract losses
        train_losses = [r['train_loss'] for r in reports]
        val_losses = [r['val_loss'] for r in reports]
        
        # Plot losses
        self.plot_losses(train_losses, val_losses, 
                        save_path=self.save_dir / 'training_losses.png')
        
        print(f"✓ Loaded and plotted data from {log_dir}")
