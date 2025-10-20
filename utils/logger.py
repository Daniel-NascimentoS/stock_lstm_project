"""
Logging utilities for training metrics and events
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path

class TrainingLogger:
    """Handles logging of training metrics and events"""
    
    def __init__(self, log_dir, experiment_name="experiment"):
        """
        Initialize training logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = self._setup_logger()
        
        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
        
    def _setup_logger(self):
        """Setup Python logging"""
        # Create unique logger name to avoid conflicts
        logger_name = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        # Remove existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger
    
    def log_info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_epoch_start(self, epoch, total_epochs):
        """Log epoch start"""
        self.log_info(f"Starting Epoch {epoch}/{total_epochs}")
    
    def log_epoch_end(self, epoch, train_loss, val_loss, epoch_time):
        """Log epoch end"""
        self.log_info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )
    
    def log_training_metrics(self, epoch, step, total_steps, metrics):
        """
        Log training metrics at specific intervals
        
        Args:
            epoch: Current epoch
            step: Current step
            total_steps: Total steps in epoch
            metrics: Dictionary of metrics to log
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'total_steps': total_steps,
            'progress_pct': (step / total_steps) * 100,
            **metrics
        }
        
        self.train_metrics.append(log_entry)
        
        # Save to JSON file
        metrics_file = self.log_dir / f"train_metrics_epoch_{epoch}.json"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_validation_metrics(self, epoch, metrics):
        """
        Log validation metrics
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of validation metrics
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            **metrics
        }
        
        self.val_metrics.append(log_entry)
        
        # Save to JSON file
        metrics_file = self.log_dir / "val_metrics.json"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def save_hyperparameters(self, hyperparams):
        """Save hyperparameters to file"""
        hyperparams_file = self.log_dir / "hyperparameters.json"
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        
        self.log_info(f"Hyperparameters saved to {hyperparams_file}")
    
    def get_train_metrics(self):
        """Get all training metrics"""
        return self.train_metrics
    
    def get_val_metrics(self):
        """Get all validation metrics"""
        return self.val_metrics

