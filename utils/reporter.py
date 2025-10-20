"""
Report generation utilities for training results.
"""
import os
import json
from pathlib import Path
from datetime import datetime  

class TrainingReporter:
    """Generates training reports and summaries."""

    def __init__(self, report_dir):
        """
        Itinialize reporter.
        
        Args:
            report_dir: Directory to save reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.reports = []
        self.report_file = self.report_dir / "training_report.json"

        # Load existing reports if available
        if self.report_file.exists():
            with open(self.report_file, 'r') as f:
                self.reports = json.load(f)

    def add_epoch_report(self, epoch, train_loss, val_loss, best_val_loss, epoch_time):
        """
        Add epoch report
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            best_val_loss: Best validation loss so far
            epoch_time: Time taken for epoch
        """
        report = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        }

        self.reports.append(report)
        self._save_reports()

    def generate_summary(self):
        """Generate summary report"""
        if not self.reports:
            return None

        summary = {
            'total_epochs': len(self.reports),
            'best_epoch': min(self.reports, key=lambda x: x['val_loss'])['epoch'],
            'best_train_loss': min(r['train_loss'] for r in self.reports),
            'best_val_loss': min(r['val_loss'] for r in self.reports),
            'final_train_loss': self.reports[-1]['train_loss'],
            'final_val_loss': self.reports[-1]['val_loss'],
            'total_time': sum(r['epoch_time'] for r in self.reports),
            'avg_epoch_time': sum(r['epoch_time'] for r in self.reports) / len(self.reports)
            }
        
        return summary
    
    def save_summary(self):
        """Save summary report to file"""
        summary = self.generate_summary()

        if summary:
            summary_file = self.report_dir / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)

            # Also save as redable text
            text_file = self.report_dir / "training_summary.txt"
            with open(text_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write("TRAINING SUMMARY\n")
                f.write("="*60 + "\n\n")

                for key, value in summary.items():
                    if 'time' in key.lower():
                        f.write(f"{key}: {value:.2f} seconds\n")
                    elif 'loss' in key.lower():
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

    def _save_reports(self):
        """Save reports to file"""
        with open(self.report_file, 'w') as f:
            json.dump(self.reports, f, indent=4)

    def get_all_reports(self):
        """Get all epoch reports"""
        return self.reports
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.generate_summary()

        if summary:
            print("="*60)
            print("TRAINING SUMMARY")
            print("="*60)

            print(f"Total Epochs: {summary['total_epochs']}")
            print(f"Best Epoch: {summary['best_epoch']}")
            print(f"Best Train Loss: {summary['best_train_loss']:.6f}")
            print(f"Best Val Loss: {summary['best_val_loss']:.6f}")
            print(f"Final Train Loss: {summary['final_train_loss']:.6f}")
            print(f"Final Val Loss: {summary['final_val_loss']:.6f}")
            print(f"Total Time: {summary['total_time']:.2f}s")
            print(f"Avg Epoch Time: {summary['avg_epoch_time']:.2f}s")
            print("=" * 60 + "\n")