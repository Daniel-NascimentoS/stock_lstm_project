"""
Custom progress bar utilities using tqdm  with colored output.
"""
from tqdm import tqdm
from colorama import Fore, Style, init
import time

init(autoreset=True)

class ColoredProgress:
    """Handles colored terminal output formatting"""
    
    @staticmethod
    def format_time(seconds):
        """Format seconds to human-readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    @staticmethod
    def format_loss(loss, color=Fore.CYAN):
        """Format loss value with color"""
        return f"{color}{loss:.6f}{Style.RESET_ALL}"
    
    @staticmethod
    def format_metric(value, unit="", color=Fore.GREEN):
        """Format generic metric with color"""
        return f"{color}{value}{unit}{Style.RESET_ALL}"
    
    @staticmethod
    def print_header(text, char="=", width=60):
        """Print colored header"""
        print(f"\n{Fore.YELLOW}{char*width}")
        print(f"{text}")
        print(f"{char*width}{Style.RESET_ALL}\n")
    
    @staticmethod
    def print_success(text):
        """Print success message"""
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def print_info(text):
        """Print info message"""
        print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def print_warning(text):
        """Print warning message"""
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")
    
    @staticmethod
    def print_error(text):
        """Print error message"""
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


class TrainingProgressBar:
    """Custom tqdm progress bar for training"""
    
    def __init__(self, dataloader, epoch, total_epochs, desc="Training"):
        self.dataloader = dataloader
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.desc = desc
        self.pbar = None
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.pbar = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc=f"{Fore.YELLOW}Epoch {self.epoch}/{self.total_epochs}{Style.RESET_ALL}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
        return False
    
    def update_metrics(self, pbar_obj, batch_idx, metrics):
        """Update progress bar with colored metrics"""
        colored_metrics = {}
        
        if 'loss' in metrics:
            colored_metrics['loss'] = ColoredProgress.format_loss(metrics['loss'], Fore.CYAN)
        
        if 'step_time' in metrics:
            step_time_ms = metrics['step_time'] * 1000
            colored_metrics['step_time'] = f"{Fore.MAGENTA}{step_time_ms:.1f}ms{Style.RESET_ALL}"
            colored_metrics['it/s'] = f"{Fore.GREEN}{1/metrics['step_time']:.2f}{Style.RESET_ALL}"
        
        if 'lr' in metrics:
            colored_metrics['lr'] = f"{Fore.BLUE}{metrics['lr']:.6f}{Style.RESET_ALL}"
        
        pbar_obj.set_postfix(colored_metrics)


class ValidationProgressBar:
    """Custom tqdm progress bar for validation"""
    
    def __init__(self, dataloader, desc="Validating"):
        self.dataloader = dataloader
        self.desc = desc
        self.pbar = None
    
    def __enter__(self):
        self.pbar = tqdm(
            self.dataloader,
            desc=f"{Fore.CYAN}{self.desc}{Style.RESET_ALL}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        )
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
        return False