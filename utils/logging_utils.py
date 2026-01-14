"""
Logging utilities for enterprise-grade monitoring
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler with UTF-8 encoding
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger



def log_metrics(metrics_dict, logger_name="default", level="INFO"):
    """
    Log metrics dictionary
    
    Args:
        metrics_dict: Dictionary of metrics to log
        logger_name: Name of logger
        level: Log level
    """
    logger = setup_logger(logger_name)
    
    if level.upper() == "INFO":
        log_func = logger.info
    elif level.upper() == "WARNING":
        log_func = logger.warning
    elif level.upper() == "ERROR":
        log_func = logger.error
    else:
        log_func = logger.info
    
    # Format metrics
    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
    log_func(f"METRICS: {metrics_str}")

class TrainingLogger:
    """Structured logger for training metrics"""
    
    def __init__(self, log_dir="reports/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / "training_metrics.json"
        self.metrics = []
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log epoch metrics"""
        
        epoch_log = {
            'epoch': epoch,
            'timestamp': str(datetime.now()),
            'train': train_metrics,
            'validation': val_metrics
        }
        
        self.metrics.append(epoch_log)
        
        # Save to file
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_training_end(self, final_metrics: dict):
        """Log final training results"""
        
        summary = {
            'training_completed': str(datetime.now()),
            'final_metrics': final_metrics,
            'total_epochs': len(self.metrics)
        }
        
        summary_file = self.log_dir / "training_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
