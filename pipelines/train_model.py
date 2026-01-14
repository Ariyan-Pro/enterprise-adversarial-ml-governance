"""
Enterprise-grade training pipeline with full monitoring and validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import json
from pathlib import Path
import time
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base.mnist_cnn import MNIST_CNN
from utils.dataset_utils import load_mnist, create_dataloaders
from utils.model_utils import save_model, evaluate_model, update_registry
from utils.logging_utils import setup_logger

class ModelTrainer:
    """Complete training pipeline with monitoring"""
    
    def __init__(self, config_path="config/training_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup
        self.device = torch.device(self.config['device'])
        self.logger = setup_logger('trainer', 'reports/logs/training.log')
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.get('seed', 42))
        
        self.logger.info(f"Training configuration: {json.dumps(self.config, indent=2)}")
    
    def setup_data(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        
        # Load dataset
        train_set, test_set = load_mnist(
            augment=self.config.get('augment', False)
        )
        
        # Create dataloaders with validation split
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_set, test_set,
            batch_size=self.config['batch_size'],
            val_split=self.config.get('validation_split', 0.1)
        )
        
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def setup_model(self):
        """Initialize model, optimizer, scheduler"""
        self.logger.info("Initializing model...")
        
        # Model
        self.model = MNIST_CNN().to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0)
            )
        elif self.config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        # Learning rate scheduler
        if self.config.get('scheduler', 'none').lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Log model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model initialized with {total_params:,} parameters")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Logging
            if batch_idx % self.config.get('log_frequency', 10) == 0:
                self.logger.debug(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate on validation set"""
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = Path(self.config['save_path']).parent / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = Path(self.config['save_path']).parent / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Log results
            self.logger.info(
                f"Epoch {epoch:03d}/{self.config['epochs']:03d} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Save checkpoint
            if epoch % self.config.get('checkpoint_frequency', 1) == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.get('early_stopping_patience', float('inf')):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Final evaluation
        self.logger.info("Training completed. Running final evaluation...")
        test_metrics = evaluate_model(self.model, self.test_loader, self.device)
        
        total_time = time.time() - start_time
        self.logger.info(f"Total training time: {total_time:.1f}s")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        
        # Save final model
        metadata = {
            'training_config': self.config,
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'training_time': total_time,
            'final_epoch': epoch
        }
        
        save_model(self.model, self.config['save_path'], metadata)
        update_registry('mnist_cnn', self.config['save_path'], metadata)
        
        return test_metrics

def main():
    """Main entry point"""
    trainer = ModelTrainer()
    trainer.setup_data()
    trainer.setup_model()
    results = trainer.train()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Final Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Model saved to: {trainer.config['save_path']}")
    print("="*50)

if __name__ == "__main__":
    main()
