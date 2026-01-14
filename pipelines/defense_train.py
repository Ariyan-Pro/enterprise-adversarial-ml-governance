"""
Defense Training Pipeline
Enterprise-grade training of defense mechanisms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from defenses.adv_training import AdversarialTraining
from defenses.model_wrappers import ModelDistillationWrapper
from utils.model_utils import load_model, save_model, evaluate_model
from utils.dataset_utils import load_mnist
from utils.visualization import setup_plotting, plot_training_history
from utils.logging_utils import setup_logger

class DefenseTrainer:
    """Complete defense training pipeline"""
    
    def __init__(self, config_path: str = "config/defense_config.yaml"):
        """
        Initialize defense trainer
        
        Args:
            config_path: Path to defense configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup
        self.device = torch.device('cpu')
        self.logger = setup_logger('defense_trainer', 'reports/logs/defense_training.log')
        
        # Load data
        self.logger.info("Loading dataset...")
        train_set, test_set = load_mnist(augment=True)
        
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=64,
            shuffle=True
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=64,
            shuffle=False
        )
        
        # Results storage
        self.results = {}
    
    def train_adversarial_defense(self) -> Dict[str, Any]:
        """Train model with adversarial training defense"""
        self.logger.info("Training adversarial defense...")
        
        # Load base model
        model, metadata = load_model(
            "models/pretrained/mnist_cnn.pth",
            device=self.device
        )
        
        # Initialize adversarial training
        adv_config = self.config.get('adversarial_training', {})
        defense = AdversarialTraining(
            model=model,
            attack_type=adv_config.get('attack', 'fgsm'),
            config=adv_config
        )
        
        # Setup optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        history = []
        epochs = adv_config.get('epochs', 8)
        
        for epoch in range(epochs):
            # Train epoch
            epoch_metrics = defense.train_epoch(
                self.train_loader,
                optimizer,
                criterion,
                epoch
            )
            
            # Validate
            val_metrics = defense.validate(self.test_loader, criterion)
            
            # Combine metrics
            combined_metrics = {
                'epoch': epoch + 1,
                **epoch_metrics,
                **val_metrics
            }
            
            history.append(combined_metrics)
            
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {epoch_metrics['loss']:.4f} | "
                f"Clean Val Acc: {val_metrics['clean_accuracy']:.2f}% | "
                f"Adv Val Acc: {val_metrics['adversarial_accuracy']:.2f}%"
            )
        
        # Final evaluation
        final_metrics = evaluate_model(model, self.test_loader, self.device)
        
        # Save model
        defense_path = "models/pretrained/mnist_cnn_adv_trained.pth"
        save_model(
            model,
            defense_path,
            metadata={
                'defense_type': 'adversarial_training',
                'config': adv_config,
                'training_history': history,
                'final_metrics': final_metrics
            }
        )
        
        results = {
            'defense_type': 'adversarial_training',
            'config': adv_config,
            'training_history': history,
            'final_metrics': final_metrics,
            'model_path': defense_path
        }
        
        self.results['adversarial'] = results
        return results

    # ... (rest of the code remains the same)

