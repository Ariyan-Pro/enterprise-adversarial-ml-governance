"""
Visualization utilities for model analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

def setup_plotting():
    """Setup plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Set figure defaults
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_training_history(metrics_file: str, save_path: str = None):
    """Plot training and validation metrics"""
    
    import json
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train']['loss'] for m in metrics]
    val_loss = [m['validation']['loss'] for m in metrics]
    train_acc = [m['train']['accuracy'] for m in metrics]
    val_acc = [m['validation']['accuracy'] for m in metrics]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(model, dataloader, device='cpu', save_path: str = None):
    """Plot confusion matrix"""
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_attacks(original, adversarial, predictions, save_path: str = None):
    """Visualize original vs adversarial examples"""
    
    n_samples = min(10, len(original))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    for i in range(n_samples):
        # Original image
        ax = axes[0, i]
        ax.imshow(original[i].squeeze(), cmap='gray')
        ax.set_title(f"Orig: {predictions['original'][i]}")
        ax.axis('off')
        
        # Adversarial image
        ax = axes[1, i]
        ax.imshow(adversarial[i].squeeze(), cmap='gray')
        ax.set_title(f"Adv: {predictions['adversarial'][i]}")
        ax.axis('off')
    
    plt.suptitle('Original vs Adversarial Examples')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
