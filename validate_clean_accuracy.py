#!/usr/bin/env python3
"""
Clean Accuracy Validation Script
Validates the 99.0% clean accuracy claim for MNIST CNN model
Generates logs to logs/accuracy/clean directory
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import sys
from pathlib import Path
from datetime import datetime
import urllib.request
import gzip
import struct
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.base.mnist_cnn import MNIST_CNN


def download_mnist(data_dir='./data'):
    """Download MNIST dataset manually without torchvision"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist"
    files = {
        'train-images': ('train-images-idx3-ubyte.gz', 'f4950846f16a88802c7cc2c70afcf552'),
        'train-labels': ('train-labels-idx1-ubyte.gz', 'a3b2d93813f57aca86da6b905b6f1cff'),
        'test-images': ('t10k-images-idx3-ubyte.gz', '2bc33d8c22d54b6a2a3e2f5c7f84d5e0'),
        'test-labels': ('t10k-labels-idx1-ubyte.gz', '0cd1deff6a213a8557f3b6ddbac5ebc2')
    }
    
    downloaded_files = {}
    for key, (filename, _) in files.items():
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            url = f"{base_url}/{filename}"
            urllib.request.urlretrieve(url, filepath)
        
        # Extract gz file
        extracted_path = Path(data_dir) / filename.replace('.gz', '')
        if not extracted_path.exists():
            with gzip.open(filepath, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        downloaded_files[key] = extracted_path
    
    return downloaded_files


def parse_mnist_images(filepath):
    """Parse MNIST image file"""
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images


def parse_mnist_labels(filepath):
    """Parse MNIST label file"""
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist_test_data(batch_size=1000, data_dir='./data'):
    """Load MNIST test dataset without torchvision"""
    files = download_mnist(data_dir)
    
    # Parse test data
    images = parse_mnist_images(files['test-images'])
    labels = parse_mnist_labels(files['test-labels'])
    
    # Normalize and convert to tensors
    images = images.astype(np.float32) / 255.0
    images = (images - 0.1307) / 0.3081  # MNIST normalization
    images = torch.from_numpy(images).unsqueeze(1)  # Add channel dimension
    labels = torch.from_numpy(labels).long()
    
    dataset = TensorDataset(images, labels)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return test_loader


def evaluate_model(model, dataloader, device):
    """Evaluate model on test dataset"""
    model.eval()
    correct = 0
    total = 0
    losses = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    avg_loss = sum(losses) / len(losses)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total
    }


def main():
    """Main validation function"""
    print("=" * 60)
    print("CLEAN ACCURACY VALIDATION")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cpu')
    model_path = 'models/pretrained/mnist_cnn.pth'
    output_dir = Path('logs/accuracy/clean')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    
    # Check if model file exists
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Training model first...")
        
        # Import and run training
        from pipelines.train_model import ModelTrainer
        trainer = ModelTrainer()
        trainer.setup_data()
        trainer.setup_model()
        results = trainer.train()
        
        # Reload the trained model
        checkpoint = torch.load(model_path, map_location=device)
        model = MNIST_CNN()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        metadata = checkpoint.get('metadata', {})
    else:
        checkpoint = torch.load(model_path, map_location=device)
        model = MNIST_CNN()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        metadata = checkpoint.get('metadata', {})
    
    print(f"Model loaded successfully")
    print(f"Model class: {checkpoint.get('model_class', 'Unknown')}")
    
    # Load test data
    print("\nLoading MNIST test dataset...")
    test_loader = load_mnist_test_data(batch_size=1000)
    print(f"Test dataset loaded: {len(test_loader.dataset)} samples")
    
    # Evaluate model
    print("\nEvaluating model on clean test data...")
    metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Samples: {metrics['total']}")
    print(f"Correct Predictions: {metrics['correct']}")
    print(f"Clean Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Average Loss: {metrics['loss']:.6f}")
    print(f"{'='*60}")
    
    # Check against claim
    claimed_accuracy = 99.0
    accuracy_met = metrics['accuracy'] >= claimed_accuracy
    
    print(f"\nCLAIM VERIFICATION:")
    print(f"Claimed Accuracy: {claimed_accuracy}%")
    print(f"Measured Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Status: {'✅ PASS' if accuracy_met else '⚠️ BELOW CLAIM'}")
    
    # Generate validation report
    timestamp = datetime.now().isoformat()
    report = {
        'validation_timestamp': timestamp,
        'model_path': model_path,
        'model_class': checkpoint.get('model_class', 'Unknown'),
        'dataset': 'MNIST',
        'test_samples': metrics['total'],
        'correct_predictions': metrics['correct'],
        'clean_accuracy_percent': round(metrics['accuracy'], 2),
        'average_loss': round(metrics['loss'], 6),
        'claimed_accuracy_percent': claimed_accuracy,
        'claim_verified': accuracy_met,
        'device': str(device),
        'metadata': metadata
    }
    
    # Save JSON report
    json_report_path = output_dir / f'validation_report_{timestamp.replace(":", "-")}.json'
    with open(json_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to: {json_report_path}")
    
    # Save summary log
    summary_log_path = output_dir / 'latest_summary.log'
    with open(summary_log_path, 'w') as f:
        f.write(f"Clean Accuracy Validation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: MNIST Test Set\n")
        f.write(f"Test Samples: {metrics['total']}\n")
        f.write(f"Correct: {metrics['correct']}\n")
        f.write(f"Clean Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Claim: {claimed_accuracy}%\n")
        f.write(f"Status: {'VERIFIED' if accuracy_met else 'BELOW CLAIM'}\n")
    print(f"Summary log saved to: {summary_log_path}")
    
    # Save detailed CSV
    csv_path = output_dir / 'detailed_results.csv'
    with open(csv_path, 'w') as f:
        f.write("metric,value\n")
        f.write(f"timestamp,{timestamp}\n")
        f.write(f"model_path,{model_path}\n")
        f.write(f"test_samples,{metrics['total']}\n")
        f.write(f"correct_predictions,{metrics['correct']}\n")
        f.write(f"clean_accuracy_percent,{metrics['accuracy']:.2f}\n")
        f.write(f"average_loss,{metrics['loss']:.6f}\n")
        f.write(f"claimed_accuracy_percent,{claimed_accuracy}\n")
        f.write(f"claim_verified,{accuracy_met}\n")
    print(f"Detailed results saved to: {csv_path}")
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    
    return 0 if accuracy_met else 1


if __name__ == '__main__':
    sys.exit(main())
