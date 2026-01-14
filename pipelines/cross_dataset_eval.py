import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Cross-Dataset Robustness Evaluation Pipeline
Tests model robustness on different data distributions
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import yaml

# Project imports
from models import MNISTCNN
from attacks import FGSMAttack
from attacks import PGDAttack
from attacks import FastCarliniWagnerL2
from datasets.dataset_registry import get_dataset, get_dataset_info, list_datasets
from utils.dataset_utils import create_dataloaders
from utils.json_utils import safe_json_dump
from utils.logging_utils import setup_logger
from utils.model_utils import load_model
from defenses.robust_loss import calculate_robustness_metrics, RobustnessScorer


class CrossDatasetEvaluator:
    """
    Evaluates model robustness across different datasets
    """
    
    def __init__(self, config_path: str = "config/eval_config.yaml"):
        """
        Initialize cross-dataset evaluator
        
        Args:
            config_path: Path to evaluation configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("cross_dataset_eval")
        
        # Initialize robustness scorer
        self.robustness_scorer = RobustnessScorer()
        
        # Results storage
        self.results = {}
    
    def load_model_for_dataset(self, model_name: str, dataset_name: str) -> nn.Module:
        """
        Load appropriate model for dataset
        
        Args:
            model_name: Name of model to load
            dataset_name: Target dataset
        
        Returns:
            Loaded model
        """
        # For now, use MNIST CNN for both datasets
        # In production, you'd have dataset-specific models
        model = MNISTCNN()
        
        if model_name == "baseline_mnist":
            model_path = "models/pretrained/mnist_cnn.pth"
        elif model_name == "trades_mnist":
            model_path = "models/pretrained/trades_mnist.pth"
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model {model_name} not found at {model_path}")
            return None
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def create_attacks(self, model: nn.Module) -> Dict[str, Any]:
        """
        Create attack instances for evaluation
        
        Args:
            model: Target model
        
        Returns:
            Dictionary of attack instances
        """
        attacks = {}
        
        # FGSM with multiple epsilon values
        epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        for eps in epsilon_values:
            attacks[f'fgsm_eps_{eps}'] = FGSMAttack(
                model=model,
                epsilon=eps
            )
        
        # PGD
        attacks['pgd'] = PGDAttack(
            model=model,
            epsilon=0.3,
            alpha=0.01,
            steps=40
        )
        
        # C&W (fast)
        attacks['cw_fast'] = FastCarliniWagnerL2(
            model=model,
            const=1.0,
            iterations=50
        )
        
        return attacks
    
    def evaluate_dataset(self,
                        model: nn.Module,
                        dataset_name: str,
                        attacks: Dict[str, Any],
                        num_samples: int = 500) -> Dict[str, Any]:
        """
        Evaluate model on specific dataset
        
        Args:
            model: Model to evaluate
            dataset_name: Name of dataset
            attacks: Dictionary of attack instances
            num_samples: Number of samples to evaluate
        
        Returns:
            Evaluation results for dataset
        """
        self.logger.info(f"Evaluating on {dataset_name} dataset...")
        
        # Load dataset
        train_set, test_set = get_dataset(dataset_name)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=64, shuffle=False
        )
        
        dataset_info = get_dataset_info(dataset_name)
        
        # Results storage
        dataset_results = {
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'clean_accuracy': 0.0,
            'attacks': {},
            'robustness_metrics': {}
        }
        
        # Calculate clean accuracy
        clean_correct = 0
        clean_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                if clean_total >= num_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                batch_correct = (preds == labels).sum().item()
                batch_size = images.size(0)
                
                clean_correct += batch_correct
                clean_total += batch_size
        
        clean_accuracy = clean_correct / clean_total * 100
        dataset_results['clean_accuracy'] = clean_accuracy
        
        # Evaluate each attack
        for attack_name, attack_instance in attacks.items():
            self.logger.info(f"  Running {attack_name}...")
            
            attack_results = self.evaluate_attack(
                model=model,
                attack=attack_instance,
                test_loader=test_loader,
                num_samples=num_samples,
                attack_name=attack_name
            )
            
            dataset_results['attacks'][attack_name] = attack_results
            
            # Calculate robustness metrics
            if 'adversarial_accuracy' in attack_results:
                robust_metrics = calculate_robustness_metrics(
                    model=model,
                    clean_images=attack_results.get('clean_images_sample', torch.Tensor()),
                    adversarial_images=attack_results.get('adversarial_images_sample', torch.Tensor()),
                    labels=attack_results.get('labels_sample', torch.Tensor())
                )
                dataset_results['attacks'][attack_name]['robustness_metrics'] = robust_metrics
        
        # Calculate dataset-level robustness score
        self.calculate_dataset_robustness(dataset_results)
        
        return dataset_results
    
    def evaluate_attack(self,
                       model: nn.Module,
                       attack: Any,
                       test_loader: torch.utils.data.DataLoader,
                       num_samples: int,
                       attack_name: str) -> Dict[str, Any]:
        """
        Evaluate specific attack
        
        Args:
            model: Target model
            attack: Attack instance
            test_loader: Test data loader
            num_samples: Number of samples
            attack_name: Name of attack
        
        Returns:
            Attack evaluation results
        """
        total_correct = 0
        total_samples = 0
        adv_correct = 0
        
        l2_norms = []
        linf_norms = []
        
        # Store samples for robustness metrics
        clean_images_list = []
        adv_images_list = []
        labels_list = []
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            if total_samples >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            
            # Check clean accuracy
            with torch.no_grad():
                clean_outputs = model(images)
                clean_preds = clean_outputs.argmax(dim=1)
                batch_correct = (clean_preds == labels).sum().item()
                total_correct += batch_correct
            
            # Generate adversarial examples
            adv_images = attack.generate(images, labels)
            
            # Calculate perturbation
            perturbation = adv_images - images
            batch_l2 = torch.norm(
                perturbation.view(batch_size, -1), 
                p=2, dim=1
            ).mean().item()
            batch_linf = torch.norm(
                perturbation.view(batch_size, -1), 
                p=float('inf'), dim=1
            ).mean().item()
            
            l2_norms.append(batch_l2)
            linf_norms.append(batch_linf)
            
            # Check adversarial accuracy
            with torch.no_grad():
                adv_outputs = model(adv_images)
                adv_preds = adv_outputs.argmax(dim=1)
                batch_adv_correct = (adv_preds == labels).sum().item()
                adv_correct += batch_adv_correct
            
            # Store samples (first batch only)
            if batch_idx == 0:
                clean_images_list.append(images.cpu())
                adv_images_list.append(adv_images.cpu())
                labels_list.append(labels.cpu())
            
            total_samples += batch_size
        
        clean_accuracy = total_correct / total_samples * 100
        adv_accuracy = adv_correct / total_samples * 100
        
        # Prepare samples for robustness metrics
        clean_images_sample = torch.cat(clean_images_list, dim=0) if clean_images_list else torch.Tensor()
        adv_images_sample = torch.cat(adv_images_list, dim=0) if adv_images_list else torch.Tensor()
        labels_sample = torch.cat(labels_list, dim=0) if labels_list else torch.Tensor()
        
        return {
            'attack_name': attack_name,
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'robustness_gap': clean_accuracy - adv_accuracy,
            'attack_success_rate': 100 - adv_accuracy,
            'avg_l2_norm': np.mean(l2_norms) if l2_norms else 0.0,
            'avg_linf_norm': np.mean(linf_norms) if linf_norms else 0.0,
            'num_samples': total_samples,
            'clean_images_sample': clean_images_sample,
            'adversarial_images_sample': adv_images_sample,
            'labels_sample': labels_sample
        }
    
    def calculate_dataset_robustness(self, dataset_results: Dict[str, Any]):
        """Calculate comprehensive robustness metrics for dataset"""
        attacks = dataset_results.get('attacks', {})
        
        if not attacks:
            return
        
        # Calculate average metrics across all attacks
        metrics = {
            'avg_clean_accuracy': dataset_results['clean_accuracy'],
            'avg_adversarial_accuracy': np.mean([a.get('adversarial_accuracy', 0) for a in attacks.values()]),
            'avg_robustness_gap': np.mean([a.get('robustness_gap', 0) for a in attacks.values()]),
            'avg_attack_success_rate': np.mean([a.get('attack_success_rate', 0) for a in attacks.values()]),
            'worst_adversarial_accuracy': min([a.get('adversarial_accuracy', 100) for a in attacks.values()]),
            'best_adversarial_accuracy': max([a.get('adversarial_accuracy', 0) for a in attacks.values()]),
            'num_attacks_evaluated': len(attacks)
        }
        
        dataset_results['robustness_metrics'] = metrics
    
    def run_comparison(self,
                      model_names: List[str],
                      dataset_names: List[str],
                      num_samples: int = 500,
                      output_dir: str = "reports/cross_dataset") -> Dict[str, Any]:
        """
        Run cross-dataset comparison
        
        Args:
            model_names: List of model names
            dataset_names: List of dataset names
            num_samples: Samples per evaluation
            output_dir: Output directory
        
        Returns:
            Comparison results
        """
        start_time = time.time()
        
        self.logger.info("Starting Cross-Dataset Evaluation")
        self.logger.info(f"Models: {model_names}")
        self.logger.info(f"Datasets: {dataset_names}")
        
        all_results = {
            'models': {},
            'datasets': dataset_names,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'num_samples': num_samples,
                'model_names': model_names
            }
        }
        
        # Evaluate each model
        for model_name in model_names:
            self.logger.info(f"\nEvaluating model: {model_name}")
            
            model_results = {
                'model_name': model_name,
                'dataset_results': {}
            }
            
            # Evaluate on each dataset
            for dataset_name in dataset_names:
                # Load model (same architecture for now)
                model = self.load_model_for_dataset(model_name, dataset_name)
                if model is None:
                    self.logger.warning(f"Skipping {dataset_name} for {model_name}")
                    continue
                
                # Create attacks for this model
                attacks = self.create_attacks(model)
                
                # Evaluate on dataset
                dataset_result = self.evaluate_dataset(
                    model=model,
                    dataset_name=dataset_name,
                    attacks=attacks,
                    num_samples=num_samples
                )
                
                model_results['dataset_results'][dataset_name] = dataset_result
                
                # Add to robustness scorer
                robust_metrics = dataset_result.get('robustness_metrics', {})
                self.robustness_scorer.add_evaluation(
                    clean_accuracy=dataset_result['clean_accuracy'],
                    adversarial_accuracy=robust_metrics.get('avg_adversarial_accuracy', 0),
                    perturbation_l2=0,  # Would need to calculate
                    perturbation_linf=0,
                    confidence_drop=0,
                    metadata={
                        'model': model_name,
                        'dataset': dataset_name
                    }
                )
            
            all_results['models'][model_name] = model_results
        
        # Calculate cross-dataset comparisons
        all_results['comparisons'] = self.calculate_comparisons(all_results)
        
        # Save results
        self.save_results(all_results, output_dir)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Cross-dataset evaluation completed in {elapsed:.2f} seconds")
        
        return all_results
    
    def calculate_comparisons(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparisons between datasets"""
        comparisons = {
            'dataset_performance': {},
            'model_consistency': {},
            'summary': {}
        }
        
        models = results.get('models', {})
        datasets = results.get('datasets', [])
        
        # Compare performance across datasets
        for dataset in datasets:
            dataset_perf = {
                'models': {},
                'avg_clean_accuracy': 0.0,
                'avg_adversarial_accuracy': 0.0
            }
            
            clean_accs = []
            adv_accs = []
            
            for model_name, model_results in models.items():
                dataset_results = model_results['dataset_results'].get(dataset)
                if dataset_results:
                    clean_acc = dataset_results['clean_accuracy']
                    robust_metrics = dataset_results.get('robustness_metrics', {})
                    adv_acc = robust_metrics.get('avg_adversarial_accuracy', 0)
                    
                    dataset_perf['models'][model_name] = {
                        'clean_accuracy': clean_acc,
                        'adversarial_accuracy': adv_acc
                    }
                    
                    clean_accs.append(clean_acc)
                    adv_accs.append(adv_acc)
            
            if clean_accs:
                dataset_perf['avg_clean_accuracy'] = np.mean(clean_accs)
                dataset_perf['avg_adversarial_accuracy'] = np.mean(adv_accs)
            
            comparisons['dataset_performance'][dataset] = dataset_perf
        
        # Calculate model consistency across datasets
        for model_name, model_results in models.items():
            dataset_results = model_results['dataset_results']
            
            clean_accs = [dr['clean_accuracy'] for dr in dataset_results.values()]
            adv_accs = [dr.get('robustness_metrics', {}).get('avg_adversarial_accuracy', 0) 
                       for dr in dataset_results.values()]
            
            if clean_accs:
                consistency = {
                    'clean_accuracy_mean': np.mean(clean_accs),
                    'clean_accuracy_std': np.std(clean_accs),
                    'adversarial_accuracy_mean': np.mean(adv_accs),
                    'adversarial_accuracy_std': np.std(adv_accs),
                    'num_datasets': len(dataset_results)
                }
                comparisons['model_consistency'][model_name] = consistency
        
        # Overall summary
        comparisons['summary'] = {
            'datasets_evaluated': len(datasets),
            'models_evaluated': list(models.keys()),
            'robustness_summary': self.robustness_scorer.get_summary()
        }
        
        return comparisons
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_file = output_path / f"cross_dataset_results_{timestamp}.json"
        safe_json_dump(results, str(results_file))
        self.logger.info(f"Saved results to {results_file}")
        
        # Save summary report
        summary_file = output_path / f"cross_dataset_summary_{timestamp}.md"
        self.generate_summary_report(results, str(summary_file))
        
        # Save robustness scorer data
        scorer_file = output_path / f"robustness_scores_{timestamp}.json"
        self.robustness_scorer.save_to_json(str(scorer_file))
    
    def generate_summary_report(self, results: Dict[str, Any], output_file: str):
        """Generate Markdown summary report"""
        with open(output_file, 'w') as f:
            f.write("# Cross-Dataset Robustness Evaluation Report\n\n")
            f.write(f"Generated: {results.get('timestamp', 'N/A')}\n\n")
            
            # Config summary
            config = results.get('config', {})
            f.write("## Configuration\n\n")
            f.write(f"- **Models**: {', '.join(config.get('model_names', []))}\n")
            f.write(f"- **Datasets**: {', '.join(results.get('datasets', []))}\n")
            f.write(f"- **Samples per evaluation**: {config.get('num_samples', 0)}\n\n")
            
            # Dataset performance comparison
            f.write("## Dataset Performance Comparison\n\n")
            comparisons = results.get('comparisons', {})
            dataset_perf = comparisons.get('dataset_performance', {})
            
            for dataset, perf in dataset_perf.items():
                f.write(f"### {dataset}\n\n")
                f.write(f"- **Average Clean Accuracy**: {perf.get('avg_clean_accuracy', 0):.1f}%\n")
                f.write(f"- **Average Adversarial Accuracy**: {perf.get('avg_adversarial_accuracy', 0):.1f}%\n\n")
                
                for model_name, model_perf in perf.get('models', {}).items():
                    f.write(f"  **{model_name}**: ")
                    f.write(f"Clean: {model_perf.get('clean_accuracy', 0):.1f}%, ")
                    f.write(f"Adv: {model_perf.get('adversarial_accuracy', 0):.1f}%\n")
                f.write("\n")
            
            # Model consistency
            f.write("## Model Consistency Across Datasets\n\n")
            model_consistency = comparisons.get('model_consistency', {})
            
            for model_name, consistency in model_consistency.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- **Clean Accuracy**: {consistency['clean_accuracy_mean']:.1f}% ± {consistency['clean_accuracy_std']:.1f}%\n")
                f.write(f"- **Adversarial Accuracy**: {consistency['adversarial_accuracy_mean']:.1f}% ± {consistency['adversarial_accuracy_std']:.1f}%\n")
                f.write(f"- **Datasets evaluated**: {consistency['num_datasets']}\n\n")
            
            # Robustness summary
            f.write("## Robustness Summary\n\n")
            robustness_summary = comparisons.get('summary', {}).get('robustness_summary', {})
            if robustness_summary:
                f.write(f"- **Average Robustness Score**: {robustness_summary.get('avg_robustness_score', 0):.1f}\n")
                f.write(f"- **Best Robustness Score**: {robustness_summary.get('best_robustness_score', 0):.1f}\n")
                f.write(f"- **Worst Robustness Score**: {robustness_summary.get('worst_robustness_score', 0):.1f}\n")
                f.write(f"- **Average Clean Accuracy**: {robustness_summary.get('avg_clean_accuracy', 0):.1f}%\n")
                f.write(f"- **Average Adversarial Accuracy**: {robustness_summary.get('avg_adversarial_accuracy', 0):.1f}%\n\n")
    
    def run(self,
           model_names: List[str] = None,
           dataset_names: List[str] = None,
           num_samples: int = 500,
           output_dir: str = "reports/cross_dataset"):
        """
        Main execution method
        
        Args:
            model_names: List of model names
            dataset_names: List of dataset names
            num_samples: Samples per evaluation
            output_dir: Output directory
        """
        if model_names is None:
            model_names = ["baseline_mnist"]
        
        if dataset_names is None:
            dataset_names = list_datasets()
        
        return self.run_comparison(
            model_names=model_names,
            dataset_names=dataset_names,
            num_samples=num_samples,
            output_dir=output_dir
        )


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-Dataset Robustness Evaluation')
    parser.add_argument('--models', nargs='+', default=['baseline_mnist'],
                       help='Models to evaluate')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Datasets to evaluate (default: all available)')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of samples per evaluation')
    parser.add_argument('--output', default='reports/cross_dataset',
                       help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = CrossDatasetEvaluator()
    results = evaluator.run(
        model_names=args.models,
        dataset_names=args.datasets,
        num_samples=args.samples,
        output_dir=args.output
    )
    
    # Print summary
    if results:
        print("\n" + "="*60)
        print("CROSS-DATASET EVALUATION SUMMARY")
        print("="*60)
        print(f"Models evaluated: {', '.join(results.get('models', {}).keys())}")
        print(f"Datasets evaluated: {', '.join(results.get('datasets', []))}")
        
        comparisons = results.get('comparisons', {})
        summary = comparisons.get('summary', {})
        robustness_summary = summary.get('robustness_summary', {})
        
        if robustness_summary:
            print(f"\nRobustness Scores:")
            print(f"  Average: {robustness_summary.get('avg_robustness_score', 0):.1f}")
            print(f"  Best: {robustness_summary.get('best_robustness_score', 0):.1f}")
            print(f"  Worst: {robustness_summary.get('worst_robustness_score', 0):.1f}")
        print("="*60)


if __name__ == "__main__":
    main()


