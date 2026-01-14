import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Attack Transferability Pipeline
Evaluates if adversarial examples transfer between models
Critical for enterprise threat modeling
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

# Project imports
from models import MNISTCNN
from models.base.fashion_cnn import FashionCNN
from attacks import FGSMAttack
from attacks import PGDAttack
from attacks import CarliniWagnerL2, FastCarliniWagnerL2
from datasets.dataset_registry import get_dataset, get_dataset_info
from utils.dataset_utils import create_dataloaders
from utils.json_utils import safe_json_dump
from utils.logging_utils import setup_logger, log_metrics
from utils.model_utils import load_model


class AttackTransferEvaluator:
    """
    Evaluates transferability of adversarial attacks between models
    """
    
    def __init__(self, config_path: str = "config/attack_config.yaml"):
        """
        Initialize transferability evaluator
        
        Args:
            config_path: Path to attack configuration
        """
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("attack_transfer")
        
        # Results storage
        self.results = {}
        
    def load_models(self, model_names: List[str]) -> Dict[str, nn.Module]:
        """
        Load multiple models for transfer evaluation
        
        Args:
            model_names: List of model names to load
        
        Returns:
            Dictionary of loaded models
        """
        models = {}
        
        for name in model_names:
            if name == "baseline_mnist":
                model = MNISTCNN()
                model_path = "models/pretrained/mnist_cnn.pth"
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            elif name == "fashion_mnist":
                model = FashionCNN()  # Would need to create this
                model_path = "models/pretrained/fashion_cnn.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                else:
                    self.logger.warning(f"Model {name} not found, training required")
                    continue
            elif name == "trades_mnist":
                model = MNISTCNN()
                model_path = "models/pretrained/trades_mnist.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                else:
                    self.logger.warning(f"Model {name} not found, TRADES training required")
                    continue
            else:
                self.logger.error(f"Unknown model: {name}")
                continue
            
            model.to(self.device)
            model.eval()
            models[name] = model
            self.logger.info(f"Loaded model: {name}")
        
        return models
    
    def create_attacks(self, source_model: nn.Module) -> Dict[str, Any]:
        """
        Create various attacks using the source model
        
        Args:
            source_model: Model to generate attacks from
        
        Returns:
            Dictionary of attack instances
        """
        attacks = {}
        
        # FGSM
        fgsm_config = self.config.get('fgsm', {})
        attacks['fgsm'] = FGSMAttack(
            model=source_model,
            epsilon=fgsm_config.get('epsilon', 0.3)
        )
        
        # PGD
        pgd_config = self.config.get('pgd', {})
        attacks['pgd'] = PGDAttack(
            model=source_model,
            epsilon=pgd_config.get('epsilon', 0.3),
            alpha=pgd_config.get('alpha', 0.01),
            steps=pgd_config.get('steps', 40)
        )
        
        # C&W (fast version for transfer testing)
        cw_config = self.config.get('cw', {})
        attacks['cw'] = FastCarliniWagnerL2(
            model=source_model,
            const=cw_config.get('const', 1.0),
            iterations=cw_config.get('iterations', 50)
        )
        
        return attacks
    
    def evaluate_transfer(self,
                         source_model: nn.Module,
                         target_model: nn.Module,
                         attack_name: str,
                         attack_instance: Any,
                         test_loader: torch.utils.data.DataLoader,
                         num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate transferability of specific attack
        
        Args:
            source_model: Model used to generate attack
            target_model: Model to test transfer to
            attack_name: Name of the attack
            attack_instance: Attack instance
            test_loader: Test data loader
            num_samples: Number of samples to evaluate
        
        Returns:
            Transferability metrics
        """
        self.logger.info(f"Evaluating {attack_name} transferability...")
        
        total_correct = 0
        total_transfer = 0
        total_samples = 0
        
        l2_norms = []
        linf_norms = []
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            if total_samples >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples using source model
            with torch.no_grad():
                source_outputs = source_model(images)
                source_preds = source_outputs.argmax(dim=1)
                correct_mask = (source_preds == labels)
                
                # Only attack correctly classified samples
                if correct_mask.sum() == 0:
                    continue
                
                correct_images = images[correct_mask]
                correct_labels = labels[correct_mask]
                
                if len(correct_images) == 0:
                    continue
                
                # Generate adversarial examples
                adv_images = attack_instance.generate(correct_images, correct_labels)
                
                # Calculate perturbation norms
                perturbation = adv_images - correct_images
                batch_l2 = torch.norm(
                    perturbation.view(perturbation.size(0), -1), 
                    p=2, dim=1
                ).mean().item()
                batch_linf = torch.norm(
                    perturbation.view(perturbation.size(0), -1), 
                    p=float('inf'), dim=1
                ).mean().item()
                
                l2_norms.append(batch_l2)
                linf_norms.append(batch_linf)
                
                # Test on target model
                target_outputs = target_model(adv_images)
                target_preds = target_outputs.argmax(dim=1)
                
                # Check if attack transfers (misclassification)
                transfer_mask = (target_preds != correct_labels)
                
                # Update counters
                batch_size = correct_images.size(0)
                total_correct += correct_mask.sum().item()
                total_transfer += transfer_mask.sum().item()
                total_samples += batch_size
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"  Batch {batch_idx}: "
                    f"Transfer rate: {total_transfer/max(1, total_samples)*100:.1f}%"
                )
        
        if total_samples == 0:
            return {
                'transfer_rate': 0.0,
                'avg_l2_norm': 0.0,
                'avg_linf_norm': 0.0,
                'num_samples': 0
            }
        
        transfer_rate = total_transfer / total_samples * 100
        avg_l2_norm = np.mean(l2_norms) if l2_norms else 0.0
        avg_linf_norm = np.mean(linf_norms) if linf_norms else 0.0
        
        return {
            'transfer_rate': transfer_rate,
            'avg_l2_norm': avg_l2_norm,
            'avg_linf_norm': avg_linf_norm,
            'num_samples': total_samples
        }
    
    def run_transfer_matrix(self,
                           models: Dict[str, nn.Module],
                           attacks: Dict[str, Any],
                           test_loader: torch.utils.data.DataLoader,
                           num_samples: int = 100) -> Dict[str, Any]:
        """
        Run full transfer matrix evaluation
        
        Args:
            models: Dictionary of models
            attacks: Dictionary of attacks
            test_loader: Test data loader
            num_samples: Samples per evaluation
        
        Returns:
            Complete transfer matrix results
        """
        model_names = list(models.keys())
        attack_names = list(attacks.keys())
        
        transfer_matrix = {
            'model_pairs': [],
            'attack_transfers': {},
            'summary': {}
        }
        
        # Evaluate each source-target model pair
        for source_name in model_names:
            for target_name in model_names:
                if source_name == target_name:
                    continue  # Skip same model
                
                self.logger.info(f"Evaluating {source_name} -> {target_name}")
                
                source_model = models[source_name]
                target_model = models[target_name]
                
                pair_results = {
                    'source': source_name,
                    'target': target_name,
                    'attacks': {}
                }
                
                # Evaluate each attack
                for attack_name, attack_instance in attacks.items():
                    metrics = self.evaluate_transfer(
                        source_model=source_model,
                        target_model=target_model,
                        attack_name=attack_name,
                        attack_instance=attack_instance,
                        test_loader=test_loader,
                        num_samples=num_samples
                    )
                    
                    pair_results['attacks'][attack_name] = metrics
                    
                    # Store in attack-specific results
                    if attack_name not in transfer_matrix['attack_transfers']:
                        transfer_matrix['attack_transfers'][attack_name] = []
                    
                    transfer_matrix['attack_transfers'][attack_name].append({
                        'source': source_name,
                        'target': target_name,
                        **metrics
                    })
                
                transfer_matrix['model_pairs'].append(pair_results)
        
        # Calculate summary statistics
        summary = {
            'total_evaluations': len(transfer_matrix['model_pairs']),
            'models_evaluated': model_names,
            'attacks_evaluated': attack_names
        }
        
        # Calculate average transfer rates per attack
        for attack_name in attack_names:
            attack_transfers = transfer_matrix['attack_transfers'][attack_name]
            if attack_transfers:
                avg_rate = np.mean([t['transfer_rate'] for t in attack_transfers])
                summary[f'avg_transfer_rate_{attack_name}'] = avg_rate
        
        transfer_matrix['summary'] = summary
        
        return transfer_matrix
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "reports/transferability"):
        """
        Save transferability results
        
        Args:
            results: Results dictionary
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"transfer_results_{timestamp}.json"
        
        safe_json_dump(results, str(results_file))
        self.logger.info(f"Saved results to {results_file}")
        
        # Save summary report
        summary_file = output_path / f"transfer_summary_{timestamp}.md"
        self.generate_summary_report(results, str(summary_file))
        
        # Save visualization data
        viz_file = output_path / f"transfer_viz_{timestamp}.json"
        self.save_visualization_data(results, str(viz_file))
    
    def generate_summary_report(self, results: Dict[str, Any], output_file: str):
        """Generate Markdown summary report"""
        with open(output_file, 'w') as f:
            f.write("# Attack Transferability Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            summary = results.get('summary', {})
            f.write(f"- **Total Evaluations**: {summary.get('total_evaluations', 0)}\n")
            f.write(f"- **Models Evaluated**: {', '.join(summary.get('models_evaluated', []))}\n")
            f.write(f"- **Attacks Evaluated**: {', '.join(summary.get('attacks_evaluated', []))}\n\n")
            
            # Attack-specific summaries
            f.write("## Attack Transfer Rates (Average)\n\n")
            for key, value in summary.items():
                if key.startswith('avg_transfer_rate_'):
                    attack_name = key.replace('avg_transfer_rate_', '')
                    f.write(f"- **{attack_name.upper()}**: {value:.1f}%\n")
            
            # Detailed results
            f.write("\n## Detailed Results\n\n")
            for pair in results.get('model_pairs', []):
                f.write(f"### {pair['source']} → {pair['target']}\n\n")
                
                for attack_name, metrics in pair['attacks'].items():
                    f.write(f"**{attack_name.upper()}**:\n")
                    f.write(f"  - Transfer Rate: {metrics['transfer_rate']:.1f}%\n")
                    f.write(f"  - Avg L2 Norm: {metrics['avg_l2_norm']:.4f}\n")
                    f.write(f"  - Avg L∞ Norm: {metrics['avg_linf_norm']:.4f}\n")
                    f.write(f"  - Samples: {metrics['num_samples']}\n\n")
    
    def save_visualization_data(self, results: Dict[str, Any], output_file: str):
        """Save data for visualization"""
        viz_data = {
            'attack_transfers': results.get('attack_transfers', {}),
            'model_pairs': results.get('model_pairs', []),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        safe_json_dump(viz_data, output_file)
    
    def run(self,
           model_names: List[str] = None,
           dataset_name: str = "mnist",
           num_samples: int = 200,
           output_dir: str = "reports/transferability"):
        """
        Main execution method
        
        Args:
            model_names: List of model names to evaluate
            dataset_name: Dataset to use for evaluation
            num_samples: Number of samples per evaluation
            output_dir: Output directory for results
        """
        start_time = time.time()
        
        self.logger.info("Starting Attack Transferability Evaluation")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Models: {model_names}")
        
        # Default models if not specified
        if model_names is None:
            model_names = ["baseline_mnist"]
        
        # Load dataset
        train_set, test_set = get_dataset(dataset_name)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=64, shuffle=False
        )
        
        # Load models
        models = self.load_models(model_names)
        if not models:
            self.logger.error("No models loaded successfully")
            return
        
        # Create attacks using first model as source
        source_model_name = list(models.keys())[0]
        source_model = models[source_model_name]
        attacks = self.create_attacks(source_model)
        
        # Run transfer matrix evaluation
        results = self.run_transfer_matrix(
            models=models,
            attacks=attacks,
            test_loader=test_loader,
            num_samples=num_samples
        )
        
        # Save results
        self.save_results(results, output_dir)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Transferability evaluation completed in {elapsed:.2f} seconds")
        
        return results


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Attack Transferability Evaluation')
    parser.add_argument('--models', nargs='+', default=['baseline_mnist'],
                       help='Models to evaluate')
    parser.add_argument('--dataset', default='mnist',
                       help='Dataset to use (mnist, fashion_mnist)')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of samples per evaluation')
    parser.add_argument('--output', default='reports/transferability',
                       help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = AttackTransferEvaluator()
    results = evaluator.run(
        model_names=args.models,
        dataset_name=args.dataset,
        num_samples=args.samples,
        output_dir=args.output
    )
    
    # Print summary
    if results:
        summary = results.get('summary', {})
        print("\n" + "="*60)
        print("TRANSFERABILITY EVALUATION SUMMARY")
        print("="*60)
        print(f"Models: {', '.join(summary.get('models_evaluated', []))}")
        print(f"Attacks: {', '.join(summary.get('attacks_evaluated', []))}")
        
        for key, value in summary.items():
            if key.startswith('avg_transfer_rate_'):
                attack_name = key.replace('avg_transfer_rate_', '').upper()
                print(f"{attack_name} Avg Transfer Rate: {value:.1f}%")
        print("="*60)


if __name__ == "__main__":
    main()


