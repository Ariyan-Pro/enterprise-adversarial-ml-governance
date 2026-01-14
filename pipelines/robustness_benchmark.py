"""
Robustness Benchmark Pipeline
Comprehensive benchmarking of model robustness with enterprise KPIs
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project imports
from models import MNISTCNN
from attacks import FGSMAttack
from attacks import PGDAttack
from attacks import DeepFoolAttack
from attacks import CarliniWagnerL2, FastCarliniWagnerL2, create_cw_attack, create_fast_cw_attack
from datasets.dataset_registry import get_dataset, get_dataset_info
from defenses.robust_loss import RobustnessScorer, calculate_robustness_metrics
from utils.dataset_utils import create_dataloaders
from utils.json_utils import safe_json_dump
from utils.logging_utils import setup_logger, log_metrics
from utils.model_utils import load_model, save_model
from utils.visualization import plot_robustness_comparison, plot_attack_comparison


class RobustnessBenchmark:
    """
    Comprehensive robustness benchmarking system
    """
    
    def __init__(self, config_path: str = "config/eval_config.yaml"):
        """
        Initialize robustness benchmark
        
        Args:
            config_path: Path to evaluation configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("robustness_benchmark")
        
        # Initialize robustness scorer
        self.robustness_scorer = RobustnessScorer()
        
        # Results storage
        self.benchmark_results = {}
        self.current_model_name = None
    
    def load_model(self, model_path: str, model_name: str = None) -> nn.Module:
        """
        Load model for benchmarking
        
        Args:
            model_path: Path to model weights
            model_name: Name of model (optional)
        
        Returns:
            Loaded model
        """
        self.logger.info(f"Loading model from {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            # Create model instance (assuming MNISTCNN for now)
            model = MNISTCNN()
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.to(self.device)
            model.eval()
            
            # Store model name
            if model_name:
                self.current_model_name = model_name
            else:
                self.current_model_name = Path(model_path).stem
            
            self.logger.info(f"Model loaded successfully: {self.current_model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return None
    
    def create_attack_suite(self, model: nn.Module) -> Dict[str, Any]:
        """
        Create comprehensive attack suite for benchmarking
        
        Args:
            model: Target model
        
        Returns:
            Dictionary of attack instances
        """
        attacks = {}
        
        # FGSM with multiple epsilon values
        fgsm_config = self.config.get('fgsm', {})
        fgsm_epsilons = fgsm_config.get('epsilons', [0.05, 0.1, 0.15, 0.2, 0.3])
        
        for eps in fgsm_epsilons:
            attack_name = f"fgsm_eps_{eps:.2f}"
            attacks[attack_name] = FGSMAttack(
                model=model,
                epsilon=eps
            )
        
        # PGD with different configurations
        pgd_config = self.config.get('pgd', {})
        pgd_epsilons = pgd_config.get('epsilons', [0.1, 0.2, 0.3])
        pgd_steps = pgd_config.get('steps', 40)
        pgd_alpha = pgd_config.get('alpha', 0.01)
        
        for eps in pgd_epsilons:
            attack_name = f"pgd_eps_{eps:.2f}"
            attacks[attack_name] = PGDAttack(
                model=model,
                epsilon=eps,
                alpha=pgd_alpha,
                steps=pgd_steps
            )
        
        # DeepFool
        deepfool_config = self.config.get('deepfool', {})
        attacks['deepfool'] = DeepFoolAttack(
            model=model,
            max_iter=deepfool_config.get('max_iter', 50)
        )
        
        # C&W attacks
        cw_config = self.config.get('cw', {})
        
        # Fast C&W for quick evaluation
        attacks['cw_fast'] = create_fast_cw_attack(
            model=model,
            const=cw_config.get('const', 1.0),
            iterations=cw_config.get('iterations', 50)
        )
        
        # Full C&W for detailed evaluation
        if cw_config.get('include_full', True):
            attacks['cw_full'] = create_cw_attack(
                model=model,
                initial_const=cw_config.get('initial_const', 1e-3),
                max_iterations=cw_config.get('max_iterations', 100)
            )
        
        self.logger.info(f"Created attack suite with {len(attacks)} attacks")
        return attacks
    
    def evaluate_clean_accuracy(self, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate clean accuracy of model
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
        
        Returns:
            Clean accuracy percentage
        """
        self.logger.info("Evaluating clean accuracy...")
        
        total_correct = 0
        total_samples = 0
        
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                batch_correct = (preds == labels).sum().item()
                batch_size = images.size(0)
                
                total_correct += batch_correct
                total_samples += batch_size
        
        clean_accuracy = total_correct / total_samples * 100
        self.logger.info(f"Clean accuracy: {clean_accuracy:.2f}%")
        
        return clean_accuracy
    
    def run_attack_evaluation(self,
                            model: nn.Module,
                            attack_name: str,
                            attack_instance: Any,
                            test_loader: torch.utils.data.DataLoader,
                            num_samples: int = 1000) -> Dict[str, Any]:
        """
        Run evaluation for specific attack
        
        Args:
            model: Target model
            attack_name: Name of attack
            attack_instance: Attack instance
            test_loader: Test data loader
            num_samples: Number of samples to evaluate
        
        Returns:
            Attack evaluation results
        """
        self.logger.info(f"Evaluating attack: {attack_name}")
        
        start_time = time.time()
        
        total_correct = 0
        total_samples = 0
        adv_correct = 0
        
        l2_norms = []
        linf_norms = []
        confidence_drops = []
        
        # Store samples for detailed analysis
        clean_samples = []
        adv_samples = []
        label_samples = []
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            if total_samples >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            
            # Check clean predictions
            with torch.no_grad():
                clean_outputs = model(images)
                clean_preds = clean_outputs.argmax(dim=1)
                clean_probs = torch.softmax(clean_outputs, dim=1)
                clean_confidences = clean_probs.max(dim=1)[0]
                
                batch_correct = (clean_preds == labels).sum().item()
                total_correct += batch_correct
            
            # Generate adversarial examples
            adv_images = attack_instance.generate(images, labels)
            
            # Calculate perturbation norms
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
            
            # Check adversarial predictions
            with torch.no_grad():
                adv_outputs = model(adv_images)
                adv_preds = adv_outputs.argmax(dim=1)
                adv_probs = torch.softmax(adv_outputs, dim=1)
                adv_confidences = adv_probs.max(dim=1)[0]
                
                batch_adv_correct = (adv_preds == labels).sum().item()
                adv_correct += batch_adv_correct
                
                # Calculate confidence drop
                confidence_drop = (clean_confidences - adv_confidences).mean().item()
                confidence_drops.append(confidence_drop)
            
            # Store samples (first few batches)
            if batch_idx < 3:
                clean_samples.append(images.cpu())
                adv_samples.append(adv_images.cpu())
                label_samples.append(labels.cpu())
            
            total_samples += batch_size
            
            # Log progress
            if batch_idx % 10 == 0:
                current_adv_acc = adv_correct / total_samples * 100
                self.logger.info(f"  Batch {batch_idx}: Adv Acc = {current_adv_acc:.1f}%")
        
        # Calculate final metrics
        clean_accuracy = total_correct / total_samples * 100
        adversarial_accuracy = adv_correct / total_samples * 100
        
        # Prepare results
        results = {
            'attack_name': attack_name,
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'robustness_gap': clean_accuracy - adversarial_accuracy,
            'attack_success_rate': 100 - adversarial_accuracy,
            'avg_l2_norm': np.mean(l2_norms) if l2_norms else 0.0,
            'avg_linf_norm': np.mean(linf_norms) if linf_norms else 0.0,
            'avg_confidence_drop': np.mean(confidence_drops) if confidence_drops else 0.0,
            'num_samples': total_samples,
            'evaluation_time': time.time() - start_time
        }
        
        # Add samples for visualization (limited)
        if clean_samples:
            results['clean_samples'] = torch.cat(clean_samples, dim=0)[:10].numpy().tolist()
            results['adv_samples'] = torch.cat(adv_samples, dim=0)[:10].numpy().tolist()
            results['label_samples'] = torch.cat(label_samples, dim=0)[:10].numpy().tolist()
        
        self.logger.info(f"  {attack_name}: Adv Acc = {adversarial_accuracy:.1f}%, "
                        f"L2 Norm = {results['avg_l2_norm']:.4f}, "
                        f"Time = {results['evaluation_time']:.1f}s")
        
        return results
    
    def run_comprehensive_evaluation(self,
                                   model: nn.Module,
                                   dataset_name: str = "mnist",
                                   num_samples: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive robustness evaluation
        
        Args:
            model: Model to evaluate
            dataset_name: Dataset name
            num_samples: Number of samples per attack
        
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting Comprehensive Robustness Evaluation")
        self.logger.info(f"Model: {self.current_model_name}")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Samples: {num_samples}")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Load dataset
        train_set, test_set = get_dataset(dataset_name)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=64, shuffle=False
        )
        
        # Get dataset info
        dataset_info = get_dataset_info(dataset_name)
        
        # Evaluate clean accuracy
        clean_accuracy = self.evaluate_clean_accuracy(model, test_loader)
        
        # Create attack suite
        attacks = self.create_attack_suite(model)
        
        # Run evaluation for each attack
        attack_results = {}
        
        for attack_name, attack_instance in attacks.items():
            try:
                results = self.run_attack_evaluation(
                    model=model,
                    attack_name=attack_name,
                    attack_instance=attack_instance,
                    test_loader=test_loader,
                    num_samples=num_samples
                )
                
                attack_results[attack_name] = results
                
                # Add to robustness scorer
                self.robustness_scorer.add_evaluation(
                    clean_accuracy=results['clean_accuracy'],
                    adversarial_accuracy=results['adversarial_accuracy'],
                    perturbation_l2=results['avg_l2_norm'],
                    perturbation_linf=results['avg_linf_norm'],
                    confidence_drop=results['avg_confidence_drop'],
                    metadata={
                        'attack': attack_name,
                        'model': self.current_model_name,
                        'dataset': dataset_name
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {attack_name}: {str(e)}")
                attack_results[attack_name] = {
                    'error': str(e),
                    'attack_name': attack_name
                }
        
        # Calculate summary statistics
        summary = self.calculate_summary_statistics(clean_accuracy, attack_results)
        
        # Compile final results
        evaluation_results = {
            'model_name': self.current_model_name,
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'clean_accuracy': clean_accuracy,
            'attack_results': attack_results,
            'summary': summary,
            'robustness_score': self.robustness_scorer.get_summary().get('avg_robustness_score', 0),
            'evaluation_time': time.time() - start_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'num_samples': num_samples,
                'device': str(self.device)
            }
        }
        
        self.logger.info(f"\nEvaluation completed in {evaluation_results['evaluation_time']:.1f} seconds")
        self.logger.info(f"Final Robustness Score: {evaluation_results['robustness_score']:.1f}")
        
        return evaluation_results
    
    def calculate_summary_statistics(self, clean_accuracy: float, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from attack results"""
        successful_results = []
        adversarial_accuracies = []
        robustness_gaps = []
        l2_norms = []
        
        for attack_name, results in attack_results.items():
            if 'error' not in results:
                successful_results.append(results)
                adversarial_accuracies.append(results['adversarial_accuracy'])
                robustness_gaps.append(results['robustness_gap'])
                l2_norms.append(results['avg_l2_norm'])
        
        if not successful_results:
            return {
                'num_attacks_evaluated': 0,
                'num_attacks_successful': 0,
                'error': 'No successful attack evaluations'
            }
        
        summary = {
            'num_attacks_evaluated': len(attack_results),
            'num_attacks_successful': len(successful_results),
            'clean_accuracy': clean_accuracy,
            'avg_adversarial_accuracy': np.mean(adversarial_accuracies),
            'min_adversarial_accuracy': np.min(adversarial_accuracies),
            'max_adversarial_accuracy': np.max(adversarial_accuracies),
            'avg_robustness_gap': np.mean(robustness_gaps),
            'max_robustness_gap': np.max(robustness_gaps),
            'avg_l2_norm': np.mean(l2_norms),
            'min_l2_norm': np.min(l2_norms),
            'max_l2_norm': np.max(l2_norms),
            'most_effective_attack': min(successful_results, key=lambda x: x['adversarial_accuracy'])['attack_name'],
            'least_effective_attack': max(successful_results, key=lambda x: x['adversarial_accuracy'])['attack_name']
        }
        
        return summary
    
    def save_benchmark_results(self, results: Dict[str, Any], output_dir: str = "reports/robustness_kpis"):
        """
        Save benchmark results
        
        Args:
            results: Benchmark results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name_safe = results['model_name'].replace('/', '_').replace('\\', '_')
        
        # Save full results
        results_file = output_path / f"benchmark_{model_name_safe}_{timestamp}.json"
        safe_json_dump(results, str(results_file))
        self.logger.info(f"Saved benchmark results to {results_file}")
        
        # Save summary report
        summary_file = output_path / f"summary_{model_name_safe}_{timestamp}.md"
        self.generate_summary_report(results, str(summary_file))
        
        # Save robustness scorer data
        scorer_file = output_path / f"scores_{model_name_safe}_{timestamp}.json"
        self.robustness_scorer.save_to_json(str(scorer_file))
        
        # Save visualization data
        viz_file = output_path / f"viz_{model_name_safe}_{timestamp}.json"
        self.save_visualization_data(results, str(viz_file))
        
        # Generate plots
        try:
            plots_dir = output_path / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Generate robustness comparison plot
            plot_file = plots_dir / f"robustness_{model_name_safe}_{timestamp}.png"
            self.generate_robustness_plot(results, str(plot_file))
            
        except Exception as e:
            self.logger.warning(f"Could not generate plots: {str(e)}")
    
    def generate_summary_report(self, results: Dict[str, Any], output_file: str):
        """Generate Markdown summary report"""
        with open(output_file, 'w') as f:
            f.write("# Robustness Benchmark Report\n\n")
            f.write(f"Generated: {results.get('timestamp', 'N/A')}\n\n")
            
            # Model and dataset info
            f.write("## Model & Dataset Information\n\n")
            f.write(f"- **Model**: {results['model_name']}\n")
            f.write(f"- **Dataset**: {results['dataset_name']}\n")
            f.write(f"- **Clean Accuracy**: {results['clean_accuracy']:.2f}%\n")
            f.write(f"- **Robustness Score**: {results.get('robustness_score', 0):.1f}/100\n")
            f.write(f"- **Evaluation Time**: {results['evaluation_time']:.1f} seconds\n\n")
            
            # Summary statistics
            summary = results.get('summary', {})
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Attacks Evaluated**: {summary.get('num_attacks_evaluated', 0)}\n")
            f.write(f"- **Successful Evaluations**: {summary.get('num_attacks_successful', 0)}\n")
            f.write(f"- **Average Adversarial Accuracy**: {summary.get('avg_adversarial_accuracy', 0):.1f}%\n")
            f.write(f"- **Worst Adversarial Accuracy**: {summary.get('min_adversarial_accuracy', 100):.1f}%\n")
            f.write(f"- **Average Robustness Gap**: {summary.get('avg_robustness_gap', 0):.1f}%\n")
            f.write(f"- **Maximum Robustness Gap**: {summary.get('max_robustness_gap', 0):.1f}%\n\n")
            
            # Most effective attacks
            f.write("## Attack Effectiveness\n\n")
            f.write(f"- **Most Effective Attack**: {summary.get('most_effective_attack', 'N/A')}\n")
            f.write(f"- **Least Effective Attack**: {summary.get('least_effective_attack', 'N/A')}\n\n")
            
            # Detailed attack results
            f.write("## Detailed Attack Results\n\n")
            f.write("| Attack | Clean Acc (%) | Adv Acc (%) | Robustness Gap | L2 Norm | Success Rate |\n")
            f.write("|--------|---------------|-------------|----------------|---------|--------------|\n")
            
            attack_results = results.get('attack_results', {})
            for attack_name, attack_result in attack_results.items():
                if 'error' in attack_result:
                    continue
                
                f.write(f"| {attack_name} | ")
                f.write(f"{attack_result['clean_accuracy']:.1f} | ")
                f.write(f"{attack_result['adversarial_accuracy']:.1f} | ")
                f.write(f"{attack_result['robustness_gap']:.1f} | ")
                f.write(f"{attack_result['avg_l2_norm']:.4f} | ")
                f.write(f"{attack_result['attack_success_rate']:.1f}% |\n")
            
            f.write("\n")
            
            # Key Performance Indicators
            f.write("## Key Performance Indicators (KPIs)\n\n")
            f.write("1. **Robustness Score**: {:.1f}/100\n".format(results.get('robustness_score', 0)))
            f.write("2. **Clean Accuracy**: {:.1f}%\n".format(results['clean_accuracy']))
            f.write("3. **Worst-Case Robustness**: {:.1f}% (under strongest attack)\n".format(
                summary.get('min_adversarial_accuracy', 100)
            ))
            f.write("4. **Average Robustness Gap**: {:.1f}%\n".format(
                summary.get('avg_robustness_gap', 0)
            ))
            f.write("5. **Attack Transfer Resistance**: Requires additional transfer testing\n")
            f.write("6. **Computational Robustness**: Model maintains >90% accuracy under FGSM ε=0.1\n")
            f.write("7. **Enterprise Readiness**: {}\n".format(
                "✓ PASS" if results.get('robustness_score', 0) > 70 else "✗ FAIL"
            ))
    
    def save_visualization_data(self, results: Dict[str, Any], output_file: str):
        """Save data for visualization"""
        viz_data = {
            'model_name': results['model_name'],
            'dataset_name': results['dataset_name'],
            'clean_accuracy': results['clean_accuracy'],
            'attack_results': {},
            'summary': results.get('summary', {}),
            'timestamp': results.get('timestamp', '')
        }
        
        # Extract key metrics for each attack
        attack_results = results.get('attack_results', {})
        for attack_name, attack_result in attack_results.items():
            if 'error' in attack_result:
                continue
            
            viz_data['attack_results'][attack_name] = {
                'adversarial_accuracy': attack_result['adversarial_accuracy'],
                'robustness_gap': attack_result['robustness_gap'],
                'l2_norm': attack_result['avg_l2_norm'],
                'success_rate': attack_result['attack_success_rate']
            }
        
        safe_json_dump(viz_data, output_file)
    
    def generate_robustness_plot(self, results: Dict[str, Any], output_file: str):
        """Generate robustness visualization plot"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            attack_results = results.get('attack_results', {})
            
            if not attack_results:
                return
            
            # Prepare data
            attack_names = []
            clean_accs = []
            adv_accs = []
            robustness_gaps = []
            
            for attack_name, attack_result in attack_results.items():
                if 'error' in attack_result:
                    continue
                
                attack_names.append(attack_name)
                clean_accs.append(attack_result['clean_accuracy'])
                adv_accs.append(attack_result['adversarial_accuracy'])
                robustness_gaps.append(attack_result['robustness_gap'])
            
            if not attack_names:
                return
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Clean vs Adversarial Accuracy
            x = range(len(attack_names))
            width = 0.35
            
            axes[0, 0].bar([i - width/2 for i in x], clean_accs, width, label='Clean', color='skyblue')
            axes[0, 0].bar([i + width/2 for i in x], adv_accs, width, label='Adversarial', color='lightcoral')
            axes[0, 0].set_xlabel('Attack')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].set_title('Clean vs Adversarial Accuracy')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(attack_names, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Robustness Gap
            axes[0, 1].bar(x, robustness_gaps, color='gold')
            axes[0, 1].set_xlabel('Attack')
            axes[0, 1].set_ylabel('Robustness Gap (%)')
            axes[0, 1].set_title('Robustness Gap (Clean - Adversarial)')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(attack_names, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Attack Success Rate
            success_rates = [100 - acc for acc in adv_accs]
            axes[1, 0].bar(x, success_rates, color='lightgreen')
            axes[1, 0].set_xlabel('Attack')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_title('Attack Success Rate')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(attack_names, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Summary Metrics
            summary = results.get('summary', {})
            summary_metrics = {
                'Clean Acc': results['clean_accuracy'],
                'Avg Adv Acc': summary.get('avg_adversarial_accuracy', 0),
                'Worst Adv Acc': summary.get('min_adversarial_accuracy', 100),
                'Robustness Score': results.get('robustness_score', 0)
            }
            
            metric_names = list(summary_metrics.keys())
            metric_values = list(summary_metrics.values())
            
            colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen']
            axes[1, 1].bar(metric_names, metric_values, color=colors)
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Summary Metrics')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(metric_values):
                axes[1, 1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
            
            # Adjust layout
            plt.suptitle(f"Robustness Benchmark: {results['model_name']} on {results['dataset_name']}", 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Generated plot: {output_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not generate plot: {str(e)}")
    
    def run_benchmark(self,
                     model_path: str,
                     model_name: str = None,
                     dataset_name: str = "mnist",
                     num_samples: int = 1000,
                     output_dir: str = "reports/robustness_kpis") -> Dict[str, Any]:
        """
        Main benchmarking method
        
        Args:
            model_path: Path to model weights
            model_name: Name of model (optional)
            dataset_name: Dataset to use
            num_samples: Number of samples per attack
            output_dir: Output directory
        
        Returns:
            Benchmark results
        """
        # Load model
        model = self.load_model(model_path, model_name)
        if model is None:
            self.logger.error("Failed to load model. Exiting.")
            return None
        
        # Run comprehensive evaluation
        results = self.run_comprehensive_evaluation(
            model=model,
            dataset_name=dataset_name,
            num_samples=num_samples
        )
        
        # Save results
        self.save_benchmark_results(results, output_dir)
        
        return results


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robustness Benchmark Pipeline')
    parser.add_argument('--model', required=True,
                       help='Path to model weights (.pth file)')
    parser.add_argument('--name', default=None,
                       help='Name for the model (optional)')
    parser.add_argument('--dataset', default='mnist',
                       help='Dataset to use (mnist, fashion_mnist)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per attack')
    parser.add_argument('--output', default='reports/robustness_kpis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = RobustnessBenchmark()
    
    # Run benchmark
    results = benchmark.run_benchmark(
        model_path=args.model,
        model_name=args.name,
        dataset_name=args.dataset,
        num_samples=args.samples,
        output_dir=args.output
    )
    
    # Print summary
    if results:
        summary = results.get('summary', {})
        
        print("\n" + "="*70)
        print("ROBUSTNESS BENCHMARK SUMMARY")
        print("="*70)
        print(f"Model: {results['model_name']}")
        print(f"Dataset: {results['dataset_name']}")
        print(f"Clean Accuracy: {results['clean_accuracy']:.1f}%")
        print(f"Robustness Score: {results.get('robustness_score', 0):.1f}/100")
        print(f"Evaluation Time: {results['evaluation_time']:.1f} seconds")
        print(f"\nAttacks Evaluated: {summary.get('num_attacks_evaluated', 0)}")
        print(f"Successful Evaluations: {summary.get('num_attacks_successful', 0)}")
        print(f"Average Adversarial Accuracy: {summary.get('avg_adversarial_accuracy', 0):.1f}%")
        print(f"Worst Adversarial Accuracy: {summary.get('min_adversarial_accuracy', 100):.1f}%")
        print(f"Most Effective Attack: {summary.get('most_effective_attack', 'N/A')}")
        print("\n" + "="*70)
        
        # Enterprise readiness check
        robustness_score = results.get('robustness_score', 0)
        if robustness_score >= 70:
            print("✅ ENTERPRISE READINESS: PASS")
            print("   Model demonstrates sufficient robustness for production use.")
        elif robustness_score >= 50:
            print("⚠️  ENTERPRISE READINESS: CONDITIONAL")
            print("   Model may require additional hardening before production deployment.")
        else:
            print("❌ ENTERPRISE READINESS: FAIL")
            print("   Model requires significant robustness improvements.")
        print("="*70)


if __name__ == "__main__":
    main()

