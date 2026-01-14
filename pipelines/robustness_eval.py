"""
Robustness Evaluation Pipeline
Enterprise-grade evaluation of model robustness against multiple attacks
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from utils.json_utils import NumpyEncoder, safe_json_dump
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from attacks.fgsm import create_fgsm_attack
from attacks.pgd import create_pgd_attack
from attacks.deepfool import create_deepfool_attack
from defenses.adv_training import AdversarialTraining
from defenses.input_smoothing import create_input_smoothing
from defenses.randomized_transform import create_randomized_transform
from defenses.model_wrappers import (
    create_ensemble_wrapper,
    create_distillation_wrapper,
    create_adversarial_detector
)
from utils.model_utils import load_model, evaluate_model
from utils.dataset_utils import load_mnist
from utils.visualization import setup_plotting, plot_confusion_matrix
from utils.logging_utils import setup_logger

class RobustnessEvaluator:
    """Complete robustness evaluation pipeline"""
    
    def __init__(self, config_path: str = "config/eval_config.yaml"):
        """
        Initialize robustness evaluator
        
        Args:
            config_path: Path to evaluation configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.logger = setup_logger('robustness_evaluator', 'reports/logs/robustness_eval.log')
        
        # Load model
        self.logger.info("Loading model...")
        self.model, self.model_metadata = load_model(
            "models/pretrained/mnist_cnn.pth",
            device=self.device
        )
        
        # Load data
        self.logger.info("Loading dataset...")
        _, test_set = load_mnist()
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.config.get('batch_size', 64),
            shuffle=False
        )
        
        # Initialize attacks for evaluation
        self._init_attacks()
        
        # Initialize defenses for evaluation
        self._init_defenses()
        
        # Results storage
        self.results = {
            'model_info': self.model_metadata,
            'evaluation_timestamp': str(datetime.now()),
            'clean_performance': {},
            'attack_results': {},
            'defense_results': {},
            'comparison': {}
        }
    
    def _init_attacks(self):
        """Initialize attacks for evaluation"""
        self.attacks = {}
        
        # Load attack config
        with open("config/attack_config.yaml", 'r') as f:
            attack_config = yaml.safe_load(f)
        
        # FGSM with multiple epsilon values
        epsilons = [0.05, 0.1, 0.15, 0.2, 0.3]
        for eps in epsilons:
            attack_name = f"fgsm_epsilon_{eps}"
            self.attacks[attack_name] = create_fgsm_attack(
                self.model,
                epsilon=eps,
                device=self.device
            )
        
        # PGD with multiple configurations
        pgd_configs = [
            {'epsilon': 0.1, 'steps': 10, 'alpha': 0.01},
            {'epsilon': 0.2, 'steps': 20, 'alpha': 0.01},
            {'epsilon': 0.3, 'steps': 40, 'alpha': 0.0075}
        ]
        
        for i, config in enumerate(pgd_configs):
            attack_name = f"pgd_config_{i+1}"
            self.attacks[attack_name] = create_pgd_attack(
                self.model,
                **config,
                device=self.device
            )
        
        # DeepFool
        self.attacks['deepfool'] = create_deepfool_attack(
            self.model,
            device=self.device
        )
        
        self.logger.info(f"Initialized {len(self.attacks)} attacks for evaluation")
    
    def _init_defenses(self):
        """Initialize defenses for evaluation"""
        self.defenses = {}
        
        # Input smoothing
        smoothing_types = ['gaussian', 'median', 'bilateral']
        for smooth_type in smoothing_types:
            defense_name = f"input_smoothing_{smooth_type}"
            self.defenses[defense_name] = create_input_smoothing(
                smoothing_type=smooth_type,
                kernel_size=3,
                sigma=1.0
            )
        
        # Randomized transformations
        transform_modes = ['random', 'ensemble']
        for mode in transform_modes:
            defense_name = f"randomized_transform_{mode}"
            self.defenses[defense_name] = create_randomized_transform(
                mode=mode,
                ensemble_size=5 if mode == 'ensemble' else 1
            )
        
        self.logger.info(f"Initialized {len(self.defenses)} defenses for evaluation")
    
    def evaluate_clean_performance(self) -> Dict[str, float]:
        """
        Evaluate model performance on clean data
        
        Returns:
            Dictionary of clean performance metrics
        """
        self.logger.info("Evaluating clean performance...")
        
        metrics = evaluate_model(self.model, self.test_loader, self.device)
        
        # Add additional metrics
        all_preds = []
        all_labels = []
        all_confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)
                confidences = probs.max(dim=1)[0]
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_confidences.append(confidences.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_confidences = torch.cat(all_confidences)
        
        # Per-class accuracy
        class_accuracies = {}
        for class_idx in range(10):
            class_mask = (all_labels == class_idx)
            if class_mask.any():
                class_acc = (all_preds[class_mask] == all_labels[class_mask]).float().mean().item()
                class_accuracies[f"class_{class_idx}_accuracy"] = class_acc * 100
        
        # Confidence statistics
        conf_stats = {
            'mean_confidence': all_confidences.mean().item(),
            'std_confidence': all_confidences.std().item(),
            'min_confidence': all_confidences.min().item(),
            'max_confidence': all_confidences.max().item()
        }
        
        # Compile results
        results = {
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            **class_accuracies,
            **conf_stats
        }
        
        self.results['clean_performance'] = results
        
        self.logger.info(f"Clean Accuracy: {metrics['accuracy']:.2f}%")
        self.logger.info(f"Clean Loss: {metrics['loss']:.4f}")
        
        return results
    
    def evaluate_attack_robustness(self, 
                                  attack_name: str,
                                  attack: Any,
                                  num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate model robustness against a specific attack
        
        Args:
            attack_name: Name of the attack
            attack: Attack object
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of attack robustness metrics
        """
        self.logger.info(f"Evaluating robustness against {attack_name}...")
        
        correct_before = 0
        correct_after = 0
        total = 0
        
        perturbation_norms = []
        confidence_drops = []
        
        sample_results = {
            'clean_images': [],
            'adversarial_images': [],
            'labels': [],
            'clean_predictions': [],
            'adversarial_predictions': []
        }
        
        self.model.eval()
        
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            if num_samples and total >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            if num_samples:
                take = min(batch_size, num_samples - total)
                images = images[:take]
                labels = labels[:take]
                batch_size = take
            
            # Get clean predictions
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_preds = clean_outputs.argmax(dim=1)
                clean_probs = torch.softmax(clean_outputs, dim=1)
                clean_confidences = clean_probs.max(dim=1)[0]
            
            # Generate adversarial examples
            if attack_name == 'deepfool':
                adversarial_images = attack.generate(images)
            else:
                adversarial_images = attack.generate(images, labels)
            
            # Get adversarial predictions
            with torch.no_grad():
                adv_outputs = self.model(adversarial_images)
                adv_preds = adv_outputs.argmax(dim=1)
                adv_probs = torch.softmax(adv_outputs, dim=1)
                adv_confidences = adv_probs.max(dim=1)[0]
            
            # Calculate metrics
            batch_correct_before = (clean_preds == labels).sum().item()
            batch_correct_after = (adv_preds == labels).sum().item()
            
            correct_before += batch_correct_before
            correct_after += batch_correct_after
            total += batch_size
            
            # Perturbation metrics
            perturbations = adversarial_images - images
            batch_l2_norms = torch.norm(
                perturbations.view(batch_size, -1), 
                p=2, dim=1
            )
            perturbation_norms.extend(batch_l2_norms.cpu().numpy())
            
            # Confidence drop
            confidence_drop = clean_confidences - adv_confidences
            confidence_drops.extend(confidence_drop.cpu().numpy())
            
            # Store sample results for visualization
            if len(sample_results['clean_images']) < 10:
                n_needed = 10 - len(sample_results['clean_images'])
                n_take = min(n_needed, batch_size)
                
                sample_results['clean_images'].append(images[:n_take].cpu())
                sample_results['adversarial_images'].append(adversarial_images[:n_take].cpu())
                sample_results['labels'].append(labels[:n_take].cpu())
                sample_results['clean_predictions'].append(clean_preds[:n_take].cpu())
                sample_results['adversarial_predictions'].append(adv_preds[:n_take].cpu())
            
            # Log progress
            if batch_idx % 10 == 0:
                batch_accuracy = batch_correct_before / batch_size * 100
                batch_robustness = batch_correct_after / batch_size * 100
                self.logger.debug(
                    f"Batch {batch_idx}: Clean Acc={batch_accuracy:.1f}%, "
                    f"Robust Acc={batch_robustness:.1f}%"
                )
        
        # Compile results
        clean_accuracy = correct_before / total * 100
        robust_accuracy = correct_after / total * 100
        attack_success_rate = 100 - robust_accuracy
        
        results = {
            'attack_name': attack_name,
            'num_samples': total,
            'clean_accuracy': clean_accuracy,
            'robust_accuracy': robust_accuracy,
            'attack_success_rate': attack_success_rate,
            'robustness_gap': clean_accuracy - robust_accuracy,
            'avg_perturbation_norm': np.mean(perturbation_norms),
            'std_perturbation_norm': np.std(perturbation_norms),
            'avg_confidence_drop': np.mean(confidence_drops),
            'std_confidence_drop': np.std(confidence_drops),
            'attack_config': getattr(attack, 'config', {})
        }
        
        # Combine sample results
        if sample_results['clean_images']:
            sample_results = {
                'clean_images': torch.cat(sample_results['clean_images'], dim=0),
                'adversarial_images': torch.cat(sample_results['adversarial_images'], dim=0),
                'labels': torch.cat(sample_results['labels'], dim=0),
                'clean_predictions': torch.cat(sample_results['clean_predictions'], dim=0),
                'adversarial_predictions': torch.cat(sample_results['adversarial_predictions'], dim=0)
            }
        
        self.logger.info(f"{attack_name}:")
        self.logger.info(f"  Clean Accuracy: {clean_accuracy:.2f}%")
        self.logger.info(f"  Robust Accuracy: {robust_accuracy:.2f}%")
        self.logger.info(f"  Attack Success: {attack_success_rate:.2f}%")
        self.logger.info(f"  Avg Perturbation: {np.mean(perturbation_norms):.4f}")
        
        return results, sample_results
    
    def evaluate_all_attacks(self, num_samples: Optional[int] = 1000) -> Dict[str, Any]:
        """
        Evaluate robustness against all attacks
        
        Args:
            num_samples: Number of samples per attack
            
        Returns:
            Dictionary of all attack results
        """
        self.logger.info(f"Evaluating robustness against all attacks (samples={num_samples})...")
        
        all_results = {}
        all_samples = {}
        
        for attack_name, attack in self.attacks.items():
            try:
                results, samples = self.evaluate_attack_robustness(
                    attack_name, attack, num_samples
                )
                all_results[attack_name] = results
                all_samples[attack_name] = samples
                
                # Save individual attack results
                self._save_attack_results(attack_name, results, samples)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {attack_name}: {e}")
                continue
        
        self.results['attack_results'] = all_results
        
        # Generate attack comparison
        self._generate_attack_comparison(all_results)
        
        return all_results
    
    def evaluate_defense_effectiveness(self,
                                      defense_name: str,
                                      defense: Any,
                                      attack_name: str,
                                      attack: Any,
                                      num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate defense effectiveness against a specific attack
        
        Args:
            defense_name: Name of the defense
            defense: Defense object
            attack_name: Name of the attack
            attack: Attack object
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of defense effectiveness metrics
        """
        self.logger.info(f"Evaluating {defense_name} against {attack_name}...")
        
        total = 0
        clean_correct = 0
        adv_correct_no_defense = 0
        adv_correct_with_defense = 0
        
        self.model.eval()
        
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            if num_samples and total >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            if num_samples:
                take = min(batch_size, num_samples - total)
                images = images[:take]
                labels = labels[:take]
                batch_size = take
            
            # Get clean predictions
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_preds = clean_outputs.argmax(dim=1)
            
            # Generate adversarial examples
            if attack_name == 'deepfool':
                adversarial_images = attack.generate(images)
            else:
                adversarial_images = attack.generate(images, labels)
            
            # Apply defense
            if defense_name.startswith('input_smoothing') or defense_name.startswith('randomized_transform'):
                defended_images = defense.apply(adversarial_images, self.model)
            else:
                defended_images = adversarial_images  # Some defenses work differently
            
            # Get predictions
            with torch.no_grad():
                # Without defense
                adv_outputs = self.model(adversarial_images)
                adv_preds = adv_outputs.argmax(dim=1)
                
                # With defense
                if defense_name.startswith('randomized_transform') and 'ensemble' in defense_name:
                    # Ensemble mode returns predictions directly
                    defended_preds = defense.apply(adversarial_images, self.model, return_predictions=True)
                    defended_preds = defended_preds.argmax(dim=1)
                else:
                    defended_outputs = self.model(defended_images)
                    defended_preds = defended_outputs.argmax(dim=1)
            
            # Update counters
            clean_correct += (clean_preds == labels).sum().item()
            adv_correct_no_defense += (adv_preds == labels).sum().item()
            adv_correct_with_defense += (defended_preds == labels).sum().item()
            total += batch_size
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.debug(f"Processed {total} samples...")
        
        # Calculate metrics
        clean_accuracy = clean_correct / total * 100
        adv_accuracy_no_defense = adv_correct_no_defense / total * 100
        adv_accuracy_with_defense = adv_correct_with_defense / total * 100
        
        defense_improvement = adv_accuracy_with_defense - adv_accuracy_no_defense
        relative_improvement = defense_improvement / (100 - adv_accuracy_no_defense) * 100
        
        results = {
            'defense_name': defense_name,
            'attack_name': attack_name,
            'num_samples': total,
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy_no_defense': adv_accuracy_no_defense,
            'adversarial_accuracy_with_defense': adv_accuracy_with_defense,
            'defense_improvement_absolute': defense_improvement,
            'defense_improvement_relative': relative_improvement,
            'defense_config': getattr(defense, 'config', {})
        }
        
        self.logger.info(f"{defense_name} vs {attack_name}:")
        self.logger.info(f"  Clean Accuracy: {clean_accuracy:.2f}%")
        self.logger.info(f"  Adv Accuracy (no defense): {adv_accuracy_no_defense:.2f}%")
        self.logger.info(f"  Adv Accuracy (with defense): {adv_accuracy_with_defense:.2f}%")
        self.logger.info(f"  Defense Improvement: {defense_improvement:.2f}%")
        
        return results
    
    def evaluate_all_defenses(self, 
                            attack_name: str = 'fgsm_epsilon_0.15',
                            num_samples: Optional[int] = 500) -> Dict[str, Any]:
        """
        Evaluate all defenses against a specific attack
        
        Args:
            attack_name: Attack to use for evaluation
            num_samples: Number of samples per defense
            
        Returns:
            Dictionary of all defense results
        """
        if attack_name not in self.attacks:
            raise ValueError(f"Attack {attack_name} not found")
        
        attack = self.attacks[attack_name]
        self.logger.info(f"Evaluating all defenses against {attack_name}...")
        
        all_results = {}
        
        for defense_name, defense in self.defenses.items():
            try:
                results = self.evaluate_defense_effectiveness(
                    defense_name, defense, attack_name, attack, num_samples
                )
                all_results[defense_name] = results
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {defense_name}: {e}")
                continue
        
        self.results['defense_results'] = all_results
        
        # Generate defense comparison
        self._generate_defense_comparison(all_results, attack_name)
        
        return all_results
    
    def _save_attack_results(self, 
                            attack_name: str, 
                            results: Dict[str, Any],
                            samples: Dict[str, Any]):
        """Save attack evaluation results"""
        from utils.visualization import visualize_attacks
        import matplotlib.pyplot as plt
        
        # Create directory
        eval_dir = Path(f"reports/metrics/robustness/attacks/{attack_name}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            safe_json_dump(results, f, indent=2)
        
        # Save samples
        if samples:
            samples_path = eval_dir / "samples.pt"
            torch.save(samples, samples_path)
            
            # Generate visualization
            if len(samples['clean_images']) > 0:
                fig = visualize_attacks(
                    samples['clean_images'],
                    samples['adversarial_images'],
                    {
                        'original': samples['clean_predictions'],
                        'adversarial': samples['adversarial_predictions']
                    }
                )
                
                viz_path = eval_dir / "attack_samples.png"
                fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        self.logger.debug(f"Saved {attack_name} results to {eval_dir}")
    
    def _generate_attack_comparison(self, all_results: Dict[str, Any]):
        """Generate attack comparison analysis"""
        # Create comparison DataFrame
        comparison_data = []
        
        for attack_name, results in all_results.items():
            row = {
                'Attack': attack_name,
                'Clean Accuracy (%)': results['clean_accuracy'],
                'Robust Accuracy (%)': results['robust_accuracy'],
                'Attack Success (%)': results['attack_success_rate'],
                'Robustness Gap (%)': results['robustness_gap'],
                'Avg Perturbation': results['avg_perturbation_norm'],
                'Avg Confidence Drop': results['avg_confidence_drop']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_dir = Path("reports/metrics/robustness/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = comparison_dir / "attack_comparison.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = comparison_dir / "attack_comparison.json"
        with open(json_path, 'w') as f:
            safe_json_dump(comparison_data, f, indent=2)
        
        # Generate visualization
        self._plot_attack_comparison(df, comparison_dir)
        
        self.logger.info(f"Saved attack comparison to {comparison_dir}")
        
        # Update main results
        self.results['comparison']['attacks'] = comparison_data
    
    def _generate_defense_comparison(self, 
                                    all_results: Dict[str, Any],
                                    attack_name: str):
        """Generate defense comparison analysis"""
        # Create comparison DataFrame
        comparison_data = []
        
        for defense_name, results in all_results.items():
            row = {
                'Defense': defense_name,
                'Attack': attack_name,
                'Clean Accuracy (%)': results['clean_accuracy'],
                'Adv Accuracy (No Defense) (%)': results['adversarial_accuracy_no_defense'],
                'Adv Accuracy (With Defense) (%)': results['adversarial_accuracy_with_defense'],
                'Defense Improvement (%)': results['defense_improvement_absolute'],
                'Relative Improvement (%)': results['defense_improvement_relative']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_dir = Path("reports/metrics/robustness/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = comparison_dir / f"defense_comparison_{attack_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = comparison_dir / f"defense_comparison_{attack_name}.json"
        with open(json_path, 'w') as f:
            safe_json_dump(comparison_data, f, indent=2)
        
        # Generate visualization
        self._plot_defense_comparison(df, comparison_dir, attack_name)
        
        self.logger.info(f"Saved defense comparison to {comparison_dir}")
        
        # Update main results
        if 'defenses' not in self.results['comparison']:
            self.results['comparison']['defenses'] = {}
        self.results['comparison']['defenses'][attack_name] = comparison_data
    
    def _plot_attack_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Plot attack comparison visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        setup_plotting()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Attack success rates
        ax1 = axes[0, 0]
        sns.barplot(data=df, x='Attack', y='Attack Success (%)', ax=ax1)
        ax1.set_title('Attack Success Rates')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_ylabel('Success Rate (%)')
        
        # Plot 2: Robustness gap
        ax2 = axes[0, 1]
        sns.barplot(data=df, x='Attack', y='Robustness Gap (%)', ax=ax2)
        ax2.set_title('Robustness Gap (Clean - Robust Accuracy)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_ylabel('Gap (%)')
        
        # Plot 3: Perturbation vs success rate
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['Avg Perturbation'], df['Attack Success (%)'], 
                             c=df['Avg Confidence Drop'], cmap='viridis', s=100)
        ax3.set_xlabel('Average Perturbation Norm')
        ax3.set_ylabel('Attack Success Rate (%)')
        ax3.set_title('Perturbation vs Success Rate')
        plt.colorbar(scatter, ax=ax3, label='Avg Confidence Drop')
        
        # Add attack names to points
        for i, row in df.iterrows():
            ax3.annotate(row['Attack'], 
                        (row['Avg Perturbation'], row['Attack Success (%)']),
                        fontsize=8, alpha=0.7)
        
        # Plot 4: Clean vs robust accuracy
        ax4 = axes[1, 1]
        x = np.arange(len(df))
        width = 0.35
        
        ax4.bar(x - width/2, df['Clean Accuracy (%)'], width, label='Clean Accuracy')
        ax4.bar(x + width/2, df['Robust Accuracy (%)'], width, label='Robust Accuracy')
        
        ax4.set_xlabel('Attack')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Clean vs Robust Accuracy')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['Attack'], rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_dir / "attack_comparison_plot.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_defense_comparison(self, df: pd.DataFrame, output_dir: Path, attack_name: str):
        """Plot defense comparison visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        setup_plotting()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0]
        
        x = np.arange(len(df))
        width = 0.25
        
        ax1.bar(x - width, df['Clean Accuracy (%)'], width, label='Clean', alpha=0.8)
        ax1.bar(x, df['Adv Accuracy (No Defense) (%)'], width, label='No Defense', alpha=0.8)
        ax1.bar(x + width, df['Adv Accuracy (With Defense) (%)'], width, label='With Defense', alpha=0.8)
        
        ax1.set_xlabel('Defense')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'Defense Effectiveness against {attack_name}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Defense'], rotation=45, ha='right')
        ax1.legend()
        
        # Plot 2: Defense improvement
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in df['Defense Improvement (%)']]
        ax2.bar(df['Defense'], df['Defense Improvement (%)'], color=colors, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_xlabel('Defense')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Defense Improvement (Absolute)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(df['Defense Improvement (%)']):
            ax2.text(i, v + (0.5 if v >= 0 else -2), f'{v:.1f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_dir / f"defense_comparison_{attack_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def save_final_report(self):
        """Save comprehensive evaluation report"""
        report_dir = Path("reports/metrics/robustness")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        report_path = report_dir / "comprehensive_evaluation.json"
        with open(report_path, 'w') as f:
            safe_json_dump(self.results, f, indent=2)
        
        # Generate summary report
        summary_path = report_dir / "evaluation_summary.md"
        self._generate_summary_report(summary_path)
        
        self.logger.info(f"Saved comprehensive evaluation report to {report_dir}")
    
    def _generate_summary_report(self, output_path: Path):
        """Generate markdown summary report"""
        lines = [
            "# Robustness Evaluation Summary",
            "",
            f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Information",
            f"- **Model:** MNIST CNN",
            f"- **Clean Accuracy:** {self.results['clean_performance'].get('accuracy', 'N/A'):.2f}%",
            f"- **Parameters:** {self.model_metadata.get('parameters', 'N/A'):,}",
            "",
            "## Attack Robustness Summary",
            "| Attack | Clean Acc (%) | Robust Acc (%) | Success Rate (%) | Avg Perturbation |",
            "|--------|---------------|----------------|------------------|------------------|"
        ]
        
        if 'attack_results' in self.results:
            for attack_name, results in self.results['attack_results'].items():
                line = (
                    f"| {attack_name} | "
                    f"{results['clean_accuracy']:.2f} | "
                    f"{results['robust_accuracy']:.2f} | "
                    f"{results['attack_success_rate']:.2f} | "
                    f"{results['avg_perturbation_norm']:.4f} |"
                )
                lines.append(line)
        
        lines.extend([
            "",
            "## Defense Effectiveness Summary",
            "| Defense | Attack | No Defense (%) | With Defense (%) | Improvement (%) |",
            "|---------|--------|----------------|------------------|-----------------|"
        ])
        
        if 'defense_results' in self.results:
            for defense_name, results in self.results['defense_results'].items():
                line = (
                    f"| {defense_name} | "
                    f"{results['attack_name']} | "
                    f"{results['adversarial_accuracy_no_defense']:.2f} | "
                    f"{results['adversarial_accuracy_with_defense']:.2f} | "
                    f"{results['defense_improvement_absolute']:.2f} |"
                )
                lines.append(line)
        
        lines.extend([
            "",
            "## Key Findings",
            "",
            "### Most Effective Attacks",
            "1. **Based on Success Rate:**",
            "2. **Based on Stealth (Low Perturbation):**",
            "3. **Based on Confidence Drop:**",
            "",
            "### Most Effective Defenses",
            "1. **Best Overall Protection:**",
            "2. **Best for Computational Efficiency:**",
            "3. **Best Trade-off (Accuracy vs Protection):**",
            "",
            "## Recommendations",
            "1. **For Critical Systems:** Use ensemble of defenses",
            "2. **For Real-time Systems:** Use input smoothing with adaptive threshold",
            "3. **For Maximum Protection:** Use adversarial training with PGD",
            "",
            "---",
            "*Generated by Adversarial ML Security Suite*"
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

def main():
    """Main entry point"""
    import matplotlib.pyplot as plt
    
    # Setup
    from utils.visualization import setup_plotting
    setup_plotting()
    
    # Initialize evaluator
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION PIPELINE")
    print("="*60)
    
    evaluator = RobustnessEvaluator()
    
    # 1. Evaluate clean performance
    print("\n1. Evaluating clean performance...")
    clean_results = evaluator.evaluate_clean_performance()
    print(f"   Clean Accuracy: {clean_results['accuracy']:.2f}%")
    
    # 2. Evaluate attack robustness
    print("\n2. Evaluating attack robustness...")
    attack_results = evaluator.evaluate_all_attacks(num_samples=1000)
    print(f"   Evaluated {len(attack_results)} attacks")
    
    # 3. Evaluate defense effectiveness
    print("\n3. Evaluating defense effectiveness...")
    defense_results = evaluator.evaluate_all_defenses(
        attack_name='fgsm_epsilon_0.15',
        num_samples=500
    )
    print(f"   Evaluated {len(defense_results)} defenses")
    
    # 4. Save comprehensive report
    print("\n4. Generating comprehensive report...")
    evaluator.save_final_report()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  - reports/metrics/robustness/")
    print("  - reports/metrics/robustness/comparison/")
    print("  - reports/metrics/robustness/attacks/")
    print("\nVisualizations saved to:")
    print("  - reports/figures/")
    print("="*60)
    
    # Print key findings
    if attack_results:
        best_attack = min(attack_results.items(), 
                         key=lambda x: x[1]['robust_accuracy'])
        worst_attack = max(attack_results.items(),
                          key=lambda x: x[1]['robust_accuracy'])
        
        print(f"\nKey Findings:")
        print(f"  Most Effective Attack: {best_attack[0]}")
        print(f"    Robust Accuracy: {best_attack[1]['robust_accuracy']:.2f}%")
        print(f"  Least Effective Attack: {worst_attack[0]}")
        print(f"    Robust Accuracy: {worst_attack[1]['robust_accuracy']:.2f}%")
    
    if defense_results:
        best_defense = max(defense_results.items(),
                          key=lambda x: x[1]['defense_improvement_absolute'])
        
        print(f"  Best Defense: {best_defense[0]}")
        print(f"    Improvement: {best_defense[1]['defense_improvement_absolute']:.2f}%")

if __name__ == "__main__":
    main()
