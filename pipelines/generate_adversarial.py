"""
Adversarial Example Generation Pipeline
Enterprise-grade with comprehensive metrics and visualization
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from attacks.fgsm import FGSMAttack, create_fgsm_attack
from attacks.pgd import PGDAttack, create_pgd_attack
from attacks.deepfool import DeepFoolAttack, create_deepfool_attack
from utils.model_utils import load_model
from utils.dataset_utils import load_mnist
from utils.visualization import visualize_attacks, setup_plotting
from utils.logging_utils import setup_logger

class AdversarialGenerator:
    """Complete adversarial example generation pipeline"""
    
    def __init__(self, config_path: str = "config/attack_config.yaml"):
        """
        Initialize adversarial generator
        
        Args:
            config_path: Path to attack configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.logger = setup_logger('adversarial_generator', 'reports/logs/adversarial_generation.log')
        
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
        
        # Initialize attacks
        self._init_attacks()
        
        # Results storage
        self.results = {}
    
    def _init_attacks(self):
        """Initialize all configured attacks"""
        self.attacks = {}
        
        # FGSM
        if 'fgsm' in self.config:
            self.attacks['fgsm'] = create_fgsm_attack(
                self.model,
                **self.config['fgsm']
            )
            self.logger.info(f"Initialized FGSM attack with epsilon={self.config['fgsm'].get('epsilon', 0.15)}")
        
        # PGD
        if 'pgd' in self.config:
            self.attacks['pgd'] = create_pgd_attack(
                self.model,
                **self.config['pgd']
            )
            self.logger.info(f"Initialized PGD attack with epsilon={self.config['pgd'].get('epsilon', 0.3)}")
        
        # DeepFool
        if 'deepfool' in self.config:
            self.attacks['deepfool'] = create_deepfool_attack(
                self.model,
                **self.config['deepfool']
            )
            self.logger.info(f"Initialized DeepFool attack with max_iter={self.config['deepfool'].get('max_iter', 50)}")
    
    def generate_for_attack(self, 
                           attack_name: str, 
                           num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate adversarial examples for a specific attack
        
        Args:
            attack_name: Name of attack to use
            num_samples: Number of samples to generate (None for all)
            
        Returns:
            Dictionary of results
        """
        if attack_name not in self.attacks:
            raise ValueError(f"Attack {attack_name} not initialized")
        
        attack = self.attacks[attack_name]
        self.logger.info(f"Generating adversarial examples using {attack_name.upper()}...")
        
        # Collect samples
        all_clean = []
        all_adv = []
        all_labels = []
        all_clean_preds = []
        all_adv_preds = []
        
        sample_count = 0
        total_batches = len(self.test_loader)
        
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            if num_samples and sample_count >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples
            if attack_name == 'deepfool':
                adversarial_images = attack.generate(images)
            else:
                adversarial_images = attack.generate(images, labels)
            
            # Get predictions
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_preds = clean_outputs.argmax(dim=1)
                
                adv_outputs = self.model(adversarial_images)
                adv_preds = adv_outputs.argmax(dim=1)
            
            # Store results
            batch_size = images.size(0)
            if num_samples:
                remaining = num_samples - sample_count
                take = min(batch_size, remaining)
                
                all_clean.append(images[:take].cpu())
                all_adv.append(adversarial_images[:take].cpu())
                all_labels.append(labels[:take].cpu())
                all_clean_preds.append(clean_preds[:take].cpu())
                all_adv_preds.append(adv_preds[:take].cpu())
                
                sample_count += take
            else:
                all_clean.append(images.cpu())
                all_adv.append(adversarial_images.cpu())
                all_labels.append(labels.cpu())
                all_clean_preds.append(clean_preds.cpu())
                all_adv_preds.append(adv_preds.cpu())
                
                sample_count += batch_size
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.debug(f"Processed {sample_count} samples...")
        
        # Combine results
        clean_images = torch.cat(all_clean, dim=0)
        adversarial_images = torch.cat(all_adv, dim=0)
        labels = torch.cat(all_labels, dim=0)
        clean_preds = torch.cat(all_clean_preds, dim=0)
        adv_preds = torch.cat(all_adv_preds, dim=0)
        
        # Calculate metrics
        clean_accuracy = (clean_preds == labels).float().mean().item() * 100
        adversarial_accuracy = (adv_preds == labels).float().mean().item() * 100
        attack_success_rate = 100 - adversarial_accuracy
        
        # Perturbation metrics
        perturbations = adversarial_images - clean_images
        l2_norms = torch.norm(perturbations.view(perturbations.size(0), -1), p=2, dim=1)
        linf_norms = torch.norm(perturbations.view(perturbations.size(0), -1), p=float('inf'), dim=1)
        
        # Confidence metrics
        with torch.no_grad():
            clean_outputs = self.model(clean_images.to(self.device))
            adv_outputs = self.model(adversarial_images.to(self.device))
            
            clean_probs = torch.softmax(clean_outputs, dim=1)
            adv_probs = torch.softmax(adv_outputs, dim=1)
            
            clean_confidence = clean_probs.max(dim=1)[0].mean().item()
            adv_confidence = adv_probs.max(dim=1)[0].mean().item()
        
        # Compile results
        results = {
            'attack_name': attack_name,
            'num_samples': sample_count,
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'attack_success_rate': attack_success_rate,
            'avg_l2_perturbation': l2_norms.mean().item(),
            'avg_linf_perturbation': linf_norms.mean().item(),
            'clean_confidence': clean_confidence,
            'adversarial_confidence': adv_confidence,
            'attack_config': self.config.get(attack_name, {}),
            'generation_timestamp': str(datetime.now())
        }
        
        # Store samples for visualization
        sample_results = {
            'clean_images': clean_images[:10],  # Store first 10 for visualization
            'adversarial_images': adversarial_images[:10],
            'labels': labels[:10],
            'clean_predictions': clean_preds[:10],
            'adversarial_predictions': adv_preds[:10]
        }
        
        self.logger.info(f"{attack_name.upper()} Results:")
        self.logger.info(f"  Clean Accuracy: {clean_accuracy:.2f}%")
        self.logger.info(f"  Adversarial Accuracy: {adversarial_accuracy:.2f}%")
        self.logger.info(f"  Attack Success Rate: {attack_success_rate:.2f}%")
        self.logger.info(f"  Avg L2 Perturbation: {l2_norms.mean().item():.4f}")
        self.logger.info(f"  Avg Linf Perturbation: {linf_norms.mean().item():.4f}")
        
        return results, sample_results
    
    def generate_all(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate adversarial examples for all configured attacks
        
        Args:
            num_samples: Number of samples per attack
            
        Returns:
            Dictionary of all results
        """
        all_results = {}
        all_samples = {}
        
        for attack_name in self.attacks.keys():
            try:
                results, samples = self.generate_for_attack(attack_name, num_samples)
                all_results[attack_name] = results
                all_samples[attack_name] = samples
                
                # Save attack-specific results
                self._save_attack_results(attack_name, results, samples)
                
            except Exception as e:
                self.logger.error(f"Failed to generate {attack_name} adversarial examples: {e}")
                continue
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        # Generate comparison visualization
        self._generate_comparison_visualization(all_samples)
        
        return all_results
    
    def _save_attack_results(self, 
                            attack_name: str, 
                            results: Dict[str, Any],
                            samples: Dict[str, Any]):
        """Save attack-specific results"""
        import matplotlib.pyplot as plt
        
        # Create directory for this attack
        attack_dir = Path(f"reports/metrics/attacks/{attack_name}")
        attack_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = attack_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save samples for later analysis
        samples_path = attack_dir / "samples.pt"
        torch.save(samples, samples_path)
        
        # Generate visualization
        if len(samples['clean_images']) > 0:
            from utils.visualization import visualize_attacks
            fig = visualize_attacks(
                samples['clean_images'],
                samples['adversarial_images'],
                {
                    'original': samples['clean_predictions'],
                    'adversarial': samples['adversarial_predictions']
                }
            )
            
            visualization_path = attack_dir / "visualization.png"
            fig.savefig(visualization_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        self.logger.info(f"Saved {attack_name} results to {attack_dir}")
    
    def _save_comprehensive_results(self, all_results: Dict[str, Any]):
        """Save comprehensive comparison results"""
        # Create comparison report
        comparison = {
            'model': self.model_metadata,
            'generation_timestamp': str(datetime.now()),
            'attacks': all_results,
            'summary': self._create_summary(all_results)
        }
        
        # Save comparison
        comparison_dir = Path("reports/metrics/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_path = comparison_dir / "attack_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Generate comparison table
        self._generate_comparison_table(all_results, comparison_dir)
        
        self.logger.info(f"Saved comprehensive results to {comparison_dir}")
    
    def _create_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics"""
        summary = {
            'best_attack': None,
            'worst_attack': None,
            'most_stealthy_attack': None,
            'most_disruptive_attack': None
        }
        
        if not all_results:
            return summary
        
        # Find best attack (lowest adversarial accuracy)
        best_attack = min(all_results.items(), 
                         key=lambda x: x[1]['adversarial_accuracy'])
        summary['best_attack'] = {
            'name': best_attack[0],
            'adversarial_accuracy': best_attack[1]['adversarial_accuracy']
        }
        
        # Find worst attack (highest adversarial accuracy)
        worst_attack = max(all_results.items(),
                          key=lambda x: x[1]['adversarial_accuracy'])
        summary['worst_attack'] = {
            'name': worst_attack[0],
            'adversarial_accuracy': worst_attack[1]['adversarial_accuracy']
        }
        
        # Find most stealthy attack (smallest perturbation)
        stealthy_attack = min(all_results.items(),
                             key=lambda x: x[1]['avg_l2_perturbation'])
        summary['most_stealthy_attack'] = {
            'name': stealthy_attack[0],
            'avg_l2_perturbation': stealthy_attack[1]['avg_l2_perturbation']
        }
        
        # Find most disruptive attack (largest success rate)
        disruptive_attack = max(all_results.items(),
                              key=lambda x: x[1]['attack_success_rate'])
        summary['most_disruptive_attack'] = {
            'name': disruptive_attack[0],
            'attack_success_rate': disruptive_attack[1]['attack_success_rate']
        }
        
        return summary
    
    def _generate_comparison_table(self, 
                                  all_results: Dict[str, Any],
                                  output_dir: Path):
        """Generate comparison table in markdown format"""
        table_lines = [
            "# Adversarial Attack Comparison",
            "",
            "| Attack | Clean Acc (%) | Adv Acc (%) | Success Rate (%) | Avg L2 | Avg Linf |",
            "|--------|---------------|-------------|------------------|--------|--------|"
        ]
        
        for attack_name, results in all_results.items():
            row = (
                f"| {attack_name.upper()} | "
                f"{results['clean_accuracy']:.2f} | "
                f"{results['adversarial_accuracy']:.2f} | "
                f"{results['attack_success_rate']:.2f} | "
                f"{results['avg_l2_perturbation']:.4f} | "
                f"{results['avg_linf_perturbation']:.4f} |"
            )
            table_lines.append(row)
        
        # Save table
        table_path = output_dir / "comparison_table.md"
        with open(table_path, 'w') as f:
            f.write('\n'.join(table_lines))
    
    def _generate_comparison_visualization(self, all_samples: Dict[str, Any]):
        """Generate comparison visualization"""
        if not all_samples:
            return
        
        import matplotlib.pyplot as plt
        
        # Use first attack as reference
        first_attack = list(all_samples.keys())[0]
        clean_images = all_samples[first_attack]['clean_images']
        labels = all_samples[first_attack]['labels']
        
        # Create comparison figure
        n_attacks = len(all_samples)
        n_samples = min(5, len(clean_images))
        
        fig, axes = plt.subplots(n_samples, n_attacks + 1, figsize=(3 * (n_attacks + 1), 3 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Plot original images
        for i in range(n_samples):
            axes[i, 0].imshow(clean_images[i].squeeze(), cmap='gray')
            axes[i, 0].set_title(f"Original\nLabel: {labels[i].item()}")
            axes[i, 0].axis('off')
        
        # Plot adversarial images for each attack
        for j, (attack_name, samples) in enumerate(all_samples.items(), 1):
            adv_images = samples['adversarial_images']
            adv_preds = samples['adversarial_predictions']
            
            for i in range(n_samples):
                axes[i, j].imshow(adv_images[i].squeeze(), cmap='gray')
                axes[i, j].set_title(f"{attack_name.upper()}\nPred: {adv_preds[i].item()}")
                axes[i, j].axis('off')
        
        plt.suptitle('Adversarial Attack Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        comparison_dir = Path("reports/figures/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = comparison_dir / "attack_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved comparison visualization to {fig_path}")

def main():
    """Main entry point"""
    import matplotlib.pyplot as plt
    
    # Setup plotting
    from utils.visualization import setup_plotting
    setup_plotting()
    
    # Initialize generator
    generator = AdversarialGenerator()
    
    # Generate adversarial examples
    print("\n" + "="*60)
    print("ADVERSARIAL EXAMPLE GENERATION")
    print("="*60)
    
    results = generator.generate_all(num_samples=1000)
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*60)
    
    if results:
        for attack_name, attack_results in results.items():
            print(f"\n{attack_name.upper()}:")
            print(f"  Clean Accuracy: {attack_results['clean_accuracy']:.2f}%")
            print(f"  Adversarial Accuracy: {attack_results['adversarial_accuracy']:.2f}%")
            print(f"  Attack Success Rate: {attack_results['attack_success_rate']:.2f}%")
            print(f"  Avg L2 Perturbation: {attack_results['avg_l2_perturbation']:.4f}")
        
        # Find best attack
        best_attack = min(results.items(), 
                         key=lambda x: x[1]['adversarial_accuracy'])
        print(f"\nMost Effective Attack: {best_attack[0].upper()}")
        print(f"  Adversarial Accuracy: {best_attack[1]['adversarial_accuracy']:.2f}%")
    
    print("\nResults saved to:")
    print("  - reports/metrics/attacks/")
    print("  - reports/metrics/comparison/")
    print("  - reports/figures/comparison/")
    print("="*60)

if __name__ == "__main__":
    main()
