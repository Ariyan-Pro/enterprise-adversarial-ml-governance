"""
Report Export Pipeline
Enterprise-grade report generation and export
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualization import setup_plotting
from utils.logging_utils import setup_logger

class ReportExporter:
    """Comprehensive report exporter"""
    
    def __init__(self):
        """Initialize report exporter"""
        self.logger = setup_logger('report_exporter', 'reports/logs/export.log')
        setup_plotting()
        
        # Define report sections
        self.sections = [
            'executive_summary',
            'model_performance',
            'attack_analysis',
            'defense_evaluation',
            'recommendations',
            'appendix'
        ]
    
    def load_all_results(self) -> Dict[str, Any]:
        """Load all available results"""
        results = {
            'model_info': {},
            'training_results': {},
            'attack_results': {},
            'defense_results': {},
            'robustness_evaluation': {},
            'defense_training': {}
        }
        
        # Load model information
        model_card_path = Path("models/pretrained/model_card.json")
        if model_card_path.exists():
            with open(model_card_path, 'r') as f:
                results['model_info'] = json.load(f)
        
        # Load training results
        training_logs = Path("reports/logs/training_metrics.json")
        if training_logs.exists():
            with open(training_logs, 'r') as f:
                results['training_results'] = json.load(f)
        
        # Load attack results
        attack_comparison = Path("reports/metrics/comparison/attack_comparison.json")
        if attack_comparison.exists():
            with open(attack_comparison, 'r') as f:
                results['attack_results'] = json.load(f)
        
        # Load defense results
        defense_comparison = Path("reports/metrics/comparison/defense_comparison_fgsm_epsilon_0.15.json")
        if defense_comparison.exists():
            with open(defense_comparison, 'r') as f:
                results['defense_results'] = json.load(f)
        
        # Load robustness evaluation
        robustness_report = Path("reports/metrics/robustness/comprehensive_evaluation.json")
        if robustness_report.exists():
            with open(robustness_report, 'r') as f:
                results['robustness_evaluation'] = json.load(f)
        
        # Load defense training
        defense_training = Path("reports/metrics/defense_training/defense_training_results.json")
        if defense_training.exists():
            with open(defense_training, 'r') as f:
                results['defense_training'] = json.load(f)
        
        return results
    
    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary"""
        summary = [
            "# EXECUTIVE SUMMARY",
            "",
            "## Project Overview",
            "This report summarizes the security assessment of the MNIST CNN model ",
            "against various adversarial attacks and evaluates the effectiveness of ",
            "multiple defense mechanisms.",
            "",
            "## Key Findings",
            ""
        ]
        
        # Extract key metrics
        if 'robustness_evaluation' in results and results['robustness_evaluation']:
            robustness = results['robustness_evaluation']
            
            # Clean accuracy
            clean_acc = robustness.get('clean_performance', {}).get('accuracy', 'N/A')
            summary.append(f"- **Baseline Model Accuracy:** {clean_acc}")
            
            # Best attack
            attack_results = robustness.get('attack_results', {})
            if attack_results:
                best_attack = min(attack_results.items(), 
                                 key=lambda x: x[1].get('robust_accuracy', 100))
                summary.append(f"- **Most Effective Attack:** {best_attack[0]} "
                              f"(Reduces accuracy to {best_attack[1].get('robust_accuracy', 'N/A')}%)")
            
            # Best defense
            defense_results = robustness.get('defense_results', {})
            if defense_results:
                best_defense = max(defense_results.items(),
                                  key=lambda x: x[1].get('defense_improvement_absolute', 0))
                summary.append(f"- **Most Effective Defense:** {best_defense[0]} "
                              f"(Improves accuracy by {best_defense[1].get('defense_improvement_absolute', 'N/A')}%)")
        
        summary.extend([
            "",
            "## Risk Assessment",
            "- **Critical Risk:** High susceptibility to PGD attacks",
            "- **Medium Risk:** Moderate vulnerability to FGSM attacks",
            "- **Low Risk:** Good robustness against simple perturbations",
            "",
            "## Recommendations",
            "1. Implement adversarial training for critical deployments",
            "2. Use input smoothing as a lightweight defense",
            "3. Deploy ensemble models for high-security applications",
            "4. Regular security audits and adversarial testing",
            ""
        ])
        
        return '\n'.join(summary)
    
    def generate_model_performance(self, results: Dict[str, Any]) -> str:
        """Generate model performance section"""
        content = [
            "# MODEL PERFORMANCE",
            "",
            "## Baseline Model",
            ""
        ]
        
        if 'model_info' in results and results['model_info']:
            model_info = results['model_info']
            content.extend([
                f"- **Model Architecture:** {model_info.get('model_class', 'MNIST CNN')}",
                f"- **Total Parameters:** {model_info.get('parameters', 'N/A')}",
                f"- **Input Dimensions:** {model_info.get('input_size', '28x28')}",
                f"- **Number of Classes:** {model_info.get('num_classes', 10)}",
                ""
            ])
        
        if 'training_results' in results and results['training_results']:
            training = results['training_results']
            if isinstance(training, list) and len(training) > 0:
                final_epoch = training[-1]
                content.extend([
                    "## Training Performance",
                    f"- **Final Training Accuracy:** {final_epoch.get('train', {}).get('accuracy', 'N/A')}%",
                    f"- **Final Validation Accuracy:** {final_epoch.get('validation', {}).get('accuracy', 'N/A')}%",
                    f"- **Training Loss:** {final_epoch.get('train', {}).get('loss', 'N/A'):.4f}",
                    ""
                ])
        
        if 'robustness_evaluation' in results and results['robustness_evaluation']:
            robustness = results['robustness_evaluation']
            clean_perf = robustness.get('clean_performance', {})
            
            content.extend([
                "## Clean Data Performance",
                f"- **Test Accuracy:** {clean_perf.get('accuracy', 'N/A')}%",
                f"- **Test Loss:** {clean_perf.get('loss', 'N/A'):.4f}",
                f"- **Mean Confidence:** {clean_perf.get('mean_confidence', 'N/A'):.3f}",
                ""
            ])
        
        return '\n'.join(content)
    
    def generate_attack_analysis(self, results: Dict[str, Any]) -> str:
        """Generate attack analysis section"""
        content = [
            "# ATTACK ANALYSIS",
            "",
            "## Overview of Evaluated Attacks",
            ""
        ]
        
        attack_descriptions = {
            'fgsm': "Fast Gradient Sign Method: Single-step attack using gradient sign",
            'pgd': "Projected Gradient Descent: Iterative attack with projection",
            'deepfool': "DeepFool: Minimal perturbation attack for misclassification"
        }
        
        for attack_name, description in attack_descriptions.items():
            content.append(f"- **{attack_name.upper()}:** {description}")
        
        content.append("")
        
        if 'robustness_evaluation' in results and results['robustness_evaluation']:
            robustness = results['robustness_evaluation']
            attack_results = robustness.get('attack_results', {})
            
            if attack_results:
                content.append("## Attack Performance Summary")
                content.append("")
                content.append("| Attack | Clean Acc (%) | Robust Acc (%) | Success Rate (%) | Perturbation |")
                content.append("|--------|---------------|----------------|------------------|--------------|")
                
                for attack_name, metrics in attack_results.items():
                    row = (
                        f"| {attack_name} | "
                        f"{metrics.get('clean_accuracy', 'N/A'):.2f} | "
                        f"{metrics.get('robust_accuracy', 'N/A'):.2f} | "
                        f"{metrics.get('attack_success_rate', 'N/A'):.2f} | "
                        f"{metrics.get('avg_perturbation_norm', 'N/A'):.4f} |"
                    )
                    content.append(row)
                
                content.append("")
                
                # Add analysis
                content.append("## Key Insights")
                content.append("")
                
                # Find best and worst attacks
                if attack_results:
                    best_attack = min(attack_results.items(), 
                                     key=lambda x: x[1].get('robust_accuracy', 100))
                    worst_attack = max(attack_results.items(),
                                      key=lambda x: x[1].get('robust_accuracy', 0))
                    
                    content.append(f"1. **Most Effective Attack:** {best_attack[0]}")
                    content.append(f"   - Reduces accuracy to {best_attack[1].get('robust_accuracy', 'N/A')}%")
                    content.append(f"   - Success rate: {best_attack[1].get('attack_success_rate', 'N/A')}%")
                    content.append("")
                    
                    content.append(f"2. **Most Stealthy Attack:** Based on perturbation magnitude")
                    content.append("")
                    
                    content.append(f"3. **Least Effective Attack:** {worst_attack[0]}")
                    content.append(f"   - Model maintains {worst_attack[1].get('robust_accuracy', 'N/A')}% accuracy")
                    content.append("")
        
        return '\n'.join(content)
    
    def generate_defense_evaluation(self, results: Dict[str, Any]) -> str:
        """Generate defense evaluation section"""
        content = [
            "# DEFENSE EVALUATION",
            "",
            "## Overview of Evaluated Defenses",
            ""
        ]
        
        defense_descriptions = {
            'adversarial_training': "Trains model on adversarial examples to improve robustness",
            'input_smoothing': "Applies smoothing filters to input images to reduce perturbations",
            'randomized_transform': "Applies random transformations to break adversarial patterns",
            'ensemble': "Uses multiple models to make predictions, increasing robustness"
        }
        
        for defense_name, description in defense_descriptions.items():
            content.append(f"- **{defense_name.replace('_', ' ').title()}:** {description}")
        
        content.append("")
        
        if 'robustness_evaluation' in results and results['robustness_evaluation']:
            robustness = results['robustness_evaluation']
            defense_results = robustness.get('defense_results', {})
            
            if defense_results:
                content.append("## Defense Performance Summary")
                content.append("")
                content.append("| Defense | Attack | No Defense (%) | With Defense (%) | Improvement (%) |")
                content.append("|---------|--------|----------------|------------------|-----------------|")
                
                for defense_name, metrics in defense_results.items():
                    row = (
                        f"| {defense_name} | "
                        f"{metrics.get('attack_name', 'N/A')} | "
                        f"{metrics.get('adversarial_accuracy_no_defense', 'N/A'):.2f} | "
                        f"{metrics.get('adversarial_accuracy_with_defense', 'N/A'):.2f} | "
                        f"{metrics.get('defense_improvement_absolute', 'N/A'):.2f} |"
                    )
                    content.append(row)
                
                content.append("")
                
                # Add analysis
                content.append("## Key Insights")
                content.append("")
                
                if defense_results:
                    best_defense = max(defense_results.items(),
                                      key=lambda x: x[1].get('defense_improvement_absolute', 0))
                    worst_defense = min(defense_results.items(),
                                       key=lambda x: x[1].get('defense_improvement_absolute', 0))
                    
                    content.append(f"1. **Most Effective Defense:** {best_defense[0]}")
                    content.append(f"   - Improves accuracy by {best_defense[1].get('defense_improvement_absolute', 'N/A')}%")
                    content.append(f"   - Final accuracy: {best_defense[1].get('adversarial_accuracy_with_defense', 'N/A')}%")
                    content.append("")
                    
                    content.append(f"2. **Least Effective Defense:** {worst_defense[0]}")
                    content.append(f"   - Improves accuracy by {worst_defense[1].get('defense_improvement_absolute', 'N/A')}%")
                    content.append("")
                    
                    content.append("3. **Defense Trade-offs:**")
                    content.append("   - **Adversarial Training:** Best protection but requires retraining")
                    content.append("   - **Input Smoothing:** Lightweight but may reduce clean accuracy")
                    content.append("   - **Randomized Transformations:** Good balance of protection and efficiency")
                    content.append("")
        
        return '\n'.join(content)
    
    def generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        content = [
            "# RECOMMENDATIONS",
            "",
            "## Based on Evaluation Results",
            ""
        ]
        
        # Extract insights for recommendations
        if 'robustness_evaluation' in results and results['robustness_evaluation']:
            robustness = results['robustness_evaluation']
            
            content.append("### 1. For Critical Security Applications")
            content.append("   - Implement **adversarial training** with PGD attacks")
            content.append("   - Use **model ensembles** for increased robustness")
            content.append("   - Deploy **multiple defense layers** (defense in depth)")
            content.append("   - Regular **adversarial testing** in CI/CD pipeline")
            content.append("")
            
            content.append("### 2. For Real-time or Resource-Constrained Systems")
            content.append("   - Use **input smoothing** with adaptive thresholds")
            content.append("   - Implement **randomized transformations** with low computational cost")
            content.append("   - Consider **model distillation** for efficient robust models")
            content.append("")
            
            content.append("### 3. For Development and Testing")
            content.append("   - Integrate **FGSM testing** in unit tests")
            content.append("   - Perform **regular robustness audits**")
            content.append("   - Maintain **adversarial example datasets** for testing")
            content.append("   - Track **robustness metrics** alongside accuracy")
            content.append("")
            
            content.append("### 4. Monitoring and Maintenance")
            content.append("   - Monitor **prediction confidence distributions**")
            content.append("   - Set up **anomaly detection** for adversarial inputs")
            content.append("   - Regular **model retraining** with new adversarial examples")
            content.append("   - Keep **defense mechanisms updated** with new attack research")
            content.append("")
        
        return '\n'.join(content)
    
    def collect_visualizations(self) -> List[Path]:
        """Collect all visualization files"""
        viz_dir = Path("reports/figures")
        visualizations = []
        
        if viz_dir.exists():
            # Training visualizations
            training_viz = viz_dir / "training"
            if training_viz.exists():
                visualizations.extend(training_viz.glob("*.png"))
            
            # Attack comparison visualizations
            comparison_viz = viz_dir / "comparison"
            if comparison_viz.exists():
                visualizations.extend(comparison_viz.glob("*.png"))
            
            # Defense training visualizations
            defense_viz = viz_dir / "defense_training"
            if defense_viz.exists():
                visualizations.extend(defense_viz.glob("*.png"))
        
        return visualizations
    
    def create_pdf_report(self, markdown_content: str, output_path: Path):
        """Create PDF report with visualizations"""
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        self.logger.info(f"Creating PDF report: {output_path}")
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2E5A88')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.HexColor('#4A6FA5')
        )
        
        # Parse markdown content (simplified)
        story = []
        
        # Add title
        story.append(Paragraph("Adversarial ML Security Assessment Report", title_style))
        story.append(Spacer(1, 12))
        
        # Add metadata
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 24))
        
        # Add sections from markdown
        lines = markdown_content.split('\n')
        current_section = []
        
        for line in lines:
            if line.startswith('# '):
                # Main title
                if current_section:
                    story.append(Paragraph(''.join(current_section), styles['Normal']))
                    current_section = []
                story.append(Paragraph(line[2:], title_style))
                story.append(Spacer(1, 12))
            elif line.startswith('## '):
                # Section heading
                if current_section:
                    story.append(Paragraph(''.join(current_section), styles['Normal']))
                    current_section = []
                story.append(Paragraph(line[3:], heading_style))
                story.append(Spacer(1, 8))
            elif line.startswith('- **'):
                # Bold list item
                if current_section:
                    story.append(Paragraph(''.join(current_section), styles['Normal']))
                    current_section = []
                # Extract bold text
                import re
                bold_match = re.match(r'- \*\*(.*?)\*\*: (.*)', line)
                if bold_match:
                    bold_text, rest = bold_match.groups()
                    story.append(Paragraph(f"<b>{bold_text}:</b> {rest}", styles['Normal']))
            elif line.strip() == '':
                # Empty line
                if current_section:
                    story.append(Paragraph(''.join(current_section), styles['Normal']))
                    current_section = []
                story.append(Spacer(1, 12))
            else:
                # Regular text
                current_section.append(line + ' ')
        
        if current_section:
            story.append(Paragraph(''.join(current_section), styles['Normal']))
        
        # Add visualizations
        visualizations = self.collect_visualizations()
        if visualizations:
            story.append(Spacer(1, 24))
            story.append(Paragraph("## Visualizations", heading_style))
            story.append(Spacer(1, 12))
            
            for viz_path in visualizations[:5]:  # Limit to 5 visualizations
                try:
                    # Add visualization with caption
                    img = Image(str(viz_path), width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(f"Figure: {viz_path.stem}", styles['Italic']))
                    story.append(Spacer(1, 12))
                except:
                    continue
        
        # Build PDF
        doc.build(story)
        self.logger.info(f"PDF report created: {output_path}")
    
    def export_all(self, output_dir: str = "reports/enterprise_summary"):
        """Export all report formats"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load results
        results = self.load_all_results()
        
        # Generate markdown report
        markdown_report = []
        markdown_report.append(self.generate_executive_summary(results))
        markdown_report.append(self.generate_model_performance(results))
        markdown_report.append(self.generate_attack_analysis(results))
        markdown_report.append(self.generate_defense_evaluation(results))
        markdown_report.append(self.generate_recommendations(results))
        
        full_markdown = '\n\n'.join(markdown_report)
        
        # Save markdown
        md_path = output_path / "security_assessment.md"
        with open(md_path, 'w') as f:
            f.write(full_markdown)
        
        # Save JSON
        json_path = output_path / "full_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create PDF
        pdf_path = output_path / "enterprise_summary.pdf"
        self.create_pdf_report(full_markdown, pdf_path)
        
        # Create HTML
        html_path = output_path / "report.html"
        self.create_html_report(full_markdown, html_path)
        
        self.logger.info(f"Reports exported to: {output_path}")
        self.logger.info(f"  - Markdown: {md_path}")
        self.logger.info(f"  - JSON: {json_path}")
        self.logger.info(f"  - PDF: {pdf_path}")
        self.logger.info(f"  - HTML: {html_path}")
        
        return {
            'markdown': md_path,
            'json': json_path,
            'pdf': pdf_path,
            'html': html_path
        }
    
    def create_html_report(self, markdown_content: str, output_path: Path):
        """Create HTML report"""
        import markdown
        
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content, extensions=['tables'])
        
        # Create full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Adversarial ML Security Assessment</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2E5A88;
                    border-bottom: 2px solid #4A6FA5;
                    padding-bottom: 10px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #4A6FA5;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .header {{
                    background: linear-gradient(135deg, #2E5A88, #4A6FA5);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 2px solid #ddd;
                    color: #666;
                    font-size: 0.9em;
                }}
                .visualization {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .recommendation {{
                    background-color: #f0f7ff;
                    border-left: 4px solid #4A6FA5;
                    padding: 15px;
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Adversarial ML Security Assessment Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                {html_content}
            </div>
            
            <div class="footer">
                <p>Generated by Adversarial ML Security Suite | Confidential</p>
                <p>This report contains sensitive security information. Handle with appropriate confidentiality measures.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(full_html)

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("REPORT EXPORT PIPELINE")
    print("="*60)
    
    exporter = ReportExporter()
    
    print("\n1. Loading results...")
    results = exporter.load_all_results()
    print(f"   Loaded {len(results)} result categories")
    
    print("\n2. Generating reports...")
    export_paths = exporter.export_all()
    
    print("\n" + "="*60)
    print("REPORT EXPORT COMPLETE")
    print("="*60)
    print("\nReports generated:")
    for format_name, path in export_paths.items():
        print(f"  - {format_name.upper()}: {path}")
    
    print("\nTo view the HTML report:")
    print(f"  open {export_paths['html']}")
    print("="*60)

if __name__ == "__main__":
    main()
