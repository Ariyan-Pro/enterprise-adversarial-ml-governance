"""
Verification script for the Adversarial ML Suite - FIXED
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"? Python 3.8+ required. Found {sys.version}")
        return False
    print(f"? Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_imports():
    """Check all required imports"""
    requirements = [
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'yaml',
        'tqdm',
        'termcolor',
        'sklearn',
        'pandas',
        'seaborn',
        'cv2'
    ]
    
    print("\nChecking imports...")
    all_good = True
    
    for req in requirements:
        try:
            importlib.import_module(req)
            print(f"? {req}")
        except ImportError as e:
            print(f"? {req}: {e}")
            all_good = False
    
    return all_good

def check_structure():
    """Check project structure"""
    print("\nChecking project structure...")
    
    required_dirs = [
        'config',
        'data/raw',
        'data/processed',
        'models/base',
        'models/pretrained',
        'attacks',
        'defenses',
        'pipelines',
        'utils',
        'notebooks',
        'reports/figures',
        'reports/metrics',
        'reports/logs'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        '.gitignore',
        'pyproject.toml',
        'config/training_config.yaml',
        'config/attack_config.yaml',
        'config/defense_config.yaml',
        'config/eval_config.yaml',
        'models/base/mnist_cnn.py',
        'models/registry.json',
        'utils/dataset_utils.py',
        'utils/model_utils.py',
        'utils/logging_utils.py',
        'utils/visualization.py',
        'pipelines/train_model.py',
        'pipelines/generate_adversarial.py',
        'pipelines/robustness_eval.py',
        'pipelines/defense_train.py',
        'pipelines/export_report.py',
        'attacks/fgsm.py',
        'attacks/pgd.py',
        'attacks/deepfool.py',
        'defenses/adv_training.py',
        'defenses/input_smoothing.py',
        'defenses/randomized_transform.py',
        'defenses/model_wrappers.py',
        'verify_installation.py'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"? Directory: {directory}")
        else:
            print(f"? Missing directory: {directory}")
            all_good = False
    
    for file in required_files:
        if Path(file).exists():
            print(f"? File: {file}")
        else:
            print(f"? Missing file: {file}")
            all_good = False
    
    return all_good

def check_attack_imports():
    """Check attack module imports"""
    print("\nChecking attack imports...")
    
    attacks = ['fgsm', 'pgd', 'deepfool']
    all_good = True
    
    for attack in attacks:
        try:
            module = __import__(f'attacks.{attack}', fromlist=[''])
            print(f"? Attack: {attack}")
        except Exception as e:
            print(f"? Attack {attack}: {e}")
            all_good = False
    
    return all_good

def check_defense_imports():
    """Check defense module imports"""
    print("\nChecking defense imports...")
    
    defenses = ['adv_training', 'input_smoothing', 'randomized_transform', 'model_wrappers']
    all_good = True
    
    for defense in defenses:
        try:
            module = __import__(f'defenses.{defense}', fromlist=[''])
            print(f"? Defense: {defense}")
        except Exception as e:
            print(f"? Defense {defense}: {e}")
            all_good = False
    
    return all_good

def quick_model_test():
    """Quick test of model instantiation - FIXED"""
    print("\nQuick model test...")
    
    try:
        # Add project root to path
        sys.path.insert(0, str(Path.cwd()))
        
        from models.base.mnist_cnn import MNIST_CNN
        
        model = MNIST_CNN()
        model.eval()  # Set to eval mode for batch norm
        
        # Test forward pass with batch size 2 (avoids batch norm issue)
        import torch
        test_input = torch.randn(2, 1, 28, 28)
        output = model(test_input)
        
        print(f"? Model instantiated successfully")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"? Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks"""
    print("="*60)
    print("ADVERSARIAL ML SUITE - VERIFICATION SCRIPT (FIXED)")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_structure),
        ("Package Imports", check_imports),
        ("Model Test", quick_model_test),
        ("Attack Imports", check_attack_imports),
        ("Defense Imports", check_defense_imports)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        result = check_func()
        results.append((check_name, result))
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{check_name:20} {status}")
        all_passed = all_passed and result
    
    print("\n" + "="*60)
    
    if all_passed:
        print("? All checks passed! The suite is ready to use.")
        print("\nNext steps:")
        print("1. Train model: python pipelines/train_model.py")
        print("2. Generate attacks: python pipelines/generate_adversarial.py")
        print("3. Evaluate robustness: python pipelines/robustness_eval.py")
        print("4. Train defenses: python pipelines/defense_train.py")
        print("5. Export report: python pipelines/export_report.py")
    else:
        print("? Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
