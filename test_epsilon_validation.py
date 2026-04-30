"""
Test script to verify epsilon validation bypass is fixed
Tests for: inf, nan, -1.0, 100.0+ values
"""
import torch
import torch.nn as nn
import math

# Import attack classes
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from attacks.deepfool import DeepFoolAttack
from attacks.cw import CarliniWagnerL2

# Simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def test_fgsm_validation():
    """Test FGSM epsilon validation"""
    print("\n=== Testing FGSM Epsilon Validation ===")
    model = SimpleModel()
    
    # Test cases that should FAIL (raise ValueError)
    invalid_values = [
        (float('inf'), "infinity"),
        (float('-inf'), "negative infinity"),
        (float('nan'), "NaN"),
        (-1.0, "negative value"),
        (100.0, "excessive value (>1.0)"),
        (1000.0, "very large value"),
    ]
    
    for value, description in invalid_values:
        try:
            attack = FGSMAttack(model, {'epsilon': value})
            print(f"  ❌ FAIL: {description} ({value}) was accepted!")
        except ValueError as e:
            print(f"  ✅ PASS: {description} ({value}) correctly rejected: {str(e)[:60]}...")
        except Exception as e:
            print(f"  ⚠️  UNEXPECTED: {description} raised {type(e).__name__}: {e}")
    
    # Test valid values that should PASS
    valid_values = [0.01, 0.15, 0.5, 1.0]
    for value in valid_values:
        try:
            attack = FGSMAttack(model, {'epsilon': value})
            print(f"  ✅ PASS: Valid epsilon {value} accepted")
        except Exception as e:
            print(f"  ❌ FAIL: Valid epsilon {value} was rejected: {e}")

def test_pgd_validation():
    """Test PGD epsilon validation"""
    print("\n=== Testing PGD Epsilon Validation ===")
    model = SimpleModel()
    
    # Test cases that should FAIL
    invalid_values = [
        (float('inf'), "infinity"),
        (float('-inf'), "negative infinity"),
        (float('nan'), "NaN"),
        (-1.0, "negative value"),
        (100.0, "excessive value (>1.0)"),
    ]
    
    for value, description in invalid_values:
        try:
            attack = PGDAttack(model, {'epsilon': value})
            print(f"  ❌ FAIL: {description} ({value}) was accepted!")
        except ValueError as e:
            print(f"  ✅ PASS: {description} ({value}) correctly rejected: {str(e)[:60]}...")
        except Exception as e:
            print(f"  ⚠️  UNEXPECTED: {description} raised {type(e).__name__}: {e}")
    
    # Test valid values
    valid_values = [0.01, 0.15, 0.3, 0.5, 1.0]
    for value in valid_values:
        try:
            attack = PGDAttack(model, {'epsilon': value})
            print(f"  ✅ PASS: Valid epsilon {value} accepted")
        except Exception as e:
            print(f"  ❌ FAIL: Valid epsilon {value} was rejected: {e}")

def test_deepfool_validation():
    """Test DeepFool overshoot validation"""
    print("\n=== Testing DeepFool Overshoot Validation ===")
    model = SimpleModel()
    
    # Test cases that should FAIL
    invalid_values = [
        (float('inf'), "infinity"),
        (float('-inf'), "negative infinity"),
        (float('nan'), "NaN"),
        (-1.0, "negative value"),
        (100.0, "excessive value (>1.0)"),
    ]
    
    for value, description in invalid_values:
        try:
            attack = DeepFoolAttack(model, {'overshoot': value})
            print(f"  ❌ FAIL: {description} ({value}) was accepted!")
        except ValueError as e:
            print(f"  ✅ PASS: {description} ({value}) correctly rejected: {str(e)[:60]}...")
        except Exception as e:
            print(f"  ⚠️  UNEXPECTED: {description} raised {type(e).__name__}: {e}")
    
    # Test valid values
    valid_values = [0.0, 0.02, 0.1, 0.5, 1.0]
    for value in valid_values:
        try:
            attack = DeepFoolAttack(model, {'overshoot': value})
            print(f"  ✅ PASS: Valid overshoot {value} accepted")
        except Exception as e:
            print(f"  ❌ FAIL: Valid overshoot {value} was rejected: {e}")

def test_cw_validation():
    """Test C&W parameter validation"""
    print("\n=== Testing C&W Parameter Validation ===")
    model = SimpleModel()
    
    # Test confidence parameter
    print("  Testing confidence parameter:")
    invalid_confidence = [float('inf'), float('nan'), -1.0]
    for value in invalid_confidence:
        try:
            attack = CarliniWagnerL2(model, {'confidence': value})
            print(f"    ❌ FAIL: confidence={value} was accepted!")
        except ValueError as e:
            print(f"    ✅ PASS: confidence={value} correctly rejected")
    
    # Test learning_rate parameter
    print("  Testing learning_rate parameter:")
    invalid_lr = [float('inf'), float('nan'), -0.01, 0]
    for value in invalid_lr:
        try:
            attack = CarliniWagnerL2(model, {'learning_rate': value})
            print(f"    ❌ FAIL: learning_rate={value} was accepted!")
        except ValueError as e:
            print(f"    ✅ PASS: learning_rate={value} correctly rejected")
    
    # Test const parameter
    print("  Testing const parameter:")
    invalid_const = [float('inf'), float('nan'), -1.0, 0]
    for value in invalid_const:
        try:
            attack = CarliniWagnerL2(model, {'initial_const': value})
            print(f"    ❌ FAIL: const={value} was accepted!")
        except ValueError as e:
            print(f"    ✅ PASS: const={value} correctly rejected")

if __name__ == "__main__":
    print("=" * 70)
    print("EPSILON VALIDATION BYPASS FIX VERIFICATION")
    print("Testing protection against: inf, nan, -1.0, 100.0+")
    print("=" * 70)
    
    test_fgsm_validation()
    test_pgd_validation()
    test_deepfool_validation()
    test_cw_validation()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
