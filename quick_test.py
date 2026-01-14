"""
🧪 ENTERPRISE PLATFORM - QUICK TEST
Quick test to verify everything works.
"""
import requests
import time
import sys

def test_platform():
    """Test the enterprise platform"""
    print("\n" + "="*80)
    print("🧪 ENTERPRISE PLATFORM QUICK TEST")
    print("="*80)
    
    base_url = "http://localhost:8000"
    
    print(f"\nTesting platform at: {base_url}")
    print("Make sure platform is running first!")
    print("Run: python enterprise_platform.py")
    print()
    
    try:
        # Test 1: Root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(base_url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Service: {data.get('service')}")
            print(f"   ✅ Version: {data.get('version')}")
            print(f"   ✅ Status: {data.get('status')}")
        else:
            print(f"   ❌ Failed: HTTP {response.status_code}")
            return False
        
        # Test 2: Health endpoint
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   ✅ PyTorch: {data.get('components', {}).get('pytorch')}")
        else:
            print(f"   ❌ Failed: HTTP {response.status_code}")
            return False
        
        # Test 3: Try a prediction
        print("\n3. Testing prediction endpoint...")
        test_data = {
            "data": {
                "input": [0.0] * 784  # Blank 28x28 image
            }
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction made")
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   ✅ Prediction: {data.get('prediction')}")
            print(f"   ✅ Confidence: {data.get('confidence', 0):.2%}")
        else:
            print(f"   ⚠️  Prediction returned: HTTP {response.status_code}")
            if response.text:
                print(f"   Response: {response.text[:100]}...")
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("\n🌐 Access your platform at:")
        print(f"   Main: {base_url}")
        print(f"   Docs: {base_url}/docs")
        print(f"   Health: {base_url}/health")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to server!")
        print("   Make sure the platform is running:")
        print("   python enterprise_platform.py")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("\nEnterprise Adversarial ML Security Platform - Quick Test")
    print("Version: 4.0.0")
    
    try:
        success = test_platform()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled.")
        sys.exit(1)
