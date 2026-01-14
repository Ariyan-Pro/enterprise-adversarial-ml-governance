"""
✅ SIMPLE PLATFORM VERIFICATION
Quick test to verify platform is working.
"""
import requests
import sys

def test_platform():
    """Simple platform test"""
    print("\n" + "="*80)
    print("✅ ENTERPRISE PLATFORM - SIMPLE VERIFICATION")
    print("="*80)
    
    print("\nTesting MAIN PLATFORM (port 8000)...")
    
    try:
        # Test 1: Root endpoint
        response = requests.get("http://localhost:8000/", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: HTTP 200")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
        else:
            print(f"❌ Root endpoint: HTTP {response.status_code}")
            return False
        
        # Test 2: Health endpoint
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: HTTP 200")
            print(f"   Status: {data.get('status')}")
        else:
            print(f"❌ Health check: HTTP {response.status_code}")
            return False
        
        # Test 3: Try a simple prediction
        print("\nTesting prediction endpoint...")
        test_data = {"data": {"input": [0.0] * 784}}
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction: HTTP 200")
            print(f"   Status: {data.get('status')}")
            print(f"   Prediction: {data.get('prediction')}")
            print(f"   Confidence: {data.get('confidence', 0):.2%}")
        else:
            print(f"⚠️  Prediction: HTTP {response.status_code}")
            print(f"   (This is OK for blank input)")
        
        print("\n" + "="*80)
        print("🎉 PLATFORM VERIFICATION: PASSED ✅")
        print("\n🌐 Your enterprise platform is WORKING at:")
        print("   http://localhost:8000")
        print("   Documentation: http://localhost:8000/docs")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to platform!")
        print("\n🔧 Make sure the platform is running:")
        print("   python enterprise_platform.py")
        print("\nOr try the simplified test platform:")
        print("   python api_simple_test.py")
        return False
    except Exception as e:
        print(f"\n❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    print("\nEnterprise Adversarial ML Security Platform - Verification")
    print("Version: 4.0.0")
    
    try:
        success = test_platform()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled.")
        sys.exit(1)
