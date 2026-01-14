"""
SIMPLE AUTONOMOUS PLATFORM VERIFICATION
Just checks if the platform can start.
"""

import subprocess
import sys
import time

def verify_platform():
    """Verify platform can start"""
    print("\n" + "="*80)
    print("VERIFYING AUTONOMOUS PLATFORM")
    print("="*80)
    
    print("\n1. Checking required files...")
    required_files = [
        "autonomous_core.py",
        "autonomous_platform_ascii.py",
        "test_autonomous_ascii.py"
    ]
    
    missing = []
    for file in required_files:
        try:
            with open(file, 'r'):
                print(f"   [OK] {file} exists")
        except FileNotFoundError:
            print(f"   [ERROR] {file} missing")
            missing.append(file)
    
    if missing:
        print(f"\n[ERROR] Missing files: {', '.join(missing)}")
        return False
    
    print("\n2. Testing autonomous core module...")
    try:
        # Simple import test
        import autonomous_core
        from autonomous_core import create_autonomous_controller
        
        controller = create_autonomous_controller()
        controller.initialize()
        
        print(f"   [OK] Autonomous core loaded")
        print(f"   [OK] Controller: {controller.__class__.__name__}")
        print(f"   [OK] Status: {controller.get_status().get('status', 'unknown')}")
        
    except Exception as e:
        print(f"   [ERROR] Autonomous core error: {e}")
        return False
    
    print("\n3. Testing platform startup...")
    print("   Starting platform for 10 seconds to verify it runs...")
    
    try:
        # Start platform in subprocess
        platform = subprocess.Popen(
            [sys.executable, "autonomous_platform_ascii.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if still running
        if platform.poll() is None:
            print("   [OK] Platform started successfully")
            print("   [OK] Platform process is running")
            
            # Stop it
            platform.terminate()
            platform.wait(timeout=3)
            print("   [OK] Platform stopped cleanly")
            
            return True
        else:
            # Platform died
            stdout, stderr = platform.communicate()
            print(f"   [ERROR] Platform failed to start")
            print(f"   STDOUT: {stdout[-200:] if stdout else 'none'}")
            print(f"   STDERR: {stderr[-200:] if stderr else 'none'}")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Platform startup test failed: {e}")
        return False

def main():
    """Main verification"""
    print("\nAutonomous Platform Verifier")
    print("Simple verification for Windows")
    
    success = verify_platform()
    
    if success:
        print("\n" + "="*80)
        print("[TARGET] VERIFICATION SUCCESSFUL")
        print("="*80)
        
        print("\nYour autonomous platform is ready!")
        print("\nTo start the platform:")
        print("   1. Double-click: start_autonomous.bat")
        print("   2. Or: python autonomous_platform_ascii.py")
        
        print("\nTo test the platform (in another terminal):")
        print("   python test_autonomous_ascii.py")
        
        print("\nTest with curl commands:")
        print("   curl http://localhost:8000/autonomous/status")
        print("   curl -X POST http://localhost:8000/predict -H \"Content-Type: application/json\" -d \"{\"data\": {\"input\": [0.1] * 784}}\"")
        
        print("\nPlatform features:")
        print("   * 10-year survivability design")
        print("   * Self-healing security")
        print("   * Zero human babysitting")
        print("   * Autonomous threat adaptation")
        
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("[WARNING] VERIFICATION FAILED")
        print("="*80)
        
        print("\nCheck the errors above.")
        print("Common issues:")
        print("   1. Missing Python dependencies")
        print("   2. Port 8000 already in use")
        print("   3. File encoding issues")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
