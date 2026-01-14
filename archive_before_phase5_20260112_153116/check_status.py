"""
🔍 ENTERPRISE PLATFORM STATUS CHECKER
Quickly check if platform is running and healthy.
"""
import requests
import sys
import time

def check_platform(port=8000, timeout=3):
    """Check if platform is running on specified port"""
    base_url = f"http://localhost:{port}"
    
    try:
        # Try health endpoint first
        response = requests.get(f"{base_url}/health", timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "running",
                "port": port,
                "url": base_url,
                "health": data.get("status", "unknown"),
                "components": data.get("components", {}),
                "timestamp": data.get("timestamp", "unknown")
            }
        else:
            return {
                "status": "error",
                "port": port,
                "url": base_url,
                "error": f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "status": "not_running",
            "port": port,
            "url": base_url,
            "error": "Connection refused"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "port": port,
            "url": base_url,
            "error": "Request timeout"
        }
    except Exception as e:
        return {
            "status": "error",
            "port": port,
            "url": base_url,
            "error": str(e)[:100]
        }

def main():
    """Main status checking function"""
    print("\n" + "="*80)
    print("🔍 ENTERPRISE PLATFORM STATUS CHECKER")
    print("="*80)
    
    ports_to_check = [8000, 8001]  # Main platform and test platform
    
    print("\nChecking platform status...")
    print("(Make sure platform is running with: python enterprise_platform.py)")
    print()
    
    for port in ports_to_check:
        platform_name = "MAIN PLATFORM" if port == 8000 else "SIMPLIFIED TEST"
        print(f"\n{platform_name} (port {port}):")
        
        result = check_platform(port)
        
        if result["status"] == "running":
            print(f"  ✅ STATUS: RUNNING")
            print(f"     URL: {result['url']}")
            print(f"     Health: {result['health']}")
            
            # Show component status
            components = result.get("components", {})
            if components:
                print(f"     Components:")
                for comp, status in components.items():
                    status_icon = "✅" if status else "❌"
                    print(f"       {status_icon} {comp}: {status}")
        
        elif result["status"] == "not_running":
            print(f"  ❌ STATUS: NOT RUNNING")
            print(f"     Error: {result['error']}")
            print(f"     Start it with: python {'enterprise_platform.py' if port == 8000 else 'api_simple_test.py'}")
        
        else:
            print(f"  ⚠️  STATUS: {result['status'].upper()}")
            print(f"     Error: {result['error']}")
    
    print("\n" + "="*80)
    print("🚀 QUICK START COMMANDS:")
    print("="*80)
    print("\nTo start MAIN PLATFORM:")
    print("  python enterprise_platform.py")
    print("  or double-click: launch_enterprise.bat")
    
    print("\nTo start SIMPLIFIED TEST:")
    print("  python api_simple_test.py")
    
    print("\n🌐 ACCESS POINTS (when running):")
    print("  Main: http://localhost:8000")
    print("  Test: http://localhost:8001")
    print("  Docs: Add /docs to either URL")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nStatus check cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
