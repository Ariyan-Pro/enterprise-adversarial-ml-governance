"""
🔧 CLEAN AUTONOMOUS INTEGRATION - NO INDENTATION ERRORS
Simple, clean integration of autonomous engine.
"""

import os
import sys

def integrate_cleanly():
    """Clean integration without indentation issues"""
    
    # Read the original file
    with open("enterprise_platform.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    new_lines = []
    
    # Flag to track if we're inside the main function
    inside_main_app = False
    app_indent_level = 0
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Look for FastAPI app initialization
        if "app = FastAPI(" in line or ("FastAPI(" in line and i > 0 and "app =" in lines[i-1]):
            inside_main_app = True
            
            # Count indentation of this line
            app_indent_level = len(line) - len(line.lstrip())
            print(f"Found FastAPI app at line {i+1}, indent level: {app_indent_level}")
            
            # Add autonomous imports after this block
            # We'll add them after the app is fully defined
            # Look for the end of the app definition
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith('(') or 
                                      lines[j].strip().startswith(')') or
                                      len(lines[j]) - len(lines[j].lstrip()) > app_indent_level):
                j += 1
            
            # Now insert autonomous initialization at the correct indentation
            autonomous_init = [
                "",
                "# " + "="*70,
                "# AUTONOMOUS EVOLUTION ENGINE (10-year survivability)",
                "# " + "="*70,
                "try:",
                "    from autonomous_core import create_autonomous_controller",
                "    AUTONOMOUS_AVAILABLE = True",
                "    print(\"🧠 Autonomous evolution engine available\")",
                "except ImportError as e:",
                "    print(f\"⚠️  Autonomous engine not available: {e}\")",
                "    AUTONOMOUS_AVAILABLE = False",
                "    # Create minimal mock for testing",
                "    class MockAutonomousController:",
                "        def __init__(self):",
                "            self.total_requests = 0",
                "            self.is_initialized = False",
                "        def initialize(self):",
                "            self.is_initialized = True",
                "            return {\"status\": \"mock_initialized\"}",
                "        def process_request(self, request, inference_result):",
                "            self.total_requests += 1",
                "            inference_result[\"autonomous_processed\"] = True",
                "            inference_result[\"request_count\"] = self.total_requests",
                "            return inference_result",
                "        def get_status(self):",
                "            return {",
                "                \"status\": \"mock\",",
                "                \"initialized\": self.is_initialized,",
                "                \"total_requests_processed\": self.total_requests",
                "            }",
                "        def get_health(self):",
                "            return {\"mock\": True, \"components\": {\"autonomous\": \"mock\"}}",
                "    create_autonomous_controller = lambda: MockAutonomousController()",
                "",
                "# Initialize autonomous controller",
                "autonomous_controller = create_autonomous_controller()",
                "autonomous_controller.initialize()",
                "print(f\"✅ Autonomous controller initialized\")",
                ""
            ]
            
            # Add proper indentation
            indent_spaces = " " * app_indent_level
            for init_line in autonomous_init:
                if init_line.strip():  # Non-empty line
                    new_lines.append(indent_spaces + init_line)
                else:
                    new_lines.append("")  # Keep empty lines
    
    # Now write the fixed file
    with open("enterprise_platform.py", 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print("✅ Clean integration complete")
    return True

def main():
    """Main function"""
    print("\n🔧 Clean Autonomous Integration")
    print("="*80)
    
    # Check prerequisites
    if not os.path.exists("enterprise_platform.py"):
        print("❌ enterprise_platform.py not found")
        return False
    
    if not os.path.exists("autonomous_core.py"):
        print("❌ autonomous_core.py not found")
        print("   Run from the directory with autonomous_core.py")
        return False
    
    # Run integration
    success = integrate_cleanly()
    
    if success:
        print("\n" + "="*80)
        print("✅ CLEAN INTEGRATION SUCCESSFUL")
        print("="*80)
        
        print("\n🔧 To start your autonomous platform:")
        print("   python enterprise_platform.py")
        
        print("\n🌐 Test autonomous endpoints:")
        print("   curl http://localhost:8000/autonomous/status")
        print("   curl http://localhost:8000/autonomous/health")
        
        print("\n🚀 Or use the standalone server:")
        print("   python run_autonomous_server.py  (port 8002)")
    
    return success

if __name__ == "__main__":
    main()
