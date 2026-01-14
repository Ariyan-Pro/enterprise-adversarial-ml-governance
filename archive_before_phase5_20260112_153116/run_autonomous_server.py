#!/usr/bin/env python3
"""
🚀 STANDALONE AUTONOMOUS SERVER
Run autonomous platform on separate port (8002) for testing.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from autonomous_integration import run_autonomous_server
    print("Starting standalone autonomous server on port 8002...")
    run_autonomous_server(port=8002)
except ImportError as e:
    print(f"Cannot import autonomous modules: {e}")
    print("Make sure autonomous_core.py and autonomous_integration.py exist")
    sys.exit(1)
