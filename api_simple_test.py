"""
🏢 ENTERPRISE PLATFORM - SIMPLIFIED TEST API
Starts just the essentials to verify everything works.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
import uvicorn
import torch
import numpy as np
from datetime import datetime

print("\n" + "="*80)
print("🏢 ENTERPRISE ADVERSARIAL ML SECURITY PLATFORM")
print("Simplified Test API")
print("="*80)

# Create the FastAPI app
app = FastAPI(
    title="Enterprise Adversarial ML Security Platform",
    description="Simplified test version",
    version="4.0.0-test"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "enterprise-adversarial-ml-security",
        "version": "4.0.0-test",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "api": True,
            "pytorch": torch.__version__,
            "numpy": np.__version__
        }
    }

@app.get("/test/firewall")
async def test_firewall():
    """Test firewall import"""
    try:
        from firewall.detector import ModelFirewall
        firewall = ModelFirewall()
        return {
            "status": "success",
            "component": "firewall",
            "message": "ModelFirewall loaded successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "component": "firewall",
            "error": str(e)
        }

@app.get("/test/intelligence")
async def test_intelligence():
    """Test intelligence import"""
    try:
        from intelligence.telemetry.attack_monitor import AttackTelemetry
        telemetry = AttackTelemetry()
        return {
            "status": "success",
            "component": "intelligence",
            "message": "AttackTelemetry loaded successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "component": "intelligence",
            "error": str(e)
        }

if __name__ == "__main__":
    print("🚀 Starting simplified enterprise API...")
    print("📡 Available at: http://localhost:8001")
    print("📚 Documentation: http://localhost:8001/docs")
    print("🛑 Press CTRL+C to stop\n")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,  # Use port 8001 to avoid conflicts
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        sys.exit(1)
