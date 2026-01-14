#!/usr/bin/env python3
"""
🚀 MINIMAL WORKING API ENTERPRISE - UTF-8 SAFE
Enterprise Adversarial ML Governance Engine API
"""

import sys
import os

# Force UTF-8 encoding
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
from typing import Dict, Any
import json

print("\n" + "="*60)
print("🚀 MINIMAL ENTERPRISE ADVERSARIAL ML GOVERNANCE ENGINE")
print("="*60)

# Try to import Phase 5
PHASE5_AVAILABLE = False
phase5_engine = None

try:
    from autonomous.core.database_engine import DatabaseAwareEngine
    PHASE5_AVAILABLE = True
    print("✅ Phase 5 engine available")
except ImportError as e:
    print(f"⚠️  Phase 5 not available: {e}")

if PHASE5_AVAILABLE:
    try:
        phase5_engine = DatabaseAwareEngine()
        print(f"✅ Phase 5 engine initialized")
    except Exception as e:
        print(f"⚠️  Phase 5 engine failed: {e}")
        phase5_engine = None

app = FastAPI(
    title="Enterprise Adversarial ML Governance Engine API",
    description="Minimal working API with Phase 5 integration",
    version="5.0.0 LTS"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Enterprise Adversarial ML Governance Engine",
        "version": "5.0.0",
        "phase": "5.1" if phase5_engine else "4.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    health = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "healthy",
        "version": "5.0.0",
        "phase": "5.1" if phase5_engine else "4.0",
        "components": {
            "api": "operational",
            "adversarial_defense": "ready",
            "autonomous_engine": "ready"
        }
    }
    
    if phase5_engine:
        try:
            ecosystem_health = phase5_engine.get_ecosystem_health()
            health["ecosystem"] = ecosystem_health
            health["components"]["database_memory"] = "operational"
        except Exception as e:
            health["ecosystem"] = {"status": "error", "message": str(e)}
            health["components"]["database_memory"] = "degraded"
    
    return JSONResponse(content=health)

@app.get("/api/ecosystem")
async def ecosystem_status():
    """Get ecosystem status"""
    if not phase5_engine:
        raise HTTPException(status_code=503, detail="Phase 5 engine not available")
    
    try:
        health = phase5_engine.get_ecosystem_health()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ecosystem check failed: {str(e)}")

@app.post("/api/predict")
async def predict(data: Dict[str, Any]):
    """Mock prediction endpoint"""
    return {
        "prediction": "protected",
        "confidence": 0.95,
        "adversarial_check": "passed",
        "model": "mnist_cnn_fixed",
        "parameters": 207018,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    print(f"\n📊 System Status:")
    print(f"   Phase 5: {'✅ Available' if phase5_engine else '❌ Not available'}")
    if phase5_engine:
        print(f"   Database mode: {phase5_engine.database_mode}")
        print(f"   System state: {phase5_engine.system_state}")
    
    print("\n🌐 Starting API server...")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/api/health")
    print("   Stop: Ctrl+C")
    print("\n" + "="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

