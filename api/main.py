"""
🛡️ ENTERPRISE ADVERSARIAL ML SECURITY API - UNIFIED LAYER
Core Rule: Inference is a privilege, not a right.
"""
import os
import sys
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Security

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any, Optional

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reports/logs/enterprise_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enterprise_api")

# Security
security = HTTPBearer()

# Global components (initialized in lifespan)
firewall = None
model_router = None
attack_intel = None
audit_logger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enterprise lifespan management"""
    global firewall, model_router, attack_intel, audit_logger
    
    logger.info("🚀 ENTERPRISE SECURITY API STARTUP")
    
    try:
        # 1. Initialize Model Firewall
        from firewall.detector import ModelFirewall
        from firewall.policies.adaptive import AdaptiveFirewallPolicy
        firewall_policy = AdaptiveFirewallPolicy()
        firewall = ModelFirewall(policy=firewall_policy)
        logger.info("✅ Model Firewall initialized")
        
        # 2. Initialize Model Router
        from models.registry.model_router import EnterpriseModelRouter
        model_router = EnterpriseModelRouter()
        logger.info("✅ Model Router initialized")
        
        # 3. Initialize Adversarial Intelligence
        from intelligence.telemetry.attack_monitor import AttackTelemetry
        attack_intel = AttackTelemetry()
        logger.info("✅ Adversarial Intelligence initialized")
        
        # 4. Initialize Audit Logger
        from governance.compliance.audit_logger import EnterpriseAuditLogger
        audit_logger = EnterpriseAuditLogger()
        logger.info("✅ Audit Logger initialized")
        
        logger.info("🎯 ENTERPRISE SECURITY API READY")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 ENTERPRISE SECURITY API SHUTDOWN")
    firewall = None
    model_router = None
    attack_intel = None
    audit_logger = None

# Create FastAPI app
app = FastAPI(
    title="Enterprise Adversarial ML Security Platform",
    description="Unified security control plane for machine learning models",
    version="4.0.0-enterprise",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware (placeholder - implement properly)
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # TODO: Implement proper rate limiting
    response = await call_next(request)
    return response

# ==================== AUTHENTICATION & RBAC ====================
async def authenticate(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Enterprise authentication with RBAC"""
    # TODO: Implement JWT validation and RBAC
    # For now, return a mock user
    return {
        "user_id": "enterprise_user",
        "roles": ["ml_engineer", "security_analyst"],
        "permissions": ["predict", "attack_test", "view_reports"]
    }

def check_permission(user: Dict[str, Any], required_permission: str):
    """Check if user has required permission"""
    if required_permission not in user.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Required: {required_permission}"
        )

# ==================== ENTERPRISE ENDPOINTS ====================
@app.get("/api/v1/health")
async def enterprise_health(user: Dict[str, Any] = Depends(authenticate)):
    """Enterprise health check with component status"""
    check_permission(user, "view_reports")
    
    components = {
        "firewall": firewall is not None,
        "model_router": model_router is not None,
        "attack_intelligence": attack_intel is not None,
        "audit_logger": audit_logger is not None
    }
    
    return {
        "status": "healthy" if all(components.values()) else "degraded",
        "service": "enterprise-adversarial-ml-security",
        "version": "4.0.0-enterprise",
        "components": components,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/predict")
async def enterprise_predict(
    request: Dict[str, Any],
    user: Dict[str, Any] = Depends(authenticate)
):
    """
    Enterprise prediction endpoint
    
    Flow:
    1. Authentication & RBAC check
    2. Firewall evaluation
    3. Model routing
    4. Inference with telemetry
    5. Audit logging
    """
    check_permission(user, "predict")
    
    # Start audit trail
    request_id = audit_logger.start_request(request, user)
    
    try:
        # 1. Firewall evaluation
        firewall_result = firewall.evaluate(request)
        if not firewall_result["allowed"]:
            audit_logger.log_blocked(request_id, firewall_result)
            raise HTTPException(
                status_code=403,
                detail={
                    "status": "blocked",
                    "reason": firewall_result["reason"],
                    "request_id": request_id
                }
            )
        
        # 2. Model routing
        model_info = model_router.route(request)
        
        # 3. Load and run model
        prediction = model_info["model"].predict(request["data"])
        
        # 4. Telemetry
        attack_intel.record_inference(request_id, request, prediction)
        
        # 5. Complete audit
        audit_logger.log_success(request_id, prediction)
        
        return {
            "status": "success",
            "request_id": request_id,
            "prediction": prediction,
            "model": model_info["name"],
            "model_version": model_info["version"],
            "firewall_check": "passed",
            "audit_trail": f"/api/v1/audit/{request_id}"
        }
        
    except Exception as e:
        audit_logger.log_failure(request_id, str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "request_id": request_id,
                "error": str(e)
            }
        )

@app.post("/api/v1/attack/test")
async def enterprise_attack_test(
    request: Dict[str, Any],
    user: Dict[str, Any] = Depends(authenticate)
):
    """Enterprise adversarial testing endpoint"""
    check_permission(user, "attack_test")
    
    # TODO: Implement proper attack testing with telemetry
    return {
        "status": "attack_test_placeholder",
        "message": "Enterprise attack testing endpoint - implement proper attack orchestration"
    }

@app.get("/api/v1/audit/{request_id}")
async def get_audit_trail(
    request_id: str,
    user: Dict[str, Any] = Depends(authenticate)
):
    """Retrieve audit trail for a request"""
    check_permission(user, "view_reports")
    
    audit_data = audit_logger.get_audit_trail(request_id)
    if not audit_data:
        raise HTTPException(status_code=404, detail="Audit trail not found")
    
    return audit_data

@app.get("/api/v1/threat/intelligence")
async def threat_intelligence_report(
    user: Dict[str, Any] = Depends(authenticate)
):
    """Get adversarial threat intelligence report"""
    check_permission(user, "view_reports")
    
    report = attack_intel.generate_threat_report()
    return report

# ==================== MODEL MANAGEMENT ====================
@app.post("/api/v1/models/register")
async def register_model(
    model_info: Dict[str, Any],
    user: Dict[str, Any] = Depends(authenticate)
):
    """Register a new model in the enterprise registry"""
    check_permission(user, "model_management")
    
    model_id = model_router.register_model(model_info)
    return {
        "status": "registered",
        "model_id": model_id,
        "message": f"Model {model_info['name']} registered successfully"
    }

@app.get("/api/v1/models")
async def list_models(user: Dict[str, Any] = Depends(authenticate)):
    """List all registered models"""
    check_permission(user, "view_reports")
    
    models = model_router.list_models()
    return {
        "count": len(models),
        "models": models
    }

# ==================== STARTUP ====================
if __name__ == "__main__":
    import uvicorn
    from datetime import datetime
    
    print("\n" + "="*80)
    print("🛡️ ENTERPRISE ADVERSARIAL ML SECURITY PLATFORM")
    print("Unified Architecture, Governance, and Intelligence System")
    print("="*80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

