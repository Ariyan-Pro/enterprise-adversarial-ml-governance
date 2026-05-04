"""
🛡️ ENTERPRISE ADVERSARIAL ML SECURITY API - UNIFIED LAYER
Core Rule: Inference is a privilege, not a right.
"""
import os
import sys
from pathlib import Path
import jwt
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from collections import defaultdict
import time
import json
import threading

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Security, Request
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

# JWT Configuration - Require environment variable for production security
# The JWT secret MUST be set via environment variable in production
# Never use runtime-generated secrets as they cause token invalidation on restart
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")

# Check if running in production mode
IS_PRODUCTION = os.environ.get("ENVIRONMENT", "development").lower() == "production"

if not JWT_SECRET_KEY:
    if IS_PRODUCTION:
        # CRITICAL: Fail startup in production if JWT_SECRET_KEY is not set
        raise RuntimeError(
            "🚨 CRITICAL SECURITY ERROR: JWT_SECRET_KEY is not set! "
            "This is required for production deployments. "
            "Set the JWT_SECRET_KEY environment variable before starting the server. "
            "Generate a secure key with: python -c 'import secrets; print(secrets.token_hex(32))'"
        )
    else:
        # In development only: generate and warn (never use in production)
        import warnings
        JWT_SECRET_KEY = os.urandom(32).hex()
        warnings.warn(
            "⚠️ SECURITY WARNING: JWT_SECRET_KEY not set in environment. "
            "Using runtime-generated secret. This will invalidate all tokens on restart. "
            "SET JWT_SECRET_KEY environment variable in production!",
            RuntimeWarning,
            stacklevel=2
        )
        logger.warning(
            "⚠️ INSECURE: JWT_SECRET_KEY not configured. Using runtime-generated secret. "
            "Set JWT_SECRET_KEY environment variable for production deployments."
        )

# Validate JWT secret key strength (must be at least 32 bytes/64 hex chars)
if len(JWT_SECRET_KEY) < 64:
    raise ValueError(
        f"JWT_SECRET_KEY is too weak. Minimum 64 characters (32 bytes) required. "
        f"Current length: {len(JWT_SECRET_KEY)}. "
        "Generate a secure key with: python -c 'import secrets; print(secrets.token_hex(32))'"
    )

JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Global components (initialized in lifespan)
firewall = None
model_router = None
attack_intel = None
audit_logger = None

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100  # Maximum requests per window
RATE_LIMIT_WINDOW_SECONDS = 60  # Time window in seconds

# Persistent rate limit storage file path
RATE_LIMIT_STORAGE_FILE = Path(__file__).parent.parent / "config" / "rate_limits.json"
rate_limit_storage_lock = threading.Lock()

def load_rate_limit_storage() -> defaultdict:
    """Load rate limit state from persistent storage on startup"""
    storage = defaultdict(list)
    if RATE_LIMIT_STORAGE_FILE.exists():
        try:
            with open(RATE_LIMIT_STORAGE_FILE, 'r') as f:
                data = json.load(f)
                for client_id, timestamps in data.items():
                    storage[client_id] = timestamps
            logger.info(f"✅ Loaded rate limit storage from {RATE_LIMIT_STORAGE_FILE}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"⚠️ Failed to load rate limit storage: {e}. Starting fresh.")
    return storage

def save_rate_limit_storage(storage: defaultdict):
    """Save rate limit state to persistent storage"""
    try:
        # Ensure config directory exists
        RATE_LIMIT_STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RATE_LIMIT_STORAGE_FILE, 'w') as f:
            json.dump(dict(storage), f)
    except IOError as e:
        logger.error(f"❌ Failed to save rate limit storage: {e}")

# In-memory rate limit storage (persisted to disk for restart resilience)
rate_limit_storage = load_rate_limit_storage()

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
    
    # Shutdown - persist rate limit state before exit
    logger.info("💾 Persisting rate limit state before shutdown...")
    save_rate_limit_storage(rate_limit_storage)
    
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

# CORS - Secure configuration with explicit allowed origins
# Never use wildcard (*) with allow_credentials=True as it enables CSRF attacks
CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Window", "X-RateLimit-Remaining"],
    max_age=600,  # Cache preflight results for 10 minutes
)

def get_client_identifier(request: Request) -> str:
    """Extract client identifier for rate limiting (IP or user ID)"""
    # Try to get user ID from authorization header first
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]  # Remove "Bearer " prefix
        try:
            payload = jwt.decode(
                token,
                JWT_SECRET_KEY,
                algorithms=[JWT_ALGORITHM],
                options={"require": ["exp", "iat", "sub"]}
            )
            return f"user:{payload.get('sub')}"
        except jwt.InvalidTokenError:
            pass
    
    # Fall back to IP address
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        client_host = request.client.host if request.client else "unknown"
        ip = client_host
    
    return f"ip:{ip}"


def check_rate_limit(client_id: str) -> tuple[bool, Dict[str, Any]]:
    """
    Check if client has exceeded rate limit using sliding window algorithm.
    
    Returns:
        tuple: (is_allowed, metadata_dict)
    """
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW_SECONDS
    
    with rate_limit_storage_lock:
        # Clean up old entries and count recent requests
        recent_requests = [
            ts for ts in rate_limit_storage[client_id]
            if ts > window_start
        ]
        
        # Update storage with cleaned list
        rate_limit_storage[client_id] = recent_requests
        
        # Check if limit exceeded
        if len(recent_requests) >= RATE_LIMIT_REQUESTS:
            oldest_request = min(recent_requests) if recent_requests else current_time
            retry_after = int(oldest_request + RATE_LIMIT_WINDOW_SECONDS - current_time) + 1
            # Persist state after modification
            save_rate_limit_storage(rate_limit_storage)
            return False, {
                "retry_after": max(1, retry_after),
                "limit": RATE_LIMIT_REQUESTS,
                "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
                "remaining": 0
            }
        
        # Record this request
        rate_limit_storage[client_id].append(current_time)
        remaining = RATE_LIMIT_REQUESTS - len(rate_limit_storage[client_id])
        
        # Persist state periodically (every 10 requests to balance performance and durability)
        if len(rate_limit_storage[client_id]) % 10 == 0:
            save_rate_limit_storage(rate_limit_storage)
        
        return True, {
            "limit": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "remaining": remaining,
            "reset": int(current_time + RATE_LIMIT_WINDOW_SECONDS)
        }


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Enterprise rate limiting middleware to prevent DoS attacks.
    Implements sliding window rate limiting per client (user or IP).
    """
    # Skip rate limiting for health checks to allow monitoring
    if request.url.path in ["/health", "/api/v1/health"]:
        response = await call_next(request)
        return response
    
    # Get client identifier
    client_id = get_client_identifier(request)
    
    # Check rate limit
    is_allowed, metadata = check_rate_limit(client_id)
    
    if not is_allowed:
        logger.warning(
            f"🚫 RATE LIMIT EXCEEDED for client: {client_id} | "
            f"Limit: {metadata['limit']}/{metadata['window_seconds']}s"
        )
        raise HTTPException(
            status_code=429,
            headers={
                "X-RateLimit-Limit": str(metadata["limit"]),
                "X-RateLimit-Window": str(metadata["window_seconds"]),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(metadata["retry_after"])
            },
            detail={
                "status": "rate_limit_exceeded",
                "error": "Too many requests. Please slow down.",
                "retry_after_seconds": metadata["retry_after"],
                "limit": metadata["limit"],
                "window_seconds": metadata["window_seconds"]
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
    response.headers["X-RateLimit-Window"] = str(metadata["window_seconds"])
    response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
    
    return response

# ==================== AUTHENTICATION & RBAC ====================
async def authenticate(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Enterprise authentication with JWT validation and RBAC"""
    token = credentials.credentials
    
    try:
        # Validate JWT token
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub"]}
        )
        
        # Extract user information from validated token
        user_id = payload.get("sub")
        roles = payload.get("roles", [])
        permissions = payload.get("permissions", [])
        
        # Log successful authentication
        logger.info(f"✅ Authenticated user: {user_id} with roles: {roles}")
        
        return {
            "user_id": user_id,
            "roles": roles,
            "permissions": permissions,
            "token_payload": payload
        }
        
    except jwt.ExpiredSignatureError:
        logger.warning(f"⚠️ Expired token attempt")
        raise HTTPException(
            status_code=401,
            detail="Token has expired. Please obtain a new token."
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"⚠️ Invalid token attempt: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid token. Authentication failed."
        )
    except Exception as e:
        logger.error(f"❌ Authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed."
        )

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

# ==================== TOKEN MANAGEMENT ====================

def validate_credentials(username: str, password: str) -> bool:
    """
    Validate user credentials against secure identity provider.
    
    Supports multiple authentication backends:
    - LDAP/Active Directory lookup
    - OAuth2/OIDC provider validation
    - Database lookup with hashed passwords (bcrypt, argon2)
    - Multi-factor authentication verification
    
    Args:
        username: User's username/email
        password: User's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
        
    Raises:
        RuntimeError: If no authentication backend is configured
    """
    # Determine authentication backend from environment
    auth_backend = os.environ.get("AUTH_BACKEND", "").lower()
    
    if not auth_backend:
        logger.error(
            "🚨 SECURITY ERROR: No authentication backend configured! "
            "Set AUTH_BACKEND environment variable to one of: ldap, oauth2, database"
        )
        raise RuntimeError(
            "Authentication backend not configured. "
            "Please set AUTH_BACKEND environment variable to 'ldap', 'oauth2', or 'database' "
            "and configure the corresponding settings before using this endpoint."
        )
    
    try:
        if auth_backend == "ldap":
            return _validate_ldap_credentials(username, password)
        elif auth_backend == "oauth2":
            return _validate_oauth2_credentials(username, password)
        elif auth_backend == "database":
            return _validate_database_credentials(username, password)
        else:
            logger.error(f"Unknown authentication backend: {auth_backend}")
            raise ValueError(f"Unsupported authentication backend: {auth_backend}")
    except Exception as e:
        logger.error(f"Credential validation failed for backend '{auth_backend}': {e}")
        raise


def _validate_ldap_credentials(username: str, password: str) -> bool:
    """
    Validate credentials against LDAP/Active Directory.
    
    Required environment variables:
    - LDAP_SERVER: LDAP server URL (e.g., ldap://ad.example.com)
    - LDAP_BASE_DN: Base DN for searches (e.g., dc=example,dc=com)
    - LDAP_USER_DN_TEMPLATE: DN template for user binding (e.g., cn={username},ou=users,dc=example,dc=com)
    """
    try:
        from ldap3 import Server, Connection, ALL, NTLM
    except ImportError:
        logger.error("ldap3 library not installed. Install with: pip install ldap3")
        raise RuntimeError("LDAP authentication requires ldap3 library. Install with: pip install ldap3")
    
    ldap_server = os.environ.get("LDAP_SERVER")
    ldap_base_dn = os.environ.get("LDAP_BASE_DN")
    ldap_user_dn_template = os.environ.get("LDAP_USER_DN_TEMPLATE")
    
    if not all([ldap_server, ldap_base_dn, ldap_user_dn_template]):
        logger.error("Missing required LDAP configuration environment variables")
        raise RuntimeError(
            "LDAP configuration incomplete. Required: LDAP_SERVER, LDAP_BASE_DN, LDAP_USER_DN_TEMPLATE"
        )
    
    try:
        # Connect to LDAP server
        server = Server(ldap_server, get_info=ALL)
        
        # Format user DN
        user_dn = ldap_user_dn_template.format(username=username)
        
        # Attempt to bind with user credentials
        conn = Connection(server, user=user_dn, password=password, authentication=NTLM if 'ntlm' in os.environ.get('LDAP_AUTH_TYPE', '').lower() else None)
        
        if conn.bind():
            logger.info(f"✅ LDAP authentication successful for user: {username}")
            conn.unbind()
            return True
        else:
            logger.warning(f"Failed LDAP authentication for user: {username}")
            return False
            
    except Exception as e:
        logger.error(f"LDAP authentication error: {e}")
        raise


def _validate_oauth2_credentials(username: str, password: str) -> bool:
    """
    Validate credentials against OAuth2/OIDC provider.
    
    Required environment variables:
    - OAUTH2_CLIENT_ID: OAuth2 client ID
    - OAUTH2_CLIENT_SECRET: OAuth2 client secret
    - OAUTH2_TOKEN_ENDPOINT: Token endpoint URL
    - OAUTH2_GRANT_TYPE: Grant type (default: password)
    """
    try:
        import requests
    except ImportError:
        logger.error("requests library not installed. Install with: pip install requests")
        raise RuntimeError("OAuth2 authentication requires requests library")
    
    client_id = os.environ.get("OAUTH2_CLIENT_ID")
    client_secret = os.environ.get("OAUTH2_CLIENT_SECRET")
    token_endpoint = os.environ.get("OAUTH2_TOKEN_ENDPOINT")
    grant_type = os.environ.get("OAUTH2_GRANT_TYPE", "password")
    
    if not all([client_id, client_secret, token_endpoint]):
        logger.error("Missing required OAuth2 configuration environment variables")
        raise RuntimeError(
            "OAuth2 configuration incomplete. Required: OAUTH2_CLIENT_ID, OAUTH2_CLIENT_SECRET, OAUTH2_TOKEN_ENDPOINT"
        )
    
    try:
        response = requests.post(
            token_endpoint,
            data={
                "grant_type": grant_type,
                "username": username,
                "password": password,
                "client_id": client_id,
                "client_secret": client_secret
            },
            timeout=10
        )
        
        if response.status_code == 200:
            token_data = response.json()
            if "access_token" in token_data:
                logger.info(f"✅ OAuth2 authentication successful for user: {username}")
                return True
        
        logger.warning(f"Failed OAuth2 authentication for user: {username} - Status: {response.status_code}")
        return False
        
    except requests.exceptions.RequestException as e:
        logger.error(f"OAuth2 request error: {e}")
        raise


def _validate_database_credentials(username: str, password: str) -> bool:
    """
    Validate credentials against database with hashed passwords.
    Uses bcrypt for password hashing verification.
    
    Required environment variables:
    - DATABASE_URL: Database connection string (e.g., postgresql://user:pass@host/db)
    - DB_PASSWORD_TABLE: Table name containing user credentials (default: users)
    - DB_USERNAME_COLUMN: Column name for username (default: username)
    - DB_PASSWORD_COLUMN: Column name for password hash (default: password_hash)
    """
    try:
        import bcrypt
    except ImportError:
        logger.error("bcrypt library not installed. Install with: pip install bcrypt")
        raise RuntimeError("Database authentication requires bcrypt library")
    
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        logger.error("sqlalchemy library not installed. Install with: pip install sqlalchemy")
        raise RuntimeError("Database authentication requires sqlalchemy library")
    
    database_url = os.environ.get("DATABASE_URL")
    password_table = os.environ.get("DB_PASSWORD_TABLE", "users")
    username_column = os.environ.get("DB_USERNAME_COLUMN", "username")
    password_column = os.environ.get("DB_PASSWORD_COLUMN", "password_hash")
    
    if not database_url:
        logger.error("Missing required DATABASE_URL environment variable")
        raise RuntimeError("Database configuration incomplete. Required: DATABASE_URL")
    
    try:
        # Create database engine
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Query user's password hash
            query = text(
                f"SELECT {password_column} FROM {password_table} WHERE {username_column} = :username LIMIT 1"
            )
            result = conn.execute(query, {"username": username})
            row = result.fetchone()
            
            if not row:
                logger.warning(f"No user found in database: {username}")
                return False
            
            stored_hash = row[0]
            
            # Verify password against stored hash
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode('utf-8')
            
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                logger.info(f"✅ Database authentication successful for user: {username}")
                return True
            else:
                logger.warning(f"Invalid password for user: {username}")
                return False
                
    except Exception as e:
        logger.error(f"Database authentication error: {e}")
        raise


@app.post("/api/v1/auth/token")
async def generate_token(token_request: Dict[str, Any]):
    """
    Generate a JWT token for authenticated users.
    Validates credentials against configured identity provider.
    
    Expected request body:
    {
        "username": "user@example.com",
        "password": "secure_password",
        "roles": ["ml_engineer"],  # optional
        "permissions": ["predict", "view_reports"]  # optional
    }
    """
    username = token_request.get("username")
    password = token_request.get("password")
    
    # Validate required fields
    if not username or not password:
        raise HTTPException(
            status_code=400,
            detail="Username and password are required"
        )
    
    # Validate credentials length to prevent DoS attacks
    if len(username) > 256 or len(password) > 256:
        raise HTTPException(
            status_code=400,
            detail="Username or password too long"
        )
    
    # CRITICAL: Validate credentials against identity provider
    try:
        credentials_valid = validate_credentials(username, password)
    except NotImplementedError as e:
        logger.error(f"🚨 CREDENTIAL VALIDATION NOT CONFIGURED: {e}")
        raise HTTPException(
            status_code=501,
            detail="Service unavailable: Identity provider not configured"
        )
    except Exception as e:
        logger.error(f"🚨 Credential validation error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to validate credentials"
        )
    
    if not credentials_valid:
        # Use generic message to prevent username enumeration
        logger.warning(f"Failed login attempt for user: {username}")
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    
    # Set default permissions based on roles if not provided
    roles = token_request.get("roles", ["ml_engineer"])
    default_permissions = {
        "ml_engineer": ["predict", "view_reports"],
        "security_analyst": ["predict", "attack_test", "view_reports"],
        "admin": ["predict", "attack_test", "view_reports", "model_management"]
    }
    
    permissions = token_request.get("permissions")
    if not permissions:
        permissions = []
        for role in roles:
            permissions.extend(default_permissions.get(role, []))
        permissions = list(set(permissions))  # Remove duplicates
    
    # Create JWT payload
    now = datetime.now(timezone.utc)
    expiration = now + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        "sub": username,
        "iat": now,
        "exp": expiration,
        "roles": roles,
        "permissions": permissions,
        "jti": hashlib.sha256(os.urandom(32)).hexdigest()  # Unique token ID
    }
    
    # Sign and encode the token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    logger.info(f"✅ Token generated for user: {username}")
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
        "expires_at": expiration.isoformat(),
        "roles": roles,
        "permissions": permissions
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

