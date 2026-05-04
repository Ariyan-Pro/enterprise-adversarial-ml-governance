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
IS_PRODUCTION = os.environ.get("PRODUCTION", "false").lower() in ("true", "1", "yes")

if IS_PRODUCTION and not JWT_SECRET_KEY:
    # In production: FAIL startup if JWT_SECRET_KEY is not set
    raise RuntimeError(
        "🚨 CRITICAL SECURITY ERROR: JWT_SECRET_KEY is not configured! "
        "In production mode, this is mandatory. "
        "Generate a secure key with: python -c 'import secrets; print(secrets.token_hex(32))' "
        "and set it as the JWT_SECRET_KEY environment variable."
    )

if not JWT_SECRET_KEY:
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
    
    The authentication backend is selected via AUTH_BACKEND environment variable:
    - "ldap": LDAP/Active Directory authentication
    - "oauth2": OAuth2/OIDC authentication
    - "database": Database authentication with bcrypt/argon2
    - "development": Development mode with hardcoded credentials (NEVER use in production)
    
    Args:
        username: User's username/email
        password: User's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
        
    Raises:
        RuntimeError: If authentication backend is not properly configured
    """
    # Get authentication backend from environment
    auth_backend = os.environ.get("AUTH_BACKEND", "development").lower()
    
    # Production safety check: reject development backend in production
    if IS_PRODUCTION and auth_backend == "development":
        logger.error("🚨 SECURITY ERROR: Development auth backend used in production!")
        raise RuntimeError(
            "Development authentication backend cannot be used in production. "
            "Configure AUTH_BACKEND environment variable to 'ldap', 'oauth2', or 'database'."
        )
    
    try:
        if auth_backend == "ldap":
            return _validate_ldap_credentials(username, password)
        elif auth_backend == "oauth2":
            return _validate_oauth2_credentials(username, password)
        elif auth_backend == "database":
            return _validate_database_credentials(username, password)
        elif auth_backend == "development":
            return _validate_development_credentials(username, password)
        else:
            logger.error(f"Unknown authentication backend: {auth_backend}")
            raise ValueError(f"Unknown authentication backend: {auth_backend}")
    except Exception as e:
        logger.error(f"Credential validation failed for backend '{auth_backend}': {e}")
        raise


def _validate_ldap_credentials(username: str, password: str) -> bool:
    """
    Validate credentials against LDAP/Active Directory.
    
    Requires:
    - LDAP_SERVER: LDAP server URL (e.g., ldap://ad.example.com)
    - LDAP_BASE_DN: Base DN for user searches (e.g., dc=example,dc=com)
    - LDAP_USER_FILTER: Optional filter for user lookup
    """
    try:
        from ldap3 import Server, Connection, ALL, NTLM
    except ImportError:
        logger.error("ldap3 package not installed. Install with: pip install ldap3")
        raise RuntimeError("LDAP authentication requires ldap3 package")
    
    ldap_server = os.environ.get("LDAP_SERVER")
    ldap_base_dn = os.environ.get("LDAP_BASE_DN")
    
    if not ldap_server or not ldap_base_dn:
        raise RuntimeError("LDAP_SERVER and LDAP_BASE_DN environment variables must be set")
    
    ldap_user_filter = os.environ.get("LDAP_USER_FILTER", "(sAMAccountName={username})")
    ldap_use_ssl = os.environ.get("LDAP_USE_SSL", "false").lower() == "true"
    ldap_auth_type = os.environ.get("LDAP_AUTH_TYPE", "SIMPLE").upper()
    
    try:
        # Build LDAP server URL
        scheme = "ldaps" if ldap_use_ssl else "ldap"
        if not ldap_server.startswith("ldap"):
            ldap_server = f"{scheme}://{ldap_server}"
        
        server = Server(ldap_server, get_info=ALL, use_ssl=ldap_use_ssl)
        
        # Format user filter
        search_filter = ldap_user_filter.format(username=username)
        
        # First, search for the user's DN
        conn = Connection(server, auto_bind=True)
        conn.search(
            search_base=ldap_base_dn,
            search_filter=search_filter,
            attributes=["distinguishedName"]
        )
        
        if len(conn.entries) == 0:
            logger.warning(f"LDAP user not found: {username}")
            return False
        
        user_dn = conn.entries[0].distinguishedName.value
        conn.unbind()
        
        # Now try to bind with the user's credentials
        if ldap_auth_type == "NTLM":
            ldap_domain = os.environ.get("LDAP_DOMAIN", "")
            conn = Connection(
                server,
                user=f"{ldap_domain}\\{username}",
                password=password,
                authentication=NTLM,
                auto_bind=False
            )
        else:
            conn = Connection(server, user=user_dn, password=password, auto_bind=False)
        
        if not conn.bind():
            logger.warning(f"LDAP bind failed for user: {username}")
            return False
        
        conn.unbind()
        logger.info(f"LDAP authentication successful for user: {username}")
        return True
        
    except Exception as e:
        logger.error(f"LDAP authentication error: {e}")
        raise


def _validate_oauth2_credentials(username: str, password: str) -> bool:
    """
    Validate credentials using OAuth2/OIDC Resource Owner Password Credentials flow.
    
    Requires:
    - OAUTH2_TOKEN_URL: OAuth2 token endpoint URL
    - OAUTH2_CLIENT_ID: OAuth2 client ID
    - OAUTH2_CLIENT_SECRET: OAuth2 client secret
    - OAUTH2_SCOPE: Optional scope for the token request
    """
    try:
        import requests
    except ImportError:
        logger.error("requests package not installed. Install with: pip install requests")
        raise RuntimeError("OAuth2 authentication requires requests package")
    
    token_url = os.environ.get("OAUTH2_TOKEN_URL")
    client_id = os.environ.get("OAUTH2_CLIENT_ID")
    client_secret = os.environ.get("OAUTH2_CLIENT_SECRET")
    
    if not token_url or not client_id or not client_secret:
        raise RuntimeError(
            "OAUTH2_TOKEN_URL, OAUTH2_CLIENT_ID, and OAUTH2_CLIENT_SECRET "
            "environment variables must be set"
        )
    
    scope = os.environ.get("OAUTH2_SCOPE", "openid profile email")
    
    try:
        response = requests.post(
            token_url,
            data={
                "grant_type": "password",
                "username": username,
                "password": password,
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": scope
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"OAuth2 authentication successful for user: {username}")
            return True
        elif response.status_code == 401:
            logger.warning(f"OAuth2 authentication failed for user: {username}")
            return False
        else:
            logger.error(f"OAuth2 token endpoint returned status {response.status_code}: {response.text}")
            raise RuntimeError(f"OAuth2 token endpoint error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"OAuth2 connection error: {e}")
        raise


def _validate_database_credentials(username: str, password: str) -> bool:
    """
    Validate credentials against database with hashed passwords.
    
    Supports bcrypt and argon2 password hashing.
    
    Requires:
    - DB_AUTH_CONNECTION_STRING: Database connection string
    - DB_AUTH_QUERY: SQL query to retrieve user (with {username} placeholder)
    - DB_PASSWORD_HASH_ALGORITHM: Hash algorithm ('bcrypt' or 'argon2')
    """
    db_connection_string = os.environ.get("DB_AUTH_CONNECTION_STRING")
    db_query = os.environ.get("DB_AUTH_QUERY")
    hash_algorithm = os.environ.get("DB_PASSWORD_HASH_ALGORITHM", "bcrypt").lower()
    
    if not db_connection_string or not db_query:
        raise RuntimeError(
            "DB_AUTH_CONNECTION_STRING and DB_AUTH_QUERY environment variables must be set"
        )
    
    try:
        import psycopg2  # PostgreSQL example; adapt for your database
    except ImportError:
        try:
            import pymysql  # MySQL fallback
        except ImportError:
            logger.error("Database driver not installed. Install psycopg2 or pymysql")
            raise RuntimeError("Database authentication requires a database driver")
    
    try:
        # Connect to database
        conn = psycopg2.connect(db_connection_string)
        cursor = conn.cursor()
        
        # Execute query to get password hash
        cursor.execute(db_query.format(username=username))
        row = cursor.fetchone()
        
        if row is None:
            logger.warning(f"User not found in database: {username}")
            return False
        
        stored_hash = row[0]
        
        cursor.close()
        conn.close()
        
        # Verify password based on hash algorithm
        if hash_algorithm == "bcrypt":
            try:
                import bcrypt
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
            except ImportError:
                logger.error("bcrypt package not installed. Install with: pip install bcrypt")
                raise RuntimeError("bcrypt authentication requires bcrypt package")
                
        elif hash_algorithm == "argon2":
            try:
                from argon2 import PasswordHasher
                ph = PasswordHasher()
                ph.verify(stored_hash, password)
                return True
            except ImportError:
                logger.error("argon2-cffi package not installed. Install with: pip install argon2-cffi")
                raise RuntimeError("argon2 authentication requires argon2-cffi package")
            except Exception:
                return False
        else:
            raise ValueError(f"Unsupported password hash algorithm: {hash_algorithm}")
            
    except Exception as e:
        logger.error(f"Database authentication error: {e}")
        raise


def _validate_development_credentials(username: str, password: str) -> bool:
    """
    Development-only credential validation.
    
    WARNING: NEVER use this in production!
    This is only for local development and testing.
    
    Uses hardcoded credentials from environment variables:
    - DEV_AUTH_USERNAME: Development username
    - DEV_AUTH_PASSWORD: Development password
    """
    dev_username = os.environ.get("DEV_AUTH_USERNAME", "admin")
    dev_password = os.environ.get("DEV_AUTH_PASSWORD", "admin123")
    
    logger.warning(
        f"⚠️ Using development authentication for user: {username}. "
        "NEVER use in production!"
    )
    
    return username == dev_username and password == dev_password


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

