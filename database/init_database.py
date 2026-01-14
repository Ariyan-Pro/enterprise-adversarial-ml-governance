"""
📦 DATABASE INITIALIZATION SCRIPT - UPDATED WITH ALL 7 MODELS
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from database.config import DATABASE_CONFIG, init_database
from database.models.base import Base

# Import all 7 models
from database.models.deployment_identity import DeploymentIdentity
from database.models.model_registry import ModelRegistry
from database.models.security_memory import SecurityMemory
from database.models.autonomous_decisions import AutonomousDecision
from database.models.policy_versions import PolicyVersion
from database.models.operator_interactions import OperatorInteraction
from database.models.system_health_history import SystemHealthHistory

def create_database():
    """Create database if it doesn't exist"""
    try:
        # First, connect to default PostgreSQL database
        admin_engine = create_engine(DATABASE_CONFIG.test_connection_string)
        
        with admin_engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": DATABASE_CONFIG.database}
            ).fetchone()
            
            if not result:
                print(f"Creating database: {DATABASE_CONFIG.database}")
                conn.execute(text("COMMIT"))  # Exit transaction
                conn.execute(text(f'CREATE DATABASE "{DATABASE_CONFIG.database}"'))
                print("✅ Database created")
            else:
                print(f"✅ Database already exists: {DATABASE_CONFIG.database}")
                
    except OperationalError as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Install PostgreSQL: https://www.postgresql.org/download/")
        print("   2. Or use Docker: docker run --name security-db -p 5432:5432 -e POSTGRES_PASSWORD=postgres -d postgres")
        print("   3. Verify PostgreSQL service is running")
        print("   4. Update credentials in database/config.py if needed")
        return False
    
    return True

def create_tables():
    """Create all 7 tables in the database"""
    try:
        # Initialize database connection
        if not init_database():
            print("❌ Failed to initialize database connection")
            return False
        
        # Create all tables
        Base.metadata.create_all(bind=DATABASE_CONFIG.engine)
        print("✅ All tables created successfully")
        
        # Count tables created
        table_count = len(Base.metadata.tables)
        print(f"📊 Tables created: {table_count}")
        
        # List all tables
        with DATABASE_CONFIG.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result]
            print("📋 Table list:")
            for table in tables:
                print(f"   - {table}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_initial_deployment():
    """Create initial deployment identity"""
    from database.config import get_db_session
    import hashlib
    import platform
    import json
    from datetime import datetime
    
    with get_db_session() as session:
        # Check if deployment already exists
        existing = session.query(DeploymentIdentity).order_by(DeploymentIdentity.created_at.desc()).first()
        if existing:
            print(f"✅ Deployment already exists: {existing.deployment_id}")
            return existing
        
        # Create environment fingerprint
        env_data = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "processor": platform.processor(),
            "init_time": datetime.utcnow().isoformat()
        }
        
        env_json = json.dumps(env_data, sort_keys=True)
        env_hash = hashlib.sha256(env_json.encode()).hexdigest()
        
        # Create new deployment
        deployment = DeploymentIdentity(
            environment_hash=env_hash,
            environment_summary=env_data,
            default_risk_posture="balanced",
            system_maturity_score=0.1,  # Just starting
            policy_envelopes={
                "max_aggressiveness": 0.7,
                "false_positive_tolerance": 0.3,
                "learning_enabled": True,
                "emergency_ceilings": {
                    "confidence_threshold": 0.95,
                    "block_rate": 0.5
                }
            }
        )
        
        session.add(deployment)
        session.commit()
        
        print(f"✅ Initial deployment created: {deployment.deployment_id}")
        print(f"   Environment hash: {env_hash[:16]}...")
        print(f"   Risk posture: {deployment.default_risk_posture}")
        print(f"   Maturity score: {deployment.system_maturity_score}")
        
        return deployment

def register_existing_models():
    """Register existing models from Phase 4/5"""
    from database.config import get_db_session
    from database.models.model_registry import ModelRegistry
    
    with get_db_session() as session:
        # Check if models already registered
        existing_count = session.query(ModelRegistry).count()
        if existing_count > 0:
            print(f"✅ Models already registered: {existing_count}")
            return existing_count
        
        # Register Phase 5 ecosystem models
        models_to_register = [
            {
                "model_id": "mnist_cnn_v1",
                "domain": "vision",
                "risk_tier": "medium",
                "confidence_baseline": 0.85,
                "robustness_baseline": 0.88,
                "inherited_intelligence_score": 0.1,
                "owner": "adversarial-ml-suite"
            },
            {
                "model_id": "fraud_detector_v2",
                "domain": "tabular",
                "risk_tier": "critical",
                "confidence_baseline": 0.92,
                "robustness_baseline": 0.75,
                "inherited_intelligence_score": 0.3,
                "owner": "fraud-team"
            },
            {
                "model_id": "sentiment_analyzer_v1",
                "domain": "text",
                "risk_tier": "high",
                "confidence_baseline": 0.88,
                "robustness_baseline": 0.70,
                "inherited_intelligence_score": 0.2,
                "owner": "nlp-team"
            },
            {
                "model_id": "time_series_forecast_v3",
                "domain": "time_series",
                "risk_tier": "medium",
                "confidence_baseline": 0.85,
                "robustness_baseline": 0.65,
                "inherited_intelligence_score": 0.15,
                "owner": "forecasting-team"
            },
            {
                "model_id": "vision_segmentation_v2",
                "domain": "vision",
                "risk_tier": "high",
                "confidence_baseline": 0.89,
                "robustness_baseline": 0.72,
                "inherited_intelligence_score": 0.25,
                "owner": "vision-team"
            }
        ]
        
        registered = 0
        for model_data in models_to_register:
            model = ModelRegistry(**model_data)
            session.add(model)
            registered += 1
        
        session.commit()
        print(f"✅ Registered {registered} models in database")
        
        # Show registered models
        models = session.query(ModelRegistry).all()
        print("📋 Registered models:")
        for model in models:
            print(f"   - {model.model_id} ({model.domain}/{model.risk_tier})")
        
        return registered

def create_initial_policies():
    """Create initial policy versions"""
    from database.config import get_db_session
    from database.models.policy_versions import PolicyVersion
    import hashlib
    import json
    
    with get_db_session() as session:
        # Check if policies exist
        existing = session.query(PolicyVersion).count()
        if existing > 0:
            print(f"✅ Policies already exist: {existing}")
            return existing
        
        policies = []
        
        # 1. Confidence Threshold Policy
        confidence_policy = {
            "model_confidence_threshold": 0.7,
            "emergency_confidence_threshold": 0.5,
            "confidence_drop_tolerance": 0.3
        }
        
        content = {
            "policy_type": "confidence_threshold",
            "policy_scope": "global",
            "version": 1,
            "parameters": confidence_policy,
            "constraints": {"max_allowed_confidence_drop": 0.5}
        }
        
        version_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        
        policies.append(PolicyVersion(
            policy_type="confidence_threshold",
            policy_scope="global",
            version_number=1,
            version_hash=version_hash,
            policy_parameters=confidence_policy,
            policy_constraints={"max_allowed_confidence_drop": 0.5},
            change_reason="Initial deployment",
            change_trigger="human_intervention"
        ))
        
        # 2. Rate Limiting Policy
        rate_policy = {
            "requests_per_minute": 100,
            "burst_capacity": 50,
            "emergency_rate_limit": 20
        }
        
        content = {
            "policy_type": "rate_limiting",
            "policy_scope": "global",
            "version": 1,
            "parameters": rate_policy,
            "constraints": {"min_requests_per_minute": 1}
        }
        
        version_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        
        policies.append(PolicyVersion(
            policy_type="rate_limiting",
            policy_scope="global",
            version_number=1,
            version_hash=version_hash,
            policy_parameters=rate_policy,
            policy_constraints={"min_requests_per_minute": 1},
            change_reason="Initial deployment",
            change_trigger="human_intervention"
        ))
        
        # Add all policies
        for policy in policies:
            session.add(policy)
        
        session.commit()
        print(f"✅ Created {len(policies)} initial policies")
        
        return len(policies)

def main():
    """Main initialization routine"""
    print("\n" + "="*80)
    print("🧠 DATABASE INITIALIZATION - SECURITY NERVOUS SYSTEM (7 TABLES)")
    print("="*80)
    
    # Step 1: Create database
    print("\n1️⃣ CHECKING/CREATING DATABASE...")
    if not create_database():
        return False
    
    # Step 2: Create tables
    print("\n2️⃣ CREATING 7 TABLES...")
    if not create_tables():
        return False
    
    # Step 3: Create initial deployment
    print("\n3️⃣ CREATING DEPLOYMENT IDENTITY...")
    deployment = create_initial_deployment()
    if not deployment:
        return False
    
    # Step 4: Register existing models
    print("\n4️⃣ REGISTERING EXISTING MODELS...")
    model_count = register_existing_models()
    
    # Step 5: Create initial policies
    print("\n5️⃣ CREATING INITIAL POLICIES...")
    policy_count = create_initial_policies()
    
    print("\n" + "="*80)
    print("✅ DATABASE INITIALIZATION COMPLETE")
    print("="*80)
    print(f"Deployment ID: {deployment.deployment_id}")
    print(f"Models registered: {model_count}")
    print(f"Policies created: {policy_count}")
    print(f"Tables ready: 7 core tables")
    print("\n📋 TABLE SCHEMA SUMMARY:")
    print("   1. deployment_identity - Personalization per installation")
    print("   2. model_registry - Model governance across domains")
    print("   3. security_memory - Compressed threat experience")
    print("   4. autonomous_decisions - Autonomous decision audit trail")
    print("   5. policy_versions - Governance over time")
    print("   6. operator_interactions - Human-aware security")
    print("   7. system_health_history - Self-healing diagnostics")
    print("\n🚀 Database layer is now operational for Phase 5")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
