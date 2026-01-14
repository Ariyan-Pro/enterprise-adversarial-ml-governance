#!/usr/bin/env python3
"""
🔍 PHASE 5.1 VERIFICATION - Final check of all components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_database_models():
    """Verify all 7 database models exist"""
    print("\n📦 VERIFYING DATABASE MODELS (7 TABLES)...")
    
    models = [
        "deployment_identity.py",
        "model_registry.py", 
        "security_memory.py",
        "autonomous_decisions.py",
        "policy_versions.py",
        "operator_interactions.py",
        "system_health_history.py"
    ]
    
    all_exist = True
    for model in models:
        path = Path("database/models") / model
        if path.exists():
            print(f"   ✅ {model}")
        else:
            print(f"   ❌ {model}")
            all_exist = False
    
    return all_exist

def verify_engine_components():
    """Verify Phase 5 engine components"""
    print("\n🧠 VERIFYING PHASE 5 ENGINE COMPONENTS...")
    
    components = [
        "autonomous/core/database_engine.py",
        "autonomous/core/compatibility.py",
        "database/config.py",
        "database/init_database.py"
    ]
    
    all_exist = True
    for component in components:
        path = Path(component)
        if path.exists():
            print(f"   ✅ {component}")
        else:
            print(f"   ❌ {component}")
            all_exist = False
    
    return all_exist

def verify_execution_scripts():
    """Verify execution scripts"""
    print("\n🚀 VERIFYING EXECUTION SCRIPTS...")
    
    scripts = [
        "execute_phase5.py",
        "test_phase5_engine.py", 
        "setup_postgresql.py",
        "test_database.py"
    ]
    
    all_exist = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script}")
            all_exist = False
    
    return all_exist

def test_minimal_integration():
    """Test minimal integration"""
    print("\n🧪 TESTING MINIMAL INTEGRATION...")
    
    try:
        # Test database engine creation
        from autonomous.core.database_engine import create_phase5_engine
        
        engine = create_phase5_engine()
        print("   ✅ Phase 5 engine created")
        
        # Test basic functionality
        health = engine.get_ecosystem_health()
        print(f"   ✅ Ecosystem health check: {health.get('health_score', 0.0):.3f}")
        
        models = engine.get_models_by_domain("vision")
        print(f"   ✅ Model retrieval: {len(models)} vision models")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Main verification routine"""
    print("\n" + "="*80)
    print("🔍 PHASE 5.1 COMPONENT VERIFICATION")
    print("="*80)
    
    # Verify components
    models_ok = verify_database_models()
    engine_ok = verify_engine_components()
    scripts_ok = verify_execution_scripts()
    
    # Test integration
    integration_ok = test_minimal_integration()
    
    print("\n" + "="*80)
    print("📊 VERIFICATION RESULTS")
    print("="*80)
    
    all_ok = models_ok and engine_ok and scripts_ok and integration_ok
    
    if all_ok:
        print("✅ PHASE 5.1: FULLY VERIFIED")
        print("\n🎯 COMPONENTS STATUS:")
        print("   Database Models: 7/7 tables ✓")
        print("   Engine Components: 4/4 files ✓")
        print("   Execution Scripts: 4/4 scripts ✓")
        print("   Integration Test: PASSED ✓")
        
        print("\n🚀 READY FOR:")
        print("   • Phase 5.2: Ecosystem Authority")
        print("   • Production deployment with PostgreSQL")
        print("   • Multi-domain security operations")
        
    else:
        print("⚠️  PHASE 5.1: PARTIALLY VERIFIED")
        print("\n🔧 ISSUES FOUND:")
        if not models_ok:
            print("   • Missing database models")
        if not engine_ok:
            print("   • Missing engine components")
        if not scripts_ok:
            print("   • Missing execution scripts")
        if not integration_ok:
            print("   • Integration test failed")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Run the creation scripts again")
        print("   2. Check file permissions")
        print("   3. Verify Python dependencies")
    
    print("\n" + "="*80)
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
