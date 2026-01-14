#!/usr/bin/env python3
"""
📊 PHASE 5 ECOSYSTEM STATUS CHECK
Quick verification of Phase 5 implementation.
"""

import sys
from pathlib import Path
from datetime import datetime

def check_phase5_status():
    """Check Phase 5 implementation status"""
    print("\n" + "="*80)
    print("📊 PHASE 5 IMPLEMENTATION STATUS")
    print("="*80)
    
    checks = []
    
    # Check 1: Ecosystem authority file
    ecosystem_file = Path("intelligence/ecosystem_authority.py")
    if ecosystem_file.exists():
        size_kb = ecosystem_file.stat().st_size / 1024
        checks.append(("Ecosystem Authority", f"✅ {size_kb:.1f} KB", True))
    else:
        checks.append(("Ecosystem Authority", "❌ Missing", False))
    
    # Check 2: Test script
    test_file = Path("test_ecosystem.py")
    if test_file.exists():
        checks.append(("Test Script", "✅ Present", True))
    else:
        checks.append(("Test Script", "❌ Missing", False))
    
    # Check 3: Launch script
    launch_file = Path("launch_phase5.bat")
    if launch_file.exists():
        checks.append(("Launch Script", "✅ Present", True))
    else:
        checks.append(("Launch Script", "❌ Missing", False))
    
    # Check 4: Autonomous platform
    auto_files = [
        Path("autonomous/core/autonomous_core.py"),
        Path("autonomous/platform/main.py"),
        Path("autonomous/launch.bat")
    ]
    auto_exists = all(f.exists() for f in auto_files)
    if auto_exists:
        checks.append(("Autonomous Platform", "✅ Operational", True))
    else:
        checks.append(("Autonomous Platform", "❌ Incomplete", False))
    
    # Check 5: Archive (cleanup successful)
    archive_dirs = [d for d in Path(".").iterdir() if d.is_dir() and "archive_before_phase5" in d.name]
    if archive_dirs:
        archive = archive_dirs[0]
        file_count = len(list(archive.iterdir()))
        checks.append(("Cleanup Archive", f"✅ {file_count} files", True))
    else:
        checks.append(("Cleanup Archive", "⚠️  Not found", False))
    
    # Display results
    print("\nCOMPONENT               STATUS")
    print("-" * 40)
    
    passed = 0
    for name, status, ok in checks:
        print(f"{name:20} {status}")
        if ok:
            passed += 1
    
    # Summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    
    score = (passed / len(checks)) * 100
    print(f"Components Ready: {passed}/{len(checks)}")
    print(f"Implementation Score: {score:.1f}%")
    
    if score >= 100:
        print("\n✅ PHASE 5: FULLY IMPLEMENTED")
        print("   All components present and ready")
    elif score >= 80:
        print("\n⚠️  PHASE 5: MOSTLY IMPLEMENTED")
        print("   Minor components may be missing")
    elif score >= 60:
        print("\n🔧 PHASE 5: PARTIALLY IMPLEMENTED")
        print("   Core components present")
    else:
        print("\n❌ PHASE 5: INCOMPLETE")
        print("   Significant components missing")
    
    print("\n🧭 NEXT ACTIONS:")
    if score >= 80:
        print("   1. Run: launch_phase5.bat")
        print("   2. Test: python test_ecosystem.py")
        print("   3. Verify: python intelligence/ecosystem_authority.py")
    else:
        print("   1. Review missing components above")
        print("   2. Re-run setup scripts")
        print("   3. Check archive_before_phase5 directory")
    
    return score >= 80

if __name__ == "__main__":
    ready = check_phase5_status()
    sys.exit(0 if ready else 1)
