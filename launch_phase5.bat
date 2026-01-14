@echo off
echo.
echo ========================================================================
echo ?? PHASE 5: STRATEGIC AUTONOMY LAUNCHER
echo ========================================================================
echo.
echo Starting Security Nervous System Transformation...
echo.
echo This script:
echo   1. Verifies Phase 5 ecosystem authority
echo   2. Tests multi-model governance
echo   3. Launches integrated platform
echo.

REM Check Python environment
python --version >nul 2>&1
if errorlevel 1 (
    echo ? Python not found. Install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo ? Python environment verified

REM Run ecosystem tests
echo.
echo ?? TESTING ECOSYSTEM AUTHORITY...
python test_ecosystem.py

if errorlevel 1 (
    echo.
    echo ??  Ecosystem tests had issues. Continuing anyway...
) else (
    echo.
    echo ? Ecosystem authority tests passed
)

REM Launch autonomous platform
echo.
echo ?? LAUNCHING AUTONOMOUS PLATFORM WITH ECOSYSTEM AUTHORITY...
cd autonomous
start cmd /k "launch.bat"

echo.
echo ?? Platform starting on: http://localhost:8000
echo ?? Ecosystem status: http://localhost:8000/autonomous/status
echo.
echo ?? Next actions:
echo   1. Open browser to check platform status
echo   2. Run: python intelligence/ecosystem_authority.py (for direct access)
echo   3. Review test_ecosystem.py results
echo.
echo ========================================================================
echo ? PHASE 5 LAUNCH SEQUENCE COMPLETE
echo ========================================================================
pause
