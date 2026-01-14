@echo off
echo.
echo ========================================================================
echo ?? AUTONOMOUS ADVERSARIAL ML SECURITY PLATFORM
echo ========================================================================
echo.
echo Starting 10-year survivability platform...
echo.
echo Platform will:
echo   1. Evolve without human intervention
echo   2. Tighten security when components fail
echo   3. Preserve knowledge for future engineers
echo   4. Survive for 10+ years
echo.
echo Core principle: Security tightens on failure
echo.
cd platform
echo Starting platform on port 8000...
python main.py
if errorlevel 1 (
    echo ? Failed to start platform
    pause
    exit /b 1
)
