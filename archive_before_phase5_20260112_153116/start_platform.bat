@echo off
chcp 65001 >nul
echo.
echo ========================================
echo 🏢 ENTERPRISE ADVERSARIAL ML SECURITY PLATFORM
echo ========================================
echo.
echo This script will:
echo 1. Fix model loading issues
echo 2. Start the enterprise platform
echo 3. Open the API documentation
echo.
echo Press any key to continue...
pause >nul

echo.
echo 🔧 Step 1: Fixing model loading...
python fix_model.py

echo.
echo 🚀 Step 2: Starting enterprise platform...
echo.
echo The platform will start on port 8000
echo You can access it at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo 🛑 Press CTRL+C in this window to stop the platform
echo.

start "" http://localhost:8000/docs
python start_enterprise_working_fixed.py
