@echo off
chcp 65001 >nul
echo.
echo ========================================
echo 🏢 ENTERPRISE ADVERSARIAL ML SECURITY PLATFORM
echo ========================================
echo.
echo Select platform to start:
echo.
echo [1] MAIN PLATFORM (Recommended)
echo     - Full enterprise features
echo     - Port 8000
echo     - http://localhost:8000
echo.
echo [2] SIMPLIFIED TEST
echo     - Basic API only
echo     - Port 8001  
echo     - http://localhost:8001
echo.
echo [3] CHECK DEPENDENCIES
echo     - Verify all requirements
echo.
echo [4] EXIT
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting MAIN PLATFORM...
    echo Access: http://localhost:8000
    echo Docs: http://localhost:8000/docs
    echo.
    echo 🛑 Press CTRL+C to stop
    echo.
    python enterprise_platform.py
) else if "%choice%"=="2" (
    echo.
    echo 🧪 Starting SIMPLIFIED TEST...
    echo Access: http://localhost:8001
    echo Docs: http://localhost:8001/docs
    echo.
    echo 🛑 Press CTRL+C to stop
    echo.
    python api_simple_test.py
) else if "%choice%"=="3" (
    echo.
    echo 🔧 Checking dependencies...
    python check_dependencies.py
    pause
    call %0
) else (
    echo.
    echo Exiting...
    exit /b 0
)
