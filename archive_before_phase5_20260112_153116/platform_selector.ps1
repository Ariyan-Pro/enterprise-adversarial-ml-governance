Write-Host "`n" + "="*80 -ForegroundColor Green
Write-Host "🏢 ENTERPRISE ADVERSARIAL ML SECURITY PLATFORM" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green

Write-Host "`nSelect platform to start:" -ForegroundColor White
Write-Host "`n[1] MAIN PLATFORM (Recommended)" -ForegroundColor Cyan
Write-Host "    • Full enterprise features" -ForegroundColor Gray
Write-Host "    • Port 8000" -ForegroundColor Gray
Write-Host "    • http://localhost:8000" -ForegroundColor Gray

Write-Host "`n[2] SIMPLIFIED TEST" -ForegroundColor Cyan
Write-Host "    • Basic API only" -ForegroundColor Gray
Write-Host "    • Port 8001" -ForegroundColor Gray
Write-Host "    • http://localhost:8001" -ForegroundColor Gray

Write-Host "`n[3] CHECK DEPENDENCIES" -ForegroundColor Cyan
Write-Host "    • Verify all requirements" -ForegroundColor Gray

Write-Host "`n[4] EXIT" -ForegroundColor Cyan

$choice = Read-Host "`nEnter choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host "`n🚀 Starting MAIN PLATFORM..." -ForegroundColor Green
        Write-Host "Access: http://localhost:8000" -ForegroundColor White
        Write-Host "Docs: http://localhost:8000/docs" -ForegroundColor White
        Write-Host "`n🛑 Press CTRL+C to stop" -ForegroundColor Yellow
        python enterprise_platform.py
    }
    "2" {
        Write-Host "`n🧪 Starting SIMPLIFIED TEST..." -ForegroundColor Green
        Write-Host "Access: http://localhost:8001" -ForegroundColor White
        Write-Host "Docs: http://localhost:8001/docs" -ForegroundColor White
        Write-Host "`n🛑 Press CTRL+C to stop" -ForegroundColor Yellow
        python api_simple_test.py
    }
    "3" {
        Write-Host "`n🔧 Checking dependencies..." -ForegroundColor Yellow
        python check_dependencies.py
        Read-Host "`nPress Enter to continue"
        .\platform_selector.ps1
    }
    default {
        Write-Host "`nExiting..." -ForegroundColor Gray
    }
}
