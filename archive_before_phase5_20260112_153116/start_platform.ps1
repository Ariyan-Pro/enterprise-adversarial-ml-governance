Write-Host "`n" + "="*80 -ForegroundColor Green
Write-Host "🏢 ENTERPRISE ADVERSARIAL ML SECURITY PLATFORM" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green

Write-Host "`nThis script will:"
Write-Host "1. Fix model loading issues" -ForegroundColor Gray
Write-Host "2. Start the enterprise platform" -ForegroundColor Gray
Write-Host "3. Open the API documentation" -ForegroundColor Gray

Read-Host "`nPress Enter to continue"

Write-Host "`n🔧 Step 1: Fixing model loading..." -ForegroundColor Yellow
python fix_model.py

Write-Host "`n🚀 Step 2: Starting enterprise platform..." -ForegroundColor Green
Write-Host "`nThe platform will start on port 8000" -ForegroundColor White
Write-Host "You can access it at: http://localhost:8000" -ForegroundColor Gray
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "`n🛑 Press CTRL+C to stop the platform" -ForegroundColor Yellow

# Open browser to docs
Start-Process "http://localhost:8000/docs"

# Start the platform
python start_enterprise_working_fixed.py
