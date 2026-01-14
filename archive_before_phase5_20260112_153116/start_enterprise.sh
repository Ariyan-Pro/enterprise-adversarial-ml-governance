#!/bin/bash
echo ""
echo "🏢 ENTERPRISE ADVERSARIAL ML SECURITY PLATFORM"
echo "============================================"
echo ""
echo "Starting Enterprise API..."
echo ""
echo "Platform will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop"
echo ""
python api/main.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ API failed to start. Check errors above."
    exit 1
fi
