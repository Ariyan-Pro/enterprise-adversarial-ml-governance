# ENTERPRISE ADMIRAL ML SECURITY API - DOCKERFILE
# Production-grade container for security API deployment

FROM python:3.9-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY api/requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/reports /app/logs /app/data
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=2)"

# Expose API port
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV SECURITY_LEVEL=enterprise

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
