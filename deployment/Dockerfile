# ENTERPRISE ADMIRAL ML SECURITY API - DOCKERFILE
# Production-grade container for security API deployment with HTTPS/TLS support

FROM python:3.9-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Install system dependencies including SSL tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    openssl \
    ca-certificates \
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
RUN mkdir -p /app/reports /app/logs /app/data /etc/ssl/certs /etc/ssl/private
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check (HTTPS)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ssl; import requests; requests.get('https://localhost:8443/api/health', timeout=2, verify='/etc/ssl/certs/server.crt')" || exit 1

# Expose HTTPS port
EXPOSE 8443

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV SECURITY_LEVEL=enterprise
ENV FORCE_HTTPS=true
ENV SSL_CERT_FILE=/etc/ssl/certs/server.crt
ENV SSL_KEY_FILE=/etc/ssl/private/server.key

# Run the API with HTTPS
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8443", "--workers", "4", "--ssl-keyfile", "/etc/ssl/private/server.key", "--ssl-certfile", "/etc/ssl/certs/server.crt"]
