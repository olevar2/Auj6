# Multi-stage build for AUJ Trading Platform
# Base image with Python 3.11 for optimal performance with financial libraries
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for trading platform
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source (required for technical analysis)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Update library cache
RUN ldconfig

# Create app user for security
RUN useradd --create-home --shell /bin/bash auj_user

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY auj_platform/requirements.txt ./requirements.txt
COPY auj_platform/dashboard/requirements.txt ./dashboard_requirements.txt

# Create a unified requirements file
RUN echo "# Combined requirements for AUJ Platform" > combined_requirements.txt && \
    cat requirements.txt >> combined_requirements.txt && \
    echo "" >> combined_requirements.txt && \
    echo "# Dashboard requirements" >> combined_requirements.txt && \
    cat dashboard_requirements.txt | grep -v "^#" | grep -v "^$" >> combined_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r combined_requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

# Copy application code
COPY auj_platform/ .

# Create necessary directories
RUN mkdir -p data logs config && \
    chown -R auj_user:auj_user /app

# Switch to non-root user
USER auj_user

# Expose ports
EXPOSE 8000 8501 3000 9090

# Default command for development
CMD ["python", "src/main.py"]

# Production stage
FROM base as production

# Copy only necessary application files
COPY auj_platform/src ./src
COPY auj_platform/config ./config
COPY auj_platform/dashboard ./dashboard
COPY auj_platform/__init__.py .
COPY auj_platform/pytest.ini .

# Create necessary directories and set permissions
RUN mkdir -p data logs && \
    chown -R auj_user:auj_user /app

# Switch to non-root user
USER auj_user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Production command
CMD ["python", "src/main.py"]