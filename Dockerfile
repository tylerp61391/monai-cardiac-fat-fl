# ==============================================================================
# Dockerfile: monai-cardiac-fat-fl
# Modernized MONAI & NVFLARE Pipeline for EAT & PVAT Cardiac Fat Quantification
# ==============================================================================

FROM python:3.11-slim

LABEL maintainer="Antigravity Advanced Agentic Coding"
LABEL description="Containerized MONAI & NVFLARE Cardiac Fat Quantification Pipeline"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install required system runtime libraries for SimpleITK and medical imaging I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy build configuration and source code
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY models ./models

# Install dependencies and the cardiac_fat_fl package in production mode
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Create persistent mount directories for input CT scans and output results
RUN mkdir -p /data /data/output /data/exports

WORKDIR /data

# Expose port 8000 for OHIF Web Viewer server
EXPOSE 8000

# Default entrypoint to the CLI tool
ENTRYPOINT ["cardiac-fat-fl"]
CMD ["--help"]

