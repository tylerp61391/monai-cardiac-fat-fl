# MONAI Deploy Application Package (MAP) Dockerfile
# Conforms to MONAI Deploy standard: /var/monai/input, /var/monai/output, /opt/holoscan/models

FROM python:3.11-slim

LABEL maintainer="Tyler Patay <tcpatay123@gmail.com>"
LABEL description="MONAI Deploy Application Package for Automated 3D Cardiac Adipose (EAT & PVAT) Quantification"
LABEL repository="https://github.com/tylerp61391/monai-cardiac-fat-fl"
LABEL version="1.0.0"

# Set non-interactive debian environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create standard MAP directory layout
RUN mkdir -p /var/monai/input /var/monai/output /opt/holoscan/models /opt/holoscan/app

WORKDIR /opt/holoscan/app

# Copy dependency specifications and install Python packages
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir onnxruntime-gpu && \
    pip install --no-cache-dir -e .

# Copy models into standard Holoscan models directory
COPY models/ /opt/holoscan/models/

# Copy MAP entrypoint script
COPY monai_deploy_roadmap/app.py /opt/holoscan/app/app.py

# Verify models and checksum integrity during build
RUN python3 -c "import hashlib, sys; from pathlib import Path; \
    p = Path('/opt/holoscan/models/deepheartseg_heart.onnx'); \
    assert p.exists(), 'Model file missing!'; \
    h = hashlib.sha256(p.read_bytes()).hexdigest(); \
    assert h == '4015c53fa3fcdaa8b93e9d528f2410b8d34cdba07286e69b4f09a6e3c9644553', f'Checksum mismatch: {h}'; \
    print('Container model SHA-256 verified successfully.')"

# Default volume mounts
VOLUME ["/var/monai/input", "/var/monai/output"]

# Default entrypoint executing the 3-operator DAG
ENTRYPOINT ["python3", "/opt/holoscan/app/app.py"]
CMD ["-i", "/var/monai/input", "-o", "/var/monai/output", "-m", "/opt/holoscan/models"]

