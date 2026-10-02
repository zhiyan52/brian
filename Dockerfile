# ============================================================
# IBSR18 3D U-Net
# CUDA-enabled environment for 3D medical image segmentation
# ============================================================

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.11 explicitly
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# ------------------------------------------------------------
# Working directory
# ------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------
# Upgrade packaging tools
# ------------------------------------------------------------

RUN python -m pip install --upgrade pip setuptools

# ------------------------------------------------------------
# Install PyTorch
# ------------------------------------------------------------

RUN pip install \
    torch==2.14.0 \
    torchvision==0.24.0

# ------------------------------------------------------------
# Copy project metadata
# ------------------------------------------------------------

COPY pyproject.toml README.md ./

# ------------------------------------------------------------
# Install project dependencies
# ------------------------------------------------------------

RUN pip install -e .

# ------------------------------------------------------------
# Copy project source
# ------------------------------------------------------------

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/README.md ./data/README.md
COPY data/splits/ ./data/splits/

# ------------------------------------------------------------
# Install project package
# ------------------------------------------------------------

RUN pip install -e .

# ------------------------------------------------------------
# Default command
# ------------------------------------------------------------

CMD ["python", "-c", "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"]