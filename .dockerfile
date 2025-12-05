# Use NVIDIA CUDA base (includes CUDA toolkit and drivers runtime libs)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System deps: build tools, ssl, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    software-properties-common \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libgdbm-dev \
    liblzma-dev \
    libnccl2 || true \
  && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA to get Python 3.11 on Ubuntu 22.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg2 dirmngr && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3.11-venv \
      python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tk \
    libfreetype6-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Ensure pip for python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

# Make python3.11 the default `python` and `pip`
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 || true

# Copy requirements
WORKDIR /workspace
COPY requirements_wsl2.txt /workspace/requirements.txt

# Recommended environment variables for pip installs (no cache)
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install wheel first (speeds installs)
RUN pip install --upgrade pip setuptools wheel

# Install requirements (will install NVIDIA pip packages too)
RUN pip install --upgrade -r /workspace/requirements.txt

RUN pip install wandb
WORKDIR /home/workspace

# Default command: open a shell

CMD ["sleep", "infinity"]

