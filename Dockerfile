# Provenance Analysis - Content Sharing Detection Container
# Based on conda for cyvlfeat support
FROM continuumio/miniconda3:24.7.1-0

LABEL maintainer="ELIS System"
LABEL description="Content sharing detection for provenance analysis"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.10
RUN conda create -n provenance python=3.10 -y

# Activate environment and install cyvlfeat via conda
SHELL ["conda", "run", "-n", "provenance", "/bin/bash", "-c"]

# Install cyvlfeat from conda-forge (required for vlfeat SIFT)
RUN conda install -c conda-forge cyvlfeat -y

# Copy requirements and install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories for input/output
RUN mkdir -p /data/input /data/output

# Set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "provenance", "python", "-m", "src.main"]

# Default command (can be overridden)
CMD ["--help"]
