# Use PyTorch official image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data and results directories
RUN mkdir -p data results/checkpoints results/samples results/plots

# Default command
CMD ["python", "src/train.py", "--config", "configs/default.yaml"]
