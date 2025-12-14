FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with OpenVINO for Intel CPU/GPU acceleration
RUN pip install --no-cache-dir \
    chromadb>=0.4.22 \
    "sentence-transformers>=2.2.0" \
    watchdog>=3.0.0 \
    mcp>=0.9.0 \
    openvino \
    optimum[openvino]

# Pre-download the sentence transformer model to avoid first-request timeout
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the server script
COPY server.py .

# Create directory for vector index
RUN mkdir -p /vector-index

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server
ENTRYPOINT ["python", "server.py"]
