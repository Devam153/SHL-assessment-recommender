# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies manually
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    fastapi \
    uvicorn

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /var/cache/model

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables
ENV MODEL_CACHE_DIR=/var/cache/model

# Command to run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]