# Use a lightweight python runtime
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Set working directory
WORKDIR /workspace

# Install system dependencies (build-essential for compiling certain python libs if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to cache dependency installation
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the CLIP model weights to cache them in the Docker image
RUN python -c "from langchain_experimental.open_clip import OpenCLIPEmbeddings; OpenCLIPEmbeddings(model_name='ViT-B-32', checkpoint='openai')"

# Copy the entire project code
COPY . .

# Run the Flask app using Gunicorn on the port specified by Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app.app:app
