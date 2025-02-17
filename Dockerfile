# Use an official Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and TensorFlow
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install tensorflow

# Install additional required libraries (e.g., HuggingFace Transformers, etc.)
RUN pip install transformers

# Set the working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .
ENV TF_ENABLE_ONEDNN_OPTS=0


EXPOSE 5000
# Define the command to run the application
CMD ["python", "app.py"]
