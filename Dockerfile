# Use a minimal Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the app code to the container
COPY . .

# Install required system dependencies (especially for FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for FastAPI (Railway expects app to use PORT env var)
EXPOSE 8000

# Run the app with Uvicorn using PORT from Railway
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
