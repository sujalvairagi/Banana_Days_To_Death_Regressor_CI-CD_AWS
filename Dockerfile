FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and general image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full application
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose Flask port
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]
