FROM python:3.10-slim

WORKDIR /app

# 🧩 Install OS dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose a less common port to avoid conflicts
EXPOSE 8082

# Run the app on host 0.0.0.0 and port 8081
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]