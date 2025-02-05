FROM nvidia/cuda:12.6.1-base-ubuntu24.04

WORKDIR /app

# Install Python, pip, and venv
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /app/venv

# Activate the virtual environment and install dependencies
COPY requirements.txt /app/
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/*.py /app/

# Set the default command to run inside the virtual environment
CMD ["/app/venv/bin/python", "app.py"]
