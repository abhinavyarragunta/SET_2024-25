FROM nvcr.io/nvidia/l4t-base:r32.7.1 

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /app/venv

# Install Python dependencies
COPY requirements.txt /app/
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/*.py /app/

# Run application inside the virtual environment
CMD ["/app/venv/bin/python", "app.py"]
