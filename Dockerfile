FROM nvidia/cuda:12.6.1-base-ubuntu24.04

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Set default Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt 

# Install additional dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    portaudio19-dev libasound2-dev libportaudio2 libportaudiocpp0 \
    v4l-utils && rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY src/*.py /app/

CMD ["python", "app.py"]
