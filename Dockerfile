FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    v4l-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install packages in stages to better handle dependencies
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir flask Flask-Cors Flask-SocketIO
RUN pip install --no-cache-dir scikit-build
RUN pip install --no-cache-dir opencv-python-headless  # Use headless version instead
RUN pip install --no-cache-dir sounddevice
RUN pip install --no-cache-dir cvzone
RUN pip install --no-cache-dir ultralytics
RUN pip install --no-cache-dir deep-sort-realtime

COPY src/*.py /app/

CMD ["python", "app.py"]
