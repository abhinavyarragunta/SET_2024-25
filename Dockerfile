FROM python:3.9-slim

WORKDIR /app

# Set Python environment variables explicitly
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/usr/local/bin:${PATH}"

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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip using python -m to ensure we're using the right interpreter
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install packages with explicit python3 -m pip calls
RUN python3 -m pip install --no-cache-dir numpy
RUN python3 -m pip install --no-cache-dir flask Flask-Cors Flask-SocketIO
RUN python3 -m pip install --no-cache-dir scikit-build
RUN python3 -m pip install --no-cache-dir opencv-python-headless
RUN python3 -m pip install --no-cache-dir sounddevice
RUN python3 -m pip install --no-cache-dir cvzone
RUN python3 -m pip install --no-cache-dir ultralytics
RUN python3 -m pip install --no-cache-dir deep-sort-realtime

COPY src/*.py /app/

CMD ["python3", "app.py"]
