FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Add NVIDIA package repository and install CUDA tools (optional, if needed)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/arm64/ubuntu1804/cuda-ubuntu1804.pin && \
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    sudo wget https://developer.download.nvidia.com/compute/cuda/repos/arm64/ubuntu1804/cuda-arm64.repo && \
    sudo mv cuda-arm64.repo /etc/apt/sources.list.d/cuda.list && \
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/arm64/ubuntu1804/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y cuda-toolkit-10-2 && \
    apt-get clean

# Install Python 3.8 and dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2 && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8 && \
    apt-get install -y python3-pip python3-dev && \
    apt-get clean && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY src/*.py /app/

CMD ["python3", "app.py"]

