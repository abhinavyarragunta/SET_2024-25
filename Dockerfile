FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install dependencies
RUN apt-get update && apt-get install -y gnupg2 curl && \
    curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main" | tee /etc/apt/sources.list.d/kitware.list > /dev/null && \
    apt-get update && apt-get install -y python3-pip python3-dev && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt


# Copy application code
COPY src/*.py /app/

CMD ["python3", "app.py"]
