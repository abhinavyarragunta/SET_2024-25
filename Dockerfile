FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install dependencie
# Remove Kitware repository if it exists
RUN rm -f /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y gnupg2 curl && \
    apt-get install -y python3-pip python3-dev && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt




# Copy application code
COPY src/*.py /app/

CMD ["python3", "app.py"]
