FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install Python 3.8 and dependencies
RUN RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2 && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8 && \
    apt-get install -y python3-pip python3-dev && \
    apt-get clean

# Upgrade pip and install requirements
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY src/*.py /app/

CMD ["python3", "app.py"]
