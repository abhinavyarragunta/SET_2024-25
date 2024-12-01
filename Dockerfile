FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libportaudio2 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY src/*.py /app/

CMD ["python", "app.py"]
