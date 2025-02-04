FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt 

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    portaudio19-dev libasound2-dev libportaudio2 libportaudiocpp0 \
    v4l-utils && rm -rf /var/lib/apt/lists/*

COPY src/*.py /app/

CMD ["python", "app.py"]
