FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libportaudio2 -y 

COPY src/*.py /app/

CMD ["python", "app.py"]
