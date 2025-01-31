FROM python:3.9-slim

WORKDIR /app

ENV PATH="/usr/local/cuda-10.2/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt 

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    portaudio19-dev libasound2-dev libportaudio2 libportaudiocpp0 \
    v4l-utils nvidia-container-runtime

COPY src/*.py /app/

CMD ["python", "app.py"]
