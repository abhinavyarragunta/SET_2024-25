FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

# Copy application code
COPY src/*.py /app/

CMD ["python3", "app.py"]

