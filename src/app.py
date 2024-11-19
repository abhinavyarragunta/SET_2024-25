from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import sounddevice as sd
import numpy as np
import threading
from vision import FallDetectionSystem

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Video and audio parameters
SAMPLE_RATE = 44100  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size
lock = threading.Lock()

def capture_audio():
    """Capture audio in real-time and send to the client."""
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        # Send audio data to React client
        socketio.emit('audio_data', indata.tolist())

    # Start audio stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        threading.Event().wait()  # Keep thread running

@app.route('/stream', methods=['GET'])
def stream():
    """Stream video frames to the client."""
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_video():
    """Generate video frames for streaming."""
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        return

    while True:
        rval, frame = vc.read()
        # If frame capture was unsuccessful, break the loop
        if not rval:
            break
        # Run pose estimation on the captured frame
        processed_frame = fall_system.process_frame(frame)
        with lock:
            _, encoded_image = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
    vc.release()

@socketio.on('audio_data_from_client')
def handle_audio_data_from_client(data):
    """Receive audio data from the client."""
    print("Received audio data from client:", data)
    # Process or save the incoming audio data as needed

if __name__ == '__main__':
    # Start audio capture in a separate thread
    #threading.Thread(target=capture_audio).start()
    # Run the Flask-SocketIO app
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
