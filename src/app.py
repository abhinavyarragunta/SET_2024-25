from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import socket
import cv2
import threading
import time
import sounddevice as sd

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Constants for video and audio parameters
SAMPLE_RATE = 44100  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size
lock = threading.Lock()

# Global state for robot movement
current_direction = None

# Start continuous movement in a separate thread
def continuous_movement():
    while True:
        if current_direction:
            print(f"Moving {current_direction}...")
            # Send movement command to robot
        time.sleep(0.1)

movement_thread = threading.Thread(target=continuous_movement, daemon=True)
movement_thread.start()

@app.route('/stream', methods=['GET'])
def stream():
    """Stream video frames to the client."""
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/get-ip', methods=['GET'])
def get_ip():
    """Return the IP addresses of the server."""
    ipv4_address = socket.gethostbyname(socket.gethostname())
    local_network_ip = socket.gethostbyname(socket.getfqdn())
    return {'ip': ipv4_address, 'local_network_ip': local_network_ip}

@app.route('/direction', methods=['POST'])
def direction():
    """Handle robot movement commands."""
    global current_direction
    data = request.json
    direction = data.get("direction")
    state = data.get("state")

    # Set direction and state
    if state == "move" and current_direction != direction:
        current_direction = direction
        print(f"Start moving {direction}")
    elif state == "stop" and current_direction == direction:
        current_direction = None
        print(f"Stop moving {direction}")

    return jsonify({"status": "success", "direction": direction, "state": state})

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """Process incoming audio stream data from React."""
    print("Received audio stream data:", data)
    # Further audio data processing can be done here

def generate_video():
    """Generate video frames for streaming."""
    vc = cv2.VideoCapture(0)  # Initialize VideoCapture
    if not vc.isOpened():
        return

    try:
        while True:
            rval, frame = vc.read()  # Read frame from the video capture device
            if not rval:
                break
            with lock:
                _, encoded_image = cv2.imencode(".jpg", frame)  # Encode the frame as JPEG
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
    finally:
        vc.release()  # Release the VideoCapture object after the loop is done


def capture_audio():
    """Capture audio in real-time and send to React."""
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        # Emit audio data to React client
        socketio.emit('audio_data', indata.tolist())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        threading.Event().wait()  # Keep thread running

@socketio.on('audio_data_from_client')
def handle_audio_data_from_client(data):
    """Receive audio data from React client."""
    print("Received audio data from client:", data)
    # Further audio data processing can be done here

if __name__ == '__main__':
    # Start audio capture in a separate thread
    threading.Thread(target=capture_audio, daemon=True).start()
    # Run Flask-SocketIO app
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
