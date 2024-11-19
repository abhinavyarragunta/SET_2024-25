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

# Video and audio parameters
SAMPLE_RATE = 44100  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size
lock = threading.Lock()

# Robot movement control
current_direction = None

def continuous_movement():
    while True:
        if current_direction:
            print(f"Moving {current_direction}...")
            # Send movement command to the robot here for `current_direction`
            # Example: robot.move(current_direction)
        time.sleep(0.1)  # Adjust interval for smooth movement

# Start the continuous movement loop in a separate thread
movement_thread = threading.Thread(target=continuous_movement)
movement_thread.daemon = True
movement_thread.start()

@app.route('/stream', methods=['GET'])
def stream():
    """Stream video frames to the client."""
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/get-ip', methods=['GET'])
def get_ip():
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

    # Ensure only one direction is active at a time
    if state == "move":
        if current_direction != direction:
            current_direction = direction  # Set new direction
            print(f"Start moving {direction}")
    elif state == "stop" and current_direction == direction:
        current_direction = None
        print(f"Stop moving {direction}")
    print(f"Received direction: {direction}, state: {state}")
    return jsonify({"status": "success", "direction": direction, "state": state})


def generate_video():
    """Generate video frames for streaming."""
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        return

    while True:
        rval, frame = vc.read()
        if not rval:
            break
        with lock:
            _, encoded_image = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
    vc.release()

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

@socketio.on('audio_data_from_client')
def handle_audio_data_from_client(data):
    """Receive audio data from the client."""
    print("Received audio data from client:", data)
    # Process or save the incoming audio data as needed

if __name__ == '__main__':
    # Start audio capture in a separate thread
    threading.Thread(target=capture_audio).start()
    # Run the Flask-SocketIO app
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)