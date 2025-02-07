from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import threading
from fall_detection_system import FallDetectionSystem
from dynamo import save_to_dynamo

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Video and audio parameters
SAMPLE_RATE = 44100  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size
lock = threading.Lock()

# Global variables
saving_data = False  # Flag to control data saving
streaming = False  # Flag to control video streaming
vc = None  # VideoCapture object


@app.route('/stream', methods=['GET'])
def stream():
    """Stream video frames to the client."""
    global streaming
    if not streaming:
        return jsonify({"message": "Streaming is off"}), 403
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/toggle_stream', methods=['POST'])
def toggle_stream():
    """Toggle the video stream on or off."""
    global streaming, vc
    data = request.get_json()

    if not data or 'action' not in data:
        return jsonify({"message": "Invalid request"}), 400

    action = data['action']

    if action == 'start':
        if not streaming:
            streaming = True
            return jsonify({"message": "Streaming started"}), 200
        return jsonify({"message": "Streaming already on"}), 200

    elif action == 'stop':
        streaming = False
        if vc:
            vc.release()
            vc = None
        return jsonify({"message": "Streaming stopped"}), 200

    return jsonify({"message": "Invalid action"}), 400


@app.route('/save_data', methods=['POST'])
def save_data():
        """Sends data to DynamoDB when triggered by a button."""
        ("Save_data function")
        command = request.get_json()

        if not command or 'runID' not in command:
            return jsonify({"message": "Invalid request, 'runID' required"}), 400

        # Example data structure
        data = {
            "runID": "A1"  # Unique identifier
        }

        # Send data to DynamoDB
        response = save_to_dynamo(data)

        if "item" in response:
            return jsonify({"message": "Data saved successfully", "data": response["item"]}), 200
        else:
            return jsonify(
                {"message": "Error saving to DynamoDB", "error": response.get("error", "Unknown error")}), 500


def generate_video():
    """Generate video frames for streaming."""
    global vc, streaming
    model_path = 'yolo11x-pose.pt'
    fall_system = FallDetectionSystem(model_path)

    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        streaming = False
        return

    while streaming:
        rval, frame = vc.read()
        if not rval:
            break

        # Run pose estimation on the captured frame
        processed_frame = fall_system.process_frame(frame)

        with lock:
            _, encoded_image = cv2.imencode(".jpg", processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

    vc.release()
    vc = None


if __name__ == '__main__':
    # Run the Flask app
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
