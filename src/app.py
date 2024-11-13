from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import socket
import cv2
import threading
import time
app = Flask(__name__)
CORS(app)

# initialize a lock used to ensure thread-safe
# exchanges of the frames (useful for multiple browsers/tabs
# are viewing tthe astream)
lock = threading.Lock()
hostname = socket.gethostname()
ipv4_address = socket.gethostbyname(hostname)

@app.route('/stream',methods = ['GET'])
def stream():
   return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/get-ip', methods=['GET'])
def get_ip():
    ipv4_address = socket.gethostbyname(socket.gethostname())
    local_network_ip = socket.gethostbyname(socket.getfqdn())
    return {'ip': ipv4_address, 'local_network_ip' : local_network_ip}  # Replace '127.0.0.1' with the dynamic IP if needed.

#NEW CHANGES

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

@app.route('/direction', methods=['POST'])
def direction():
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

    return jsonify({"status": "success", "direction": direction, "state": state})

# End of new changesa

def generate():
   # grab global references to the lock variable
   global lock
   # initialize the video stream
   vc = cv2.VideoCapture(0)
   
   # check camera is open
   if vc.isOpened():
      rval, frame = vc.read()
   else:
      rval = False

   # while streaming
   while rval:
      # wait until the lock is acquired
      with lock:
         # read next frame
         rval, frame = vc.read()
         # if blank frame
         if frame is None:
            continue

         # encode the frame in JPEG format
         (flag, encodedImage) = cv2.imencode(".jpg", frame)

         # ensure the frame was successfully encoded
         if not flag:
            continue

      # yield the output frame in the byte format
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
   # release the camera
   vc.release()

if __name__ == '__main__':
   host = "0.0.0.0"
   port = 8000
   debug = False
   options = None
   app.run(host, port, debug, options, threaded=True)