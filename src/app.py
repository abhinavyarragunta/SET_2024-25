from flask import Flask, Response
from flask_cors import CORS
import socket
import cv2
import threading
app = Flask(__name__)
CORS(app)

# initialize a lock used to ensure thread-safe
# exchanges of the frames (useful for multiple browsers/tabs
# are viewing tthe stream)
lock = threading.Lock()
hostname = socket.gethostname()
ipv4_address = socket.gethostbyname(hostname)

@app.route('/stream',methods = ['GET'])
def stream():
   return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/get-ip', methods=['GET'])
def get_ip():
    return {'ip': ipv4_address}  # Replace '127.0.0.1' with the dynamic IP if needed.

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
   host = "127.0.0.1"
   port = 8000
   debug = False
   options = None
   app.run(host, port, debug, options, threaded=True)