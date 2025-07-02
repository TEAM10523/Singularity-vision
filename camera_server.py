from flask import Flask, Response, send_file, jsonify, request
import cv2
import os
import numpy as np
from pupil_apriltags import Detector
from concurrent.futures import ProcessPoolExecutor

app = Flask(__name__)

# Camera initialization
camera = cv2.VideoCapture(0)

def detect_apriltags_in_frame(frame, families='tag36h11'):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = Detector(families=families)
    results = detector.detect(gray)
    tags = []
    for r in results:
        tags.append({
            'id': r.tag_id,
            'family': r.tag_family.decode() if hasattr(r.tag_family, 'decode') else str(r.tag_family),
            'center': [float(x) for x in r.center],
            'corners': [[float(x) for x in corner] for corner in r.corners]
        })
    return tags

@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(__file__), 'webui.html'))

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/apriltags')
def apriltags_api():
    # Read one frame for detection
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Camera read failed'}), 500
    # Use multi-core processing for detection
    with ProcessPoolExecutor() as executor:
        future = executor.submit(detect_apriltags_in_frame, frame)
        tags = future.result()
    return jsonify({'tags': tags})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 