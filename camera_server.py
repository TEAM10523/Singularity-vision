from flask import Flask, Response, send_file, jsonify, request
import cv2
import os
import numpy as np
from pupil_apriltags import Detector
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO

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

def detect_objects_in_frame(frame):
    model = YOLO('yolov8n.pt')
    results = model(frame)[0]
    detections = []
    for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        detections.append({
            'bbox': [float(x) for x in box],
            'confidence': float(conf),
            'class_id': int(cls),
            'class_name': model.model.names[int(cls)]
        })
    return detections

@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(__file__), 'webui.html'))

@app.route('/video_feed')
def video_feed():
    model = YOLO('yolov8n.pt')
    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            # YOLOv8 draw detection boxes directly on frame
            results = model(frame)[0]
            frame_with_boxes = results.plot()
            ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/apriltags')
def apriltags_api():
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Camera read failed'}), 500
    with ProcessPoolExecutor() as executor:
        future = executor.submit(detect_apriltags_in_frame, frame)
        tags = future.result()
    return jsonify({'tags': tags})

@app.route('/api/detections')
def detections_api():
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Camera read failed'}), 500
    height, width = frame.shape[:2]
    with ProcessPoolExecutor() as executor:
        f1 = executor.submit(detect_apriltags_in_frame, frame)
        f2 = executor.submit(detect_objects_in_frame, frame)
        tags = f1.result()
        objects = f2.result()
    return jsonify({'tags': tags, 'objects': objects, 'frame_width': width, 'frame_height': height})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 