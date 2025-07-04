from flask import Flask, Response, send_file, jsonify, request
import cv2
import os
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO
import threading
import time
import torch

app = Flask(__name__)

# FPS setting
FPS = 60  # Default FPS

# Camera initialization
camera = cv2.VideoCapture(0)

# Global variables for latest data
latest_frame = None
latest_yolo_result = None
latest_apriltag_result = None
frame_lock = threading.Lock()
yolo_lock = threading.Lock()
apriltag_lock = threading.Lock()

# Detect Metal (MPS) support
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    yolo_device = torch.device('mps')
    print('Using Apple Silicon Metal (MPS) for YOLOv8 inference.')
else:
    yolo_device = torch.device('cpu')
    print('Using CPU for YOLOv8 inference.')

# Global YOLO model (load only once)
yolo_model = YOLO('yolov8n.pt')
yolo_model.to(yolo_device)

# Camera reader thread
def camera_reader():
    global latest_frame
    interval = 1.0 / FPS
    while True:
        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame
        time.sleep(interval)

# YOLO inference thread
def yolo_inferencer():
    global latest_yolo_result
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            # Convert frame to torch tensor and move to device if using MPS
            if yolo_device.type == 'mps':
                # YOLOv8 expects numpy array, but underlying torch ops会自动用MPS
                pass  # ultralytics会自动处理
            results = yolo_model(frame)[0]
            with yolo_lock:
                latest_yolo_result = results
        time.sleep(0.01)

# AprilTag inference thread
def apriltag_inferencer():
    global latest_apriltag_result
    detector = Detector(families='tag36h11')
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)
            tags = []
            for r in results:
                tags.append({
                    'id': r.tag_id,
                    'family': r.tag_family.decode() if hasattr(r.tag_family, 'decode') else str(r.tag_family),
                    'center': [float(x) for x in r.center],
                    'corners': [[float(x) for x in corner] for corner in r.corners]
                })
            with apriltag_lock:
                latest_apriltag_result = tags
        time.sleep(0.01)

# Start all threads
threading.Thread(target=camera_reader, daemon=True).start()
threading.Thread(target=yolo_inferencer, daemon=True).start()
threading.Thread(target=apriltag_inferencer, daemon=True).start()

def get_latest_frame():
    with frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
        else:
            return None

def get_latest_yolo_result():
    with yolo_lock:
        return latest_yolo_result

def get_latest_apriltag_result():
    with apriltag_lock:
        return latest_apriltag_result

@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(__file__), 'webui.html'))

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        frame_interval = 1.0 / FPS
        last_time = 0
        while True:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(frame_interval - (now - last_time))
                now = time.time()
            last_time = now
            frame = get_latest_frame()
            yolo_result = get_latest_yolo_result()
            if frame is None or yolo_result is None:
                time.sleep(0.01)
                continue
            frame_with_boxes = yolo_result.plot()
            ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/apriltags')
def apriltags_api():
    tags = get_latest_apriltag_result()
    if tags is None:
        return jsonify({'error': 'No AprilTag result yet'}), 500
    return jsonify({'tags': tags})

@app.route('/api/detections')
def detections_api():
    frame = get_latest_frame()
    yolo_result = get_latest_yolo_result()
    tags = get_latest_apriltag_result()
    if frame is None or yolo_result is None or tags is None:
        return jsonify({'error': 'Detection not ready'}), 500
    height, width = frame.shape[:2]
    # Parse YOLO detections
    detections = []
    for box, conf, cls in zip(yolo_result.boxes.xyxy.cpu().numpy(), yolo_result.boxes.conf.cpu().numpy(), yolo_result.boxes.cls.cpu().numpy()):
        detections.append({
            'bbox': [float(x) for x in box],
            'confidence': float(conf),
            'class_id': int(cls),
            'class_name': yolo_model.model.names[int(cls)]
        })
    return jsonify({'tags': tags, 'objects': detections, 'frame_width': width, 'frame_height': height})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 