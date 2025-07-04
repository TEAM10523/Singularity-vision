from flask import Flask, Response, send_file, jsonify, request
import cv2
import os
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO
import threading
import time
import torch
import coremltools as ct
from PIL import Image
import torchvision

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

# Inference backend selection: 'torch' or 'coreml'
INFERENCE_BACKEND = 'coreml'  # Switch to 'coreml' to use CoreML

# Device determination and model loading are only performed when using torch as the inference backend
if INFERENCE_BACKEND == 'torch':
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
else:
    yolo_model = None
    print('Using CoreML for YOLOv8 inference.')

# CoreML model loading (lazy loading)
coreml_model = None
coreml_class_names = None

def load_coreml_model():
    global coreml_model, coreml_class_names
    if coreml_model is None:
        coreml_model = ct.models.MLModel('yolov8n.mlpackage')
        # Directly access the class name using dict
        import ast
        coreml_class_names = ast.literal_eval(coreml_model.get_spec().description.metadata.userDefined['names'])

# CoreML Preprocessing
def preprocess_for_coreml(frame):
    # BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # resize to 640x640
    img = cv2.resize(img, (640, 640))
    # Convert to PIL.Image
    pil_img = Image.fromarray(img)
    return pil_img

def nms_coreml(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep.numpy()

def postprocess_coreml_output(output, conf_thres=0.25, iou_thres=0.5):
    arr = output['var_914']
    arr = np.array(arr)
    arr = arr.reshape(1, 84, 8400)
    arr = arr[0]  # (84, 8400)
    boxes = arr[:4, :].T  # (8400, 4)
    scores = arr[4:84, :].T  # (8400, 80)
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    keep = confidences > conf_thres
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    # Assuming boxes are in xywh format, convert to xyxy
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    # NMS
    keep_idx = nms_coreml(xyxy_boxes, confidences, iou_threshold=iou_thres)
    xyxy_boxes = xyxy_boxes[keep_idx]
    confidences = confidences[keep_idx]
    class_ids = class_ids[keep_idx]
    return {
        'boxes': xyxy_boxes,
        'confidences': confidences,
        'class_ids': class_ids
    }

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
            if INFERENCE_BACKEND == 'torch':
                results = yolo_model(frame, verbose=False)[0]
                with yolo_lock:
                    latest_yolo_result = results
            elif INFERENCE_BACKEND == 'coreml':
                load_coreml_model()
                input_data = preprocess_for_coreml(frame)
                # CoreML input named 'image'
                result = coreml_model.predict({'image': input_data})
                yolo_result = postprocess_coreml_output(result)
                with yolo_lock:
                    latest_yolo_result = yolo_result
        time.sleep(1.0 / FPS)

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
        time.sleep(1.0 / FPS)

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
            if INFERENCE_BACKEND == 'torch':
                frame_with_boxes = yolo_result.plot()
            elif INFERENCE_BACKEND == 'coreml':
                frame_with_boxes = frame.copy()
                boxes = yolo_result['boxes']
                class_ids = yolo_result['class_ids']
                confidences = yolo_result['confidences']
                h, w = frame_with_boxes.shape[:2]
                scale_x = w / 640.0
                scale_y = h / 640.0
                # Generate categorical color table
                num_classes = len(coreml_class_names) if coreml_class_names else 80
                colors = [tuple(int(x*255) for x in cv2.cvtColor(np.uint8([[[int(i*255/num_classes),255,255]]]), cv2.COLOR_HSV2BGR)[0][0]) for i in range(num_classes)]
                thickness = max(2, int(0.002 * (w + h) / 2))
                font_scale = max(0.5, 0.001 * (w + h) / 2)
                for box, cls, conf in zip(boxes, class_ids, confidences):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    color = colors[int(cls) % num_classes]
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, thickness)
                    label = f"{coreml_class_names[int(cls)] if coreml_class_names else str(cls)} {conf:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(2, thickness))
                    # Draw a darker base color (use black semi-transparent)
                    y0 = max(0, y1 - th - baseline)
                    y1_clip = min(h, y1)
                    x0 = max(0, x1)
                    x1_clip = min(w, x1 + tw)
                    if y0 < y1_clip and x0 < x1_clip:
                        overlay = frame_with_boxes.copy()
                        cv2.rectangle(overlay, (x0, y0), (x1_clip, y1_clip), (0,0,0), -1)
                        alpha = 0.7
                        frame_with_boxes[y0:y1_clip, x0:x1_clip] = cv2.addWeighted(
                            overlay[y0:y1_clip, x0:x1_clip], alpha,
                            frame_with_boxes[y0:y1_clip, x0:x1_clip], 1 - alpha, 0)
                    # Draw the black outline first, then the white bold font.
                    cv2.putText(frame_with_boxes, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), max(2, thickness), lineType=cv2.LINE_AA)
                    cv2.putText(frame_with_boxes, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), max(2, thickness-1), lineType=cv2.LINE_AA)
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
    detections = []
    if INFERENCE_BACKEND == 'torch':
        # Parse YOLO detections (torch)
        for box, conf, cls in zip(yolo_result.boxes.xyxy.cpu().numpy(), yolo_result.boxes.conf.cpu().numpy(), yolo_result.boxes.cls.cpu().numpy()):
            detections.append({
                'bbox': [float(x) for x in box],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': yolo_model.model.names[int(cls)]
            })
    elif INFERENCE_BACKEND == 'coreml':
        # Parse CoreML detections
        boxes = yolo_result['boxes']
        confidences = yolo_result['confidences']
        class_ids = yolo_result['class_ids']
        scale_x = width / 640.0
        scale_y = height / 640.0
        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            x1 = float(x1 * scale_x)
            y1 = float(y1 * scale_y)
            x2 = float(x2 * scale_x)
            y2 = float(y2 * scale_y)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': coreml_class_names[int(cls)] if coreml_class_names else str(cls)
            })
    return jsonify({'tags': tags, 'objects': detections, 'frame_width': width, 'frame_height': height})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 