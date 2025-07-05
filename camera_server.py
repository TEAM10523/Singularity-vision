import cv2
import numpy as np
import threading
import time
import json
import os
from flask import Flask, Response, jsonify, request, render_template
from ultralytics import YOLO
import torch
from PIL import Image
import coremltools as ct

app = Flask(__name__)

# Load configuration
def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

config = load_config()

# Global variables
latest_frame = None
latest_yolo_result = None
latest_apriltag_result = None
frame_lock = threading.Lock()
yolo_lock = threading.Lock()
apriltag_lock = threading.Lock()
actual_fps = 0

# Current settings (can be changed via API)
current_backend = config['inference']['backend']
current_torch_model = config['models']['torch']['default']
current_coreml_model = config['models']['coreml']['default']
FPS = config['inference']['fps']
CONFIDENCE_THRESHOLD = config['inference']['confidence_threshold']
NMS_THRESHOLD = config['inference']['nms_threshold']
threshold_lock = threading.Lock()

# Model instances
yolo_model = None
coreml_model = None
coreml_class_names = None
coreml_input_key = None
coreml_output_key = None

# Device determination and model loading are only performed when using torch as the inference backend
if current_backend == 'torch':
    # Detect Metal (MPS) support
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        yolo_device = torch.device('mps')
        print('Using Apple Silicon Metal (MPS) for YOLOv8 inference.')
    else:
        yolo_device = torch.device('cpu')
        print('Using CPU for YOLOv8 inference.')
    # Global YOLO model (load only once)
    yolo_model = YOLO(f"models/torch/{current_torch_model}")
    yolo_model.to(yolo_device)
    
else:
    yolo_model = None
    print('Using CoreML for YOLOv8 inference.')

# CoreML model loading (lazy loading)
def load_coreml_model():
    global coreml_model, coreml_class_names, coreml_input_key, coreml_output_key
    if coreml_model is None:
        model_path = f"models/coreml/{current_coreml_model}"
        coreml_model = ct.models.MLModel(model_path)
        # Get input key
        coreml_input_key = coreml_model.get_spec().description.input[0].name
        # Get output key (use the first output)
        coreml_output_key = coreml_model.get_spec().description.output[0].name
        # Class names
        import ast
        user_meta = coreml_model.get_spec().description.metadata.userDefined
        if 'names' in user_meta:
            coreml_class_names = ast.literal_eval(user_meta['names'])
        else:
            coreml_class_names = None

# CoreML Preprocessing
def preprocess_for_coreml(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    pil_img = Image.fromarray(img)
    return pil_img

# CoreML postprocessing
def postprocess_coreml_output(result):
    # Use dynamic output key
    output = result[coreml_output_key]
    arr = np.array(output)
    # Try to adapt to (1, 84, N) or (1, N, 84)
    if arr.shape[1] == 84:
        arr = arr[0]  # (84, N)
        arr = arr.T   # (N, 84)
    elif arr.shape[2] == 84:
        arr = arr[0]  # (N, 84)
    else:
        raise ValueError(f"Unexpected CoreML output shape: {arr.shape}")
    boxes = arr[:, :4]
    scores = arr[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    keep = confidences > CONFIDENCE_THRESHOLD
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    indices = cv2.dnn.NMSBoxes(xyxy_boxes.tolist(), confidences.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(indices) > 0:
        indices = indices.flatten()
        xyxy_boxes = xyxy_boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]
    return {
        'boxes': xyxy_boxes,
        'confidences': confidences,
        'class_ids': class_ids
    }

# Camera reader thread
def camera_reader():
    global latest_frame
    cap = cv2.VideoCapture(config['camera']['device_id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    
    interval = 1.0 / FPS
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame
        time.sleep(interval)
    
    cap.release()

# YOLO inference thread
def yolo_inferencer():
    global latest_yolo_result, actual_fps
    last = time.time()
    count = 0
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            if current_backend == 'torch' and yolo_model is not None and hasattr(yolo_model, '__call__'):
                try:
                    results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD, verbose=False)[0]
                    with yolo_lock:
                        latest_yolo_result = results
                except Exception as e:
                    print(f"Torch inference error: {e}")
                    with yolo_lock:
                        latest_yolo_result = None
            elif current_backend == 'coreml':
                try:
                    load_coreml_model()
                    input_data = preprocess_for_coreml(frame)
                    result = coreml_model.predict({coreml_input_key: input_data})
                    yolo_result = postprocess_coreml_output(result)
                    with yolo_lock:
                        latest_yolo_result = yolo_result
                except Exception as e:
                    print(f"CoreML inference error: {e}")
                    with yolo_lock:
                        latest_yolo_result = None
        count += 1
        if time.time() - last > 1:
            actual_fps = count
            count = 0
            last = time.time()
        time.sleep(1.0 / FPS)

# AprilTag inference thread
def apriltag_inferencer():
    global latest_apriltag_result
    detector = cv2.aruco.ArucoDetector()
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)
            with apriltag_lock:
                latest_apriltag_result = {'corners': corners, 'ids': ids}
        time.sleep(1.0 / FPS)

# Start threads
threading.Thread(target=camera_reader, daemon=True).start()
threading.Thread(target=yolo_inferencer, daemon=True).start()
threading.Thread(target=apriltag_inferencer, daemon=True).start()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        last_time = time.time()
        frame_interval = 1.0 / FPS
        
        while True:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            
            if frame is not None:
                with yolo_lock:
                    yolo_result = latest_yolo_result
                
                if yolo_result is not None:
                    if current_backend == 'torch' and hasattr(yolo_result, 'plot'):
                        frame_with_boxes = yolo_result.plot()
                    elif current_backend == 'coreml':
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
                            color = colors[int(cls) % len(colors)]
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
                    else:
                        frame_with_boxes = frame
                else:
                    frame_with_boxes = frame
                
                ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # FPS control
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(frame_interval - (now - last_time))
            last_time = time.time()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def detections_api():
    with yolo_lock:
        yolo_result = latest_yolo_result
    with apriltag_lock:
        apriltag_result = latest_apriltag_result
    
    detections = []
    tags = []
    
    if yolo_result is not None:
        if current_backend == 'torch' and hasattr(yolo_result, 'boxes'):
            # Parse torch detections
            boxes = yolo_result.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        'bbox': box.xyxy[0].tolist(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': yolo_result.names[int(box.cls[0])]
                    })
        elif current_backend == 'coreml' and isinstance(yolo_result, dict):
            # Parse CoreML detections
            boxes = yolo_result['boxes']
            confidences = yolo_result['confidences']
            class_ids = yolo_result['class_ids']
            height, width = latest_frame.shape[:2] if latest_frame is not None else (480, 640)
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
    
    if apriltag_result is not None:
        corners = apriltag_result['corners']
        ids = apriltag_result['ids']
        if ids is not None:
            for i, (corner, tag_id) in enumerate(zip(corners, ids)):
                center = corner[0].mean(axis=0)
                tags.append({
                    'id': int(tag_id[0]),
                    'family': 'aruco',
                    'center': center.tolist(),
                    'corners': corner[0].tolist()
                })
    
    return jsonify({
        'objects': detections,
        'tags': tags
    })

@app.route('/api/update_thresholds', methods=['POST'])
def update_thresholds():
    global CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    data = request.get_json()
    conf = float(data.get('confidence_threshold', CONFIDENCE_THRESHOLD))
    nms = float(data.get('nms_threshold', NMS_THRESHOLD))
    with threshold_lock:
        CONFIDENCE_THRESHOLD = conf
        NMS_THRESHOLD = nms
    return jsonify({'success': True, 'confidence_threshold': conf, 'nms_threshold': nms})

@app.route('/api/config')
def get_config():
    # Return current in-memory values for thresholds
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    return jsonify({
        'current_backend': current_backend,
        'current_torch_model': current_torch_model,
        'current_coreml_model': current_coreml_model,
        'fps': cfg['inference']['fps'],
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'nms_threshold': NMS_THRESHOLD,
        'actual_fps': actual_fps,
        'available_models': cfg['models']
    })

@app.route('/api/switch_backend', methods=['POST'])
def switch_backend():
    global current_backend, yolo_model, yolo_device, latest_yolo_result
    data = request.get_json()
    new_backend = data.get('backend')
    
    if new_backend not in ['torch', 'coreml']:
        return jsonify({'error': 'Invalid backend'}), 400
    
    current_backend = new_backend
    latest_yolo_result = None  # Clear previous result to avoid type mismatch
    # Reload model for new backend
    if current_backend == 'torch':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            yolo_device = torch.device('mps')
        else:
            yolo_device = torch.device('cpu')
        model = YOLO(f"models/torch/{current_torch_model}")
        model.to(yolo_device)
        yolo_model = model
        print(f'Switched to torch backend with model: {current_torch_model}')
    else:
        yolo_model = None
        print(f'Switched to coreml backend with model: {current_coreml_model}')
    
    return jsonify({'success': True, 'backend': current_backend})

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    global current_torch_model, current_coreml_model, yolo_model, yolo_device
    data = request.get_json()
    backend = data.get('backend')
    model_file = data.get('model')
    
    if backend == 'torch':
        current_torch_model = model_file
        # Temporarily set yolo_model to None to prevent inference thread from using old model
        yolo_model = None
        # Always load the torch model when switching torch models, regardless of current backend
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            yolo_device = torch.device('mps')
        else:
            yolo_device = torch.device('cpu')
        try:
            yolo_model = YOLO(f"models/torch/{current_torch_model}")
            yolo_model.to(yolo_device)
            print(f'Switched torch model to: {current_torch_model}')
        except Exception as e:
            print(f"Error loading torch model: {e}")
            yolo_model = None
    elif backend == 'coreml':
        current_coreml_model = model_file
        global coreml_model, coreml_class_names
        coreml_model = None  # Force reload
        print(f'Switched coreml model to: {current_coreml_model}')
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False) 