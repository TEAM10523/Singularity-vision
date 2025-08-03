from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import coremltools as ct
from PIL import Image

from .config import config, get_cam_fps, get_confidence_threshold, get_nms_threshold
from .camera import get_latest_frame

# ---------------------------------------------------------------------------
# Module-level shared state
# ---------------------------------------------------------------------------

yolo_lock = threading.Lock()
latest_yolo_result: Optional[object] = None  # torch Results instance OR dict when using CoreML
actual_fps: int = 0

# Inference thresholds guarded by their own lock so they can be updated at
# runtime via the REST API.
threshold_lock = threading.Lock()
CONFIDENCE_THRESHOLD: float = get_confidence_threshold()
NMS_THRESHOLD: float = get_nms_threshold()

# Backend + model bookkeeping
current_backend: str = config["inference"]["backend"]
current_torch_model: str = config["models"]["torch"]["default"]
current_coreml_model: str = config["models"]["coreml"]["default"]

yolo_model: Optional[YOLO] = None
coreml_model: Optional[ct.models.MLModel] = None
coreml_class_names = None
coreml_input_key = None
coreml_output_key = None


def _init_torch_model() -> None:
    global yolo_model, yolo_device

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        yolo_device = torch.device("mps")
    else:
        yolo_device = torch.device("cpu")

    yolo_model = YOLO(f"models/torch/{current_torch_model}")
    yolo_model.to(yolo_device)


def _load_coreml_model():
    """Lazy-load the CoreML model only when required."""
    global coreml_model, coreml_class_names, coreml_input_key, coreml_output_key

    if coreml_model is not None:
        return

    model_path = f"models/coreml/{current_coreml_model}"
    coreml_model = ct.models.MLModel(model_path)

    spec = coreml_model.get_spec()
    coreml_input_key = spec.description.input[0].name
    coreml_output_key = spec.description.output[0].name

    # Extract YOLO class names (the exporter stores them as a string repr)
    import ast

    user_meta = spec.description.metadata.userDefined
    coreml_class_names = ast.literal_eval(user_meta.get("names", "{}")) if user_meta else None


def _preprocess_for_coreml(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    return Image.fromarray(img)


def _postprocess_coreml_output(result):
    output = result[coreml_output_key]
    arr = np.array(output)

    # Possible shapes: (1, 84, N) or (1, N, 84)
    if arr.shape[1] == 84:
        arr = arr[0].T  # (N, 84)
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

    # Convert cxcywh -> xyxy
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    indices = cv2.dnn.NMSBoxes(xyxy_boxes.tolist(), confidences.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(indices):
        indices = indices.flatten()
        xyxy_boxes = xyxy_boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]

    return {
        "boxes": xyxy_boxes,
        "confidences": confidences,
        "class_ids": class_ids,
    }


# ---------------------------------------------------------------------------
# Background inference thread
# ---------------------------------------------------------------------------

def _yolo_inferencer():
    global latest_yolo_result, actual_fps

    last = time.time()
    count = 0
    interval = 1.0 / float(get_cam_fps())

    while True:
        frame = get_latest_frame()
        if frame is not None:
            if current_backend == "torch":
                if yolo_model is None:
                    _init_torch_model()
                try:
                    results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD, verbose=False)[0]
                    with yolo_lock:
                        latest_yolo_result = results
                except Exception as e:
                    print(f"Torch inference error: {e}")
                    with yolo_lock:
                        latest_yolo_result = None
            else:  # coreml backend
                try:
                    _load_coreml_model()
                    input_data = _preprocess_for_coreml(frame)
                    result = coreml_model.predict({coreml_input_key: input_data})
                    processed = _postprocess_coreml_output(result)
                    with yolo_lock:
                        latest_yolo_result = processed
                except Exception as e:
                    print(f"CoreML inference error: {e}")
                    with yolo_lock:
                        latest_yolo_result = None

        # FPS bookkeeping
        count += 1
        if time.time() - last > 1:
            actual_fps = count
            count = 0
            last = time.time()

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def start_yolo_thread() -> threading.Thread:
    t = threading.Thread(target=_yolo_inferencer, daemon=True, name="YOLOThread")
    t.start()
    return t


def get_latest_yolo_result():
    with yolo_lock:
        return latest_yolo_result


def get_actual_fps() -> int:
    return actual_fps


# ---------------------------------------------------------------------------
# Runtime configuration update helpers (called from the API layer)
# ---------------------------------------------------------------------------

def update_thresholds(confidence: float, nms: float):
    global CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    with threshold_lock:
        CONFIDENCE_THRESHOLD = confidence
        NMS_THRESHOLD = nms


def switch_backend(new_backend: str):
    global current_backend, yolo_model, coreml_model, latest_yolo_result
    if new_backend not in {"torch", "coreml"}:
        raise ValueError("Invalid backend")

    current_backend = new_backend
    latest_yolo_result = None  # reset cache so consumers don't get mixed types

    if current_backend == "torch":
        coreml_model = None
        _init_torch_model()
    else:
        yolo_model = None  # freed when GC'd


def switch_model(backend: str, model_file: str):
    global current_torch_model, current_coreml_model, yolo_model, coreml_model

    if backend == "torch":
        current_torch_model = model_file
        yolo_model = None  # reload lazily on next inference
        if current_backend == "torch":
            _init_torch_model()
    elif backend == "coreml":
        current_coreml_model = model_file
        coreml_model = None  # reload lazily
    else:
        raise ValueError("Unknown backend")


__all__ = [
    "start_yolo_thread",
    "get_latest_yolo_result",
    "get_actual_fps",
    "update_thresholds",
    "switch_backend",
    "switch_model",
] 