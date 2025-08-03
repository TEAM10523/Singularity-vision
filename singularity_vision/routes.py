from __future__ import annotations

import time
from typing import Any, Dict, List

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request

from .camera import frame_lock, get_latest_frame
from .yolo import (
    get_latest_yolo_result,
    switch_backend,
    switch_model,
    update_thresholds,
    get_actual_fps,
)
from .apriltag_module import (
    get_latest_apriltag_result,
    get_latest_apriltag_poses,
    apriltag_enabled,
)
from .config import config, get_confidence_threshold, get_nms_threshold

# Flask blueprint to keep the main app factory tidy
api_bp = Blueprint("api", __name__)

# ---------------------------------------------------------------------------
# Dashboard + video feed
# ---------------------------------------------------------------------------


@api_bp.route("/")
def index():
    return render_template("dashboard.html")


@api_bp.route("/video_feed")
def video_feed():
    from .yolo import current_backend  # local import to avoid circular
    from .yolo import coreml_class_names

    FPS = config["inference"]["fps"]

    def generate():
        last_time = time.time()
        frame_interval = 1.0 / float(FPS)

        while True:
            frame = get_latest_frame(copy=False)

            if frame is None:
                # Yield a blank frame to keep the MJPEG stream alive until we get real frames
                blank = np.zeros((config["camera"]["height"], config["camera"]["width"], 3), dtype=np.uint8)
                ret, buffer = cv2.imencode(".jpg", blank)
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                time.sleep(0.1)
                continue

            # We have a real frame
            yolo_result = get_latest_yolo_result()
            apriltag_result = get_latest_apriltag_result()
            apriltag_poses = get_latest_apriltag_poses()

            frame_with_boxes = frame
            if yolo_result is not None:
                if current_backend == "torch" and hasattr(yolo_result, "plot"):
                    frame_with_boxes = yolo_result.plot()
                elif current_backend == "coreml":
                    boxes = yolo_result["boxes"]
                    class_ids = yolo_result["class_ids"]
                    confidences = yolo_result["confidences"]
                    frame_with_boxes = _draw_coreml_boxes(
                        frame_with_boxes.copy(), boxes, class_ids, confidences, coreml_class_names
                    )

            # Draw AprilTag annotations
            if apriltag_result is not None and apriltag_result.get("ids") is not None:
                frame_with_boxes = _draw_apriltag_annotations(
                    frame_with_boxes, apriltag_result, apriltag_poses
                )

            ret, buffer = cv2.imencode(".jpg", frame_with_boxes)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(frame_interval - (now - last_time))
            last_time = time.time()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------


@api_bp.route("/api/detections")
def detections_api():
    yolo_result = get_latest_yolo_result()
    apriltag_result = get_latest_apriltag_result()

    detections: List[Dict[str, Any]] = []
    tags: List[Dict[str, Any]] = []

    from .yolo import current_backend, coreml_class_names
    from .camera import latest_frame

    if yolo_result is not None:
        if current_backend == "torch" and hasattr(yolo_result, "boxes"):
            boxes = yolo_result.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "bbox": xyxy,
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                            "class_name": yolo_result.names[int(box.cls[0])],
                        }
                    )
        elif current_backend == "coreml" and isinstance(yolo_result, dict):
            boxes = yolo_result["boxes"]
            confidences = yolo_result["confidences"]
            class_ids = yolo_result["class_ids"]
            height, width = latest_frame.shape[:2] if latest_frame is not None else (480, 640)
            scale_x = width / 640.0
            scale_y = height / 640.0
            for box, conf, cls in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                detections.append(
                    {
                        "bbox": [float(x1 * scale_x), float(y1 * scale_y), float(x2 * scale_x), float(y2 * scale_y)],
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "class_name": coreml_class_names[int(cls)] if coreml_class_names else str(cls),
                    }
                )

    if apriltag_result is not None:
        corners = apriltag_result["corners"]
        ids = apriltag_result["ids"]
        if ids is not None:
            for corner, tag_id in zip(corners, ids):
                center = corner[0].mean(axis=0)
                tags.append(
                    {
                        "id": int(tag_id[0]),
                        "family": "aruco",  # kept as-is from original design
                        "center": center.tolist(),
                        "corners": corner[0].tolist(),
                    }
                )

    return jsonify({"objects": detections, "tags": tags})


@api_bp.route("/api/apriltag_poses")
def apriltag_poses_api():
    poses = get_latest_apriltag_poses()

    if poses is None:
        return jsonify(
            {
                "single_tag_poses": [],
                "multi_tag_pose": None,
                "message": "No AprilTag poses available",
            }
        )

    formatted_single = _format_single_tag_poses(poses.get("single_tag", {}))
    multi_tag_pose = _format_multi_tag_pose(poses.get("multi_tag", {}))

    return jsonify(
        {
            "single_tag_poses": formatted_single,
            "multi_tag_pose": multi_tag_pose,
            "timestamp": time.time(),
        }
    )


@api_bp.route("/api/apriltag_config")
def apriltag_config_api():
    apriltag_cfg = config.get("apriltag", {})
    return jsonify(
        {
            "enabled": apriltag_cfg.get("enabled", False),
            "tag_size": apriltag_cfg.get("tag_size", 0.0),
            "camera_pose": apriltag_cfg.get("camera_pose", {}),
            "tag_layout": apriltag_cfg.get("tag_layout", []),
            "pose_estimation_available": apriltag_enabled(),
        }
    )


@api_bp.route("/api/update_thresholds", methods=["POST"])
def update_thresholds_api():
    data = request.get_json(force=True)
    conf = float(data.get("confidence_threshold", get_confidence_threshold()))
    nms = float(data.get("nms_threshold", get_nms_threshold()))
    update_thresholds(conf, nms)
    return jsonify({"success": True, "confidence_threshold": conf, "nms_threshold": nms})


@api_bp.route("/api/config")
def get_config_api():
    from .yolo import current_backend, current_torch_model, current_coreml_model

    return jsonify(
        {
            "current_backend": current_backend,
            "current_torch_model": current_torch_model,
            "current_coreml_model": current_coreml_model,
            "fps": config["inference"]["fps"],
            "confidence_threshold": get_confidence_threshold(),
            "nms_threshold": get_nms_threshold(),
            "actual_fps": get_actual_fps(),
            "available_models": config["models"],
        }
    )


@api_bp.route("/api/switch_backend", methods=["POST"])
def switch_backend_api():
    data = request.get_json(force=True)
    backend = data.get("backend")
    try:
        switch_backend(backend)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"success": True, "backend": backend})


@api_bp.route("/api/switch_model", methods=["POST"])
def switch_model_api():
    data = request.get_json(force=True)
    backend = data.get("backend")
    model_file = data.get("model")
    switch_model(backend, model_file)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Helper drawing / formatting functions (kept internal)
# ---------------------------------------------------------------------------


def _draw_coreml_boxes(frame, boxes, class_ids, confidences, class_names):
    h, w = frame.shape[:2]
    scale_x = w / 640.0
    scale_y = h / 640.0

    num_classes = len(class_names) if class_names else 80
    colors = [
        tuple(
            int(x * 255)
            for x in cv2.cvtColor(
                np.uint8([[[int(i * 255 / num_classes), 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0]
        )
        for i in range(num_classes)
    ]
    thickness = max(2, int(0.002 * (w + h) / 2))
    font_scale = max(0.5, 0.001 * (w + h) / 2)

    for box, cls, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        color = colors[int(cls) % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_names[int(cls)] if class_names else str(cls)} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(2, thickness))
        y0 = max(0, y1 - th - baseline)
        y1_clip = min(h, y1)
        x0 = max(0, x1)
        x1_clip = min(w, x1 + tw)
        if y0 < y1_clip and x0 < x1_clip:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1_clip, y1_clip), (0, 0, 0), -1)
            alpha = 0.7
            frame[y0:y1_clip, x0:x1_clip] = cv2.addWeighted(
                overlay[y0:y1_clip, x0:x1_clip], alpha, frame[y0:y1_clip, x0:x1_clip], 1 - alpha, 0
            )
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            max(2, thickness),
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(2, thickness - 1),
            lineType=cv2.LINE_AA,
        )

    return frame


def _draw_apriltag_annotations(frame, apriltag_result, apriltag_poses):
    corners = apriltag_result["corners"]
    ids = apriltag_result["ids"]

    for corner, tag_id in zip(corners, ids):
        pts = corner[0].astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID: {tag_id[0]}", (center[0] - 20, center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if apriltag_poses is not None and apriltag_poses.get("single_tag"):
            single_tags = apriltag_poses["single_tag"]
            if "tag_ids" in single_tags and "field_to_robot_poses" in single_tags:
                tag_ids = single_tags["tag_ids"]
                field_poses = single_tags["field_to_robot_poses"]
                for pose_tag_id, pose in zip(tag_ids, field_poses):
                    if pose_tag_id == tag_id[0] and pose[0] != -9999:
                        pose_text = f"Robot: ({pose[0]:.2f}, {pose[1]:.2f})"
                        cv2.putText(frame, pose_text, (center[0] - 40, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        break

    if apriltag_poses is not None and apriltag_poses.get("multi_tag"):
        multi_tag = apriltag_poses["multi_tag"]
        if multi_tag["pose"][0] != -9999:
            pose_text = f"Robot (Multi): ({multi_tag['pose'][0]:.2f}, {multi_tag['pose'][1]:.2f})"
            cv2.putText(frame, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame


def _format_single_tag_poses(single_tag_results):
    formatted = []
    if not single_tag_results:
        return formatted

    tag_ids = single_tag_results.get("tag_ids", [])
    camera_to_tag = single_tag_results.get("camera_to_tag_poses", [])
    robot_to_tag = single_tag_results.get("robot_to_tag_poses", [])
    field_to_robot = single_tag_results.get("field_to_robot_poses", [])
    errors = single_tag_results.get("errors", [])

    for idx, tag_id in enumerate(tag_ids):
        if tag_id == -9999:
            continue
        formatted.append(
            {
                "tag_id": tag_id,
                "camera_to_tag_pose": _pose_list_to_dict(camera_to_tag[idx]),
                "robot_to_tag_pose": _pose_list_to_dict(robot_to_tag[idx]),
                "field_to_robot_pose": _pose_list_to_dict(field_to_robot[idx]) if field_to_robot[idx][0] != -9999 else None,
                "estimation_error": errors[idx][0] if errors[idx][0] != -9999 else None,
            }
        )
    return formatted


def _pose_list_to_dict(lst):
    return {"x": lst[0], "y": lst[1], "z": lst[2], "roll": lst[3], "pitch": lst[4], "yaw": lst[5]}


def _format_multi_tag_pose(multi_tag):
    if multi_tag.get("pose", [ -9999])[0] == -9999:
        return None
    return {
        "field_to_robot_pose": _pose_list_to_dict(multi_tag["pose"]),
        "estimation_error": multi_tag["error"] if multi_tag["error"] != -9999 else None,
    } 