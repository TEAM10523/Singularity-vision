from __future__ import annotations

"""NTCore publisher
====================
Publishes Vision outputs (YOLO detections + AprilTag poses) to the FRC
roboRIO via NetworkTables so that robot code can consume them.
"""

import json
import os
import threading
import time
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# NT import shim: prefer fast native `ntcore` binding, fall back to pure-python
# `pynetworktables` so the app can run even if compiled wheels are unavailable.
# ---------------------------------------------------------------------------


def _nt_instance():
    """Return an initialised NetworkTablesInstance (client mode)."""

    try:
        from ntcore import NetworkTableInstance  # type: ignore

        return NetworkTableInstance.getDefault()
    except Exception:
        # Fallback to pure-python implementation
        try:
            from networktables import NetworkTablesInstance  # type: ignore

            return NetworkTablesInstance.getDefault()
        except Exception as exc:
            raise ImportError(
                "Neither 'ntcore' nor 'pynetworktables' are available; install via\n"
                "    pip install ntcore  # for compiled native binding\n"
                "or\n"
                "    pip install pynetworktables  # pure-python fallback"
            ) from exc


from .config import config
from .yolo import get_latest_yolo_result, current_backend, coreml_class_names
from .apriltag_module import get_latest_apriltag_poses

# ---------------------------------------------------------------------------
# NetworkTables initialisation helpers
# ---------------------------------------------------------------------------


def _init_ntcore():
    """Initialise NetworkTables client and return the (instance, table) pair."""

    inst = _nt_instance()

    # Determine which API variant we have (ntcore vs pynetworktables)
    is_ntcore = hasattr(inst, "startClient4")

    nt_cfg = config.get("ntcore", {})
    server_addr: str | None = nt_cfg.get("server")
    team_num = nt_cfg.get("team")

    # ------------------------------
    # ntcore (compiled) branch
    # ------------------------------
    if is_ntcore:
        inst.startClient4("SingularityVision")

        if server_addr:
            inst.setServer(server_addr)
        elif team_num is not None:
            inst.setServerTeam(int(team_num))
        else:
            fallback = os.getenv("FRC_ROBORIO_IP", "10.0.0.2")
            inst.setServer(fallback)

    # ------------------------------
    # pynetworktables (pure-python) branch
    # ------------------------------
    else:
        # Network identity (if supported)
        if hasattr(inst, "setNetworkIdentity"):
            inst.setNetworkIdentity("SingularityVision")

        # Select connection strategy
        if server_addr:
            inst.startClient(server_addr)
        elif team_num is not None and hasattr(inst, "startClientTeam"):
            inst.startClientTeam(int(team_num))
        else:
            fallback = os.getenv("FRC_ROBORIO_IP", "10.0.0.2")
            inst.startClient(fallback)

    table = inst.getTable("vision")
    print("NTCore client started, publishing to table '/vision'")
    return inst, table


# ---------------------------------------------------------------------------
# Helpers to transform inference results to serialisable formats
# ---------------------------------------------------------------------------


def _build_yolo_detections() -> List[Dict[str, Any]]:
    """Return latest YOLO detections as a list of simple dicts."""

    detections: List[Dict[str, Any]] = []
    yolo_result = get_latest_yolo_result()

    if yolo_result is None:
        return detections

    # Torch backend
    if current_backend == "torch" and hasattr(yolo_result, "boxes"):
        boxes = yolo_result.boxes
        if boxes is not None:
            for box in boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "bbox": xyxy,  # [x1, y1, x2, y2]
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "class_name": yolo_result.names[int(box.cls[0])],
                    }
                )

    # CoreML backend
    elif current_backend == "coreml" and isinstance(yolo_result, dict):
        boxes = yolo_result["boxes"]
        confidences = yolo_result["confidences"]
        class_ids = yolo_result["class_ids"]
        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "class_name": coreml_class_names[int(cls)] if coreml_class_names else str(cls),
                }
            )

    return detections


# ---------------------------------------------------------------------------
# Background publisher thread
# ---------------------------------------------------------------------------


def _ntcore_publisher():
    """Continuously push vision results to NetworkTables."""

    inst, table = _init_ntcore()

    # Sub-entries to reuse each iteration (avoids reallocation cost)
    detections_entry = table.getEntry("detections")
    apriltag_entry = table.getEntry("apriltag_poses")

    publish_hz: float = config.get("ntcore", {}).get("publish_hz", 20.0)
    interval = 1.0 / publish_hz if publish_hz > 0 else 0.05

    while True:
        # Build payloads
        detections = _build_yolo_detections()
        apriltag_poses = get_latest_apriltag_poses()

        # Serialise to JSON strings – easier to consume on robot side
        detections_entry.setString(json.dumps(detections))
        apriltag_entry.setString(json.dumps(apriltag_poses if apriltag_poses else {}))

        inst.flush()
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_ntcore_thread() -> threading.Thread:
    """Start the NTCore publisher in a daemon thread and return it."""

    t = threading.Thread(target=_ntcore_publisher, daemon=True, name="NTCorePublisher")
    t.start()
    return t


__all__ = ["start_ntcore_thread"] 