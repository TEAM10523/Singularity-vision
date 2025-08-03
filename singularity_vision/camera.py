from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import cv2
import platform
import subprocess
import json as _json

from .config import config, get_cam_fps

# ---------------------------------------------------------------------------
# Module-level state (shared across the application)
# ---------------------------------------------------------------------------

# NOTE: These objects are intended to be *read* from other modules. Try to
# modify them only inside this module to minimise races.

frame_lock = threading.Lock()
latest_frame: Optional["cv2.Mat"] = None

# We expose this flag so that a graceful shutdown can be initiated from the
# outside (e.g. Flask signal handler or aatexit cleanup).
camera_thread_running = True

# Keep a reference to the cv2.VideoCapture object so we can release it later.
camera_cap: Optional[cv2.VideoCapture] = None
selected_device_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Helper – find an operational camera from a set of indices (fallback scan)
# ---------------------------------------------------------------------------


def _find_working_camera(
    fallback_range: Sequence[int],
    width: int,
    height: int,
) -> Optional[int]:
    """Return first device in *fallback_range* that delivers a valid frame."""

    for device_id in fallback_range:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            cap.release()
            continue

        # Try to set resolution to help filter out virtual devices
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ret, _ = cap.read()
        if ret:
            cap.release()
            return device_id

        cap.release()

    return None


# ---------------------------------------------------------------------------
# macOS-specific helper: map camera name substring to AVFoundation index
# ---------------------------------------------------------------------------


def _device_index_by_name(name_substr: str) -> Optional[int]:
    """Return AVFoundation device index whose descriptive name contains
    *name_substr* (case-insensitive). Only implemented on macOS; returns None
    otherwise or if not found."""

    if platform.system() != "Darwin":
        return None

    try:
        out = subprocess.check_output([
            "system_profiler",
            "-json",
            "SPCameraDataType",
        ], text=True)
        data = _json.loads(out)
        cameras = data.get("SPCameraDataType", [{}])[0].get("_items", [])
        # AVFoundation orders devices as they appear in this list
        for idx, cam in enumerate(cameras):
            if name_substr.lower() in cam.get("_name", "").lower():
                return idx
    except Exception as exc:
        print(f"Warning: unable to map camera identifier via system_profiler: {exc}")

    return None


# ---------------------------------------------------------------------------
# Background thread implementation
# ---------------------------------------------------------------------------

def _camera_reader() -> None:
    """Continuously grab frames from the configured camera and store the
    most recent frame in *latest_frame*.
    """
    global latest_frame, camera_cap, camera_thread_running

    cam_cfg = config["camera"]

    identifier = cam_cfg.get("identifier")

    if not identifier:
        raise RuntimeError("camera.identifier missing in config.json")

    dev_id = _device_index_by_name(identifier)

    if dev_id is None:
        # Fallback scan (0-5) if name mapping failed
        fallback_ids = cam_cfg.get("fallback_ids", list(range(5)))
        dev_id = _find_working_camera(fallback_ids, cam_cfg["width"], cam_cfg["height"])

    if dev_id is None:
        raise RuntimeError("No working camera found")

    global selected_device_id
    selected_device_id = dev_id

    cap = cv2.VideoCapture(dev_id)
    camera_cap = cap  # keep global ref so we can release later

    # Apply resolution settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    # Target frame interval derived from the configured FPS
    interval = 1.0 / float(get_cam_fps())

    while camera_thread_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame
        time.sleep(interval)

    # Cleanup when the thread is asked to stop
    cap.release()
    camera_cap = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def start_camera_thread() -> threading.Thread:
    """Spawn and start the daemon camera reader thread. Returns the thread
    object so the caller can *join* it if desired."""
    t = threading.Thread(target=_camera_reader, daemon=True, name="CameraThread")
    t.start()
    return t


def get_latest_frame(copy: bool = False):
    """Return the most recent camera frame. If *copy* is True (default) we
    return a shallow copy so the caller can safely manipulate it without
    interfering with other threads."""
    with frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy() if copy else latest_frame


def cleanup_camera() -> None:
    """Stop the camera thread and release resources."""
    global camera_thread_running, camera_cap
    camera_thread_running = False
    if camera_cap is not None:
        camera_cap.release()
        camera_cap = None


__all__ = [
    "start_camera_thread",
    "get_latest_frame",
    "cleanup_camera",
    "frame_lock",
] 