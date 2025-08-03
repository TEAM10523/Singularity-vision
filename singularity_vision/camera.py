from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

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


# ---------------------------------------------------------------------------
# Background thread implementation
# ---------------------------------------------------------------------------

def _camera_reader() -> None:
    """Continuously grab frames from the configured camera and store the
    most recent frame in *latest_frame*.
    """
    global latest_frame, camera_cap, camera_thread_running

    cam_cfg = config["camera"]
    cap = cv2.VideoCapture(cam_cfg["device_id"])
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