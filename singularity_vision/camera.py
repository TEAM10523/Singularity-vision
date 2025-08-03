from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

# add back platform and subprocess
import platform
import subprocess
import json as _json

# Attempt to import AVFoundation via PyObjC for reliable device name mapping
try:
    import AVFoundation  # type: ignore
    _HAS_AVF = True
except ImportError:
    _HAS_AVF = False

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
# Helper: map macOS camera names to indices via system_profiler or imagesnap.
# ---------------------------------------------------------------------------


def _device_index_by_name(name_substr: str) -> Optional[int]:
    """Return index of camera whose name contains *name_substr* (case-insensitive).
    Supported on macOS using system_profiler/imagesnap. None if not found."""

    # macOS specific
    if platform.system() != "Darwin":
        return None

    # 1. Use ffmpeg avfoundation list (indices match OpenCV directly)
    try:
        proc = subprocess.run([
            "ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=False
        )
        for line in proc.stderr.splitlines():
            if "AVFoundation video devices" in line:
                continue
            if "input device" in line and "] [" in line:
                # format: ... [N] Name
                parts = line.split("[", 2)
                if len(parts) >= 3:
                    idx_str = parts[2].split("]",1)[0]
                    name = parts[2].split("]",1)[1].strip()
                    try:
                        idx_num = int(idx_str)
                    except ValueError:
                        continue
                    if name_substr.lower() in name.lower():
                        return idx_num
    except FileNotFoundError:
        pass

    # 2. Fallback to PyObjC DiscoverySession
    if not _HAS_AVF:
        return None

    try:
        # Use the non-deprecated DiscoverySession API so the order matches AVFoundation / OpenCV.
        session = AVFoundation.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            [
                AVFoundation.AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVFoundation.AVCaptureDeviceTypeExternalUnknown,
            ],
            AVFoundation.AVMediaTypeVideo,
            AVFoundation.AVCaptureDevicePositionUnspecified,
        )
        devices = list(session.devices())
        total = len(devices)

        # First pass: exact (case-insensitive) match
        for idx, dev in enumerate(devices):
            name = str(dev.localizedName())
            if name.lower() == name_substr.lower():
                return total - 1 - idx  # reverse to match OpenCV order

        # Second pass: substring match
        name_sub_lc = name_substr.lower()
        for idx, dev in enumerate(devices):
            if name_sub_lc in str(dev.localizedName()).lower():
                return total - 1 - idx
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Camera reader thread
# ---------------------------------------------------------------------------

def _camera_reader() -> None:
    """Continuously grab frames from the configured camera and store the
    most recent frame in *latest_frame*.
    """
    global latest_frame, camera_cap, camera_thread_running

    cam_cfg = config["camera"]

    dev_id = cam_cfg.get("device_id", 0)

    # Simple validation: attempt to open once to ensure device exists
    test_cap = cv2.VideoCapture(dev_id)
    if not test_cap.isOpened():
        raise RuntimeError(f"Unable to open camera with device_id={dev_id}. Check config.json and available cameras.")
    test_cap.release()

    print(f"Selected camera device {dev_id} as configured in config.json")

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