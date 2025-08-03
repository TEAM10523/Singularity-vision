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
# New helpers: resolve camera index via uniqueID / name / fallback to integer
# ---------------------------------------------------------------------------


def _device_index_by_uid(uid: str) -> Optional[int]:
    """Return OpenCV index for the camera whose AVFoundation *uniqueID* equals *uid*.

    This follows the same ordering logic as `cv2.CAP_AVFOUNDATION`—devices are
    sorted by their uniqueID strings so that the resulting index lines up with
    what OpenCV expects.
    """
    if not (_HAS_AVF and platform.system() == "Darwin"):
        return None
    try:
        devices = sorted(
            AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo),
            key=lambda d: str(d.uniqueID()),
        )
        for idx, dev in enumerate(devices):
            if str(dev.uniqueID()) == str(uid):
                return idx
    except Exception:
        pass
    return None


def _resolve_device_index(cam_cfg: dict) -> int:
    """Determine which camera index to open based on *cam_cfg*.

    Priority order:
    1. *unique_id*  – most stable on macOS (requires AVFoundation / PyObjC)
    2. *name*       – substring or exact match of camera name
    3. *device_id*  – plain integer fallback.
    """
    # 1. UniqueID (AVFoundation)
    uid = cam_cfg.get("unique_id")
    if uid:
        idx = _device_index_by_uid(uid)
        if idx is not None:
            print(f"Resolved camera unique_id={uid} -> index {idx}")
            return idx
        else:
            print(f"Warning: unique_id '{uid}' not found. Falling back …")

    # 2. Name substring
    name_sub = cam_cfg.get("name") or cam_cfg.get("device_name")
    if name_sub:
        idx = _device_index_by_name(name_sub)
        if idx is not None:
            print(f"Resolved camera name '{name_sub}' -> index {idx}")
            return idx
        else:
            print(f"Warning: camera name '{name_sub}' not found. Falling back …")

    # 3. Integer device_id (default 0)
    return int(cam_cfg.get("device_id", 0))


# ---------------------------------------------------------------------------
# Camera reader thread
# ---------------------------------------------------------------------------

def _camera_reader() -> None:
    """Continuously grab frames from the configured camera and store the
    most recent frame in *latest_frame*.
    """
    global latest_frame, camera_cap, camera_thread_running

    cam_cfg = config["camera"]

    # Resolve device index using the new helper (unique_id / name / fallback)
    dev_id = _resolve_device_index(cam_cfg)

    # Simple validation: attempt to open once to ensure device exists
    test_cap = cv2.VideoCapture(dev_id, cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else 0)
    if not test_cap.isOpened():
        raise RuntimeError(
            f"Unable to open camera (resolved id={dev_id}). Check config.json and available cameras."
        )
    test_cap.release()

    print(f"Selected camera device {dev_id} (resolved from config)")

    global selected_device_id
    selected_device_id = dev_id

    # Always prefer AVFoundation backend on macOS for stable ordering
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(dev_id, cv2.CAP_AVFOUNDATION)
    else:
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