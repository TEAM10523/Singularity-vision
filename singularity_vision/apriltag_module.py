from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import cv2

from .config import config, get_cam_fps
from .camera import get_latest_frame

from apriltag_detector import AprilTagDetector
from pose_estimator import SingleTagPoseEstimator, MultiTagPoseEstimator
import convertor

# ---------------------------------------------------------------------------
# Module-level shared state
# ---------------------------------------------------------------------------

apriltag_lock = threading.Lock()
latest_apriltag_result: Optional[dict] = None  # {"corners": ..., "ids": ...}

apriltag_pose_lock = threading.Lock()
latest_apriltag_poses: Optional[dict] = None  # see original structure

# These will be initialised if the feature is enabled in config
apriltag_detector: Optional[AprilTagDetector] = None
single_tag_estimator: Optional[SingleTagPoseEstimator] = None
multi_tag_estimator: Optional[MultiTagPoseEstimator] = None


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _initialize_apriltag():
    global apriltag_detector, single_tag_estimator, multi_tag_estimator

    cfg = config.get("apriltag", {})
    if not cfg.get("enabled", False):
        return  # Disabled

    # Detector itself
    apriltag_detector = AprilTagDetector()

    # Camera parameters
    camera_matrix = np.array(cfg["camera_matrix"])
    distortion_coeffs = np.array(cfg["distortion_coeffs"])

    # Camera pose
    from wpimath.geometry import Transform3d, Translation3d, Rotation3d

    camera_pose_dict = cfg["camera_pose"]
    camera_pose = Transform3d(
        Translation3d(camera_pose_dict["x"], camera_pose_dict["y"], camera_pose_dict["z"]),
        Rotation3d(camera_pose_dict["roll"], camera_pose_dict["pitch"], camera_pose_dict["yaw"]),
    )

    # Pose estimators
    tag_size = cfg["tag_size"]
    tag_layout = cfg["tag_layout"]

    single_tag_estimator = SingleTagPoseEstimator(tag_size, tag_layout, camera_matrix, distortion_coeffs, camera_pose)
    multi_tag_estimator = MultiTagPoseEstimator(tag_size, tag_layout, camera_matrix, distortion_coeffs, camera_pose)

    print("AprilTag pose estimation initialised")


# Call on import so detectors are ready before first request
_initialize_apriltag()

# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

def _apriltag_inferencer():
    global latest_apriltag_result, latest_apriltag_poses

    interval = 1.0 / float(get_cam_fps())

    while True:
        frame = get_latest_frame()
        if frame is not None and apriltag_detector is not None:
            try:
                ids, corners = apriltag_detector.detect(frame)

                with apriltag_lock:
                    latest_apriltag_result = {"corners": corners, "ids": ids}

                if ids is not None and len(ids):
                    single_tag_results = single_tag_estimator.estimate_poses(ids, corners)
                    multi_tag_pose, multi_tag_error = multi_tag_estimator.estimate_pose(ids, corners)

                    if multi_tag_pose is not None:
                        multi_tag_pose_list = convertor.robotPoseToList(multi_tag_pose)
                    else:
                        multi_tag_pose_list = [-9999] * 6

                    with apriltag_pose_lock:
                        latest_apriltag_poses = {
                            "single_tag": single_tag_results,
                            "multi_tag": {
                                "pose": multi_tag_pose_list,
                                "error": multi_tag_error if multi_tag_error is not None else -9999,
                            },
                        }
                else:
                    with apriltag_pose_lock:
                        latest_apriltag_poses = None
            except Exception as e:
                print(f"AprilTag inference error: {e}")
                with apriltag_lock:
                    latest_apriltag_result = {"corners": [], "ids": None}
                with apriltag_pose_lock:
                    latest_apriltag_poses = None

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def start_apriltag_thread() -> threading.Thread:
    t = threading.Thread(target=_apriltag_inferencer, daemon=True, name="AprilTagThread")
    t.start()
    return t


def get_latest_apriltag_result():
    with apriltag_lock:
        return latest_apriltag_result


def get_latest_apriltag_poses():
    with apriltag_pose_lock:
        return latest_apriltag_poses


def apriltag_enabled() -> bool:
    return apriltag_detector is not None


__all__ = [
    "start_apriltag_thread",
    "get_latest_apriltag_result",
    "get_latest_apriltag_poses",
    "apriltag_enabled",
] 