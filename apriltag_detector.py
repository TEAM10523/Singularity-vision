import cv2
import numpy as np
from pupil_apriltags import Detector


class AprilTagDetector:
    """AprilTag detector using *pupil_apriltags* for higher accuracy and speed.

    The public interface remains compatible with the previous OpenCV-based
    implementation: :py:meth:`detect` returns *(ids, corners)* where *ids* is an
    ``np.ndarray`` of shape *(N, 1)`` (or ``None`` when no tags were found) and
    *corners* is an ``np.ndarray`` of shape *(N, 1, 4, 2)`` matching OpenCV’s
    ArUco output so the rest of the codebase does not need to change.
    """

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 4,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: bool = True,
        decode_sharpening: float = 0.25,
        debug: int = 0,
    ) -> None:
        self.detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=debug,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def detect(self, frame):
        """Detect AprilTags in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            The input BGR or grayscale image.

        Returns
        -------
        tuple (ids, corners)
            - *ids*: ``np.ndarray`` of detected tag IDs with shape *(N, 1)* or
              ``None`` if no tags are detected.
            - *corners*: ``np.ndarray`` of corner coordinates with shape
              *(N, 1, 4, 2)* following the same convention as OpenCV’s
              `aruco.detectMarkers`.
        """
        # Convert to grayscale as required by pupil_apriltags
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        detections = self.detector.detect(gray)

        if not detections:
            return None, []

        ids = np.array([[d.tag_id] for d in detections], dtype=np.int32)
        # Convert corners list to expected array shape (N,1,4,2)
        corners = np.array([[d.corners] for d in detections], dtype=np.float32)
        return ids, corners

    # Allow instances to be called directly
    __call__ = detect 