import cv2
import numpy as np

class AprilTagDetector:
    """AprilTag detector using OpenCV ArUco implementation"""
    
    def __init__(self, dictionary_id=cv2.aruco.DICT_APRILTAG_36H11):
        """Initialize AprilTag detector
        
        Args:
            dictionary_id: ArUco dictionary ID (default: DICT_APRILTAG_36H11)
        """
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
    
    def detect(self, frame):
        """Detect AprilTags in the frame
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            tuple: (ids, corners) where ids is array of detected tag IDs 
                   and corners is array of corner coordinates
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = frame
            
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray_image)
        
        return ids, corners
    
    def __call__(self, frame):
        """Make the detector callable"""
        return self.detect(frame) 