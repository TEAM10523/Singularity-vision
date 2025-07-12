import cv2
import numpy as np
from wpimath.geometry import *
import convertor
from scipy.spatial import ConvexHull

class SingleTagPoseEstimator:
    """Single AprilTag pose estimator"""
    
    def __init__(self, tag_size: float, tag_layout: list, camera_matrix: np.ndarray, 
                 distortion_coeffs: np.ndarray, camera_pose: Transform3d):
        """Initialize single tag pose estimator
        
        Args:
            tag_size: Size of AprilTag in meters
            tag_layout: List of tag poses in field coordinate system
            camera_matrix: Camera intrinsic matrix
            distortion_coeffs: Camera distortion coefficients
            camera_pose: Camera pose relative to robot
        """
        self.tag_size = tag_size
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.robot_to_camera = camera_pose
        self.camera_to_robot = camera_pose.inverse()
        self.tag_layout = tag_layout
        
        # Create mapping of tag ID to field pose
        self.field_to_tag_poses = {}
        for tag in tag_layout:
            tag_pose = convertor.poseDictToWPIPose3d(tag["pose"])
            self.field_to_tag_poses[str(tag["ID"])] = tag_pose
            
        # Define tag corner points in tag coordinate system
        self.object_points = np.array([
            (-self.tag_size / 2, self.tag_size / 2, 0),
            (self.tag_size / 2, self.tag_size / 2, 0),
            (self.tag_size / 2, -self.tag_size / 2, 0),
            (-self.tag_size / 2, -self.tag_size / 2, 0)
        ])
    
    def calculate_area(self, corners):
        """Calculate area of detected tag corners"""
        hull = ConvexHull(corners[0])
        return hull.area
    
    def estimate_poses(self, ids: np.ndarray, corners):
        """Estimate poses from detected tags
        
        Args:
            ids: Array of detected tag IDs
            corners: Array of corner coordinates
            
        Returns:
            dict: Contains camera-to-tag poses, robot-to-tag poses, field-to-robot poses, and errors
        """
        camera_to_tag_poses = []
        robot_to_tag_poses = []
        field_to_robot_poses = []
        errors = []
        tag_ids = ids.tolist() if ids is not None else []
        
        for i in range(len(corners)):
            try:
                # Use solvePnP to calculate tag pose relative to camera
                _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    self.object_points, 
                    np.array(corners[i]),
                    self.camera_matrix,
                    self.distortion_coeffs
                )
                
                # Convert OpenCV pose to WPILib pose
                wpi_pose = convertor.openCvPoseToWpilib(tvecs[0], rvecs[0])
                
                # Add results to lists
                camera_to_tag_poses.append(convertor.robotPoseToList(wpi_pose))
                robot_to_tag_poses.append(convertor.robotPoseToList(wpi_pose.transformBy(self.robot_to_camera)))
                
                # Calculate field-to-robot pose using tag position
                if str(ids[i][0]) in self.field_to_tag_poses:
                    field_to_robot_pose = self.field_to_tag_poses[str(ids[i][0])].transformBy(
                        convertor.pose3dToTransform3d(wpi_pose).inverse()
                    ).transformBy(self.robot_to_camera)
                    field_to_robot_poses.append(convertor.robotPoseToList(field_to_robot_pose))
                else:
                    field_to_robot_poses.append([-9999, -9999, -9999, -9999, -9999, -9999])
                
                errors.append([1 / self.calculate_area(corners[i])])
                
            except Exception as e:
                print(f"Error estimating pose for tag {ids[i][0]}: {e}")
                camera_to_tag_poses.append([-9999, -9999, -9999, -9999, -9999, -9999])
                robot_to_tag_poses.append([-9999, -9999, -9999, -9999, -9999, -9999])
                field_to_robot_poses.append([-9999, -9999, -9999, -9999, -9999, -9999])
                errors.append([-9999])
        
        # Pad arrays to fixed size (10 elements)
        while len(camera_to_tag_poses) < 10:
            camera_to_tag_poses.append([-9999, -9999, -9999, -9999, -9999, -9999])
        while len(robot_to_tag_poses) < 10:
            robot_to_tag_poses.append([-9999, -9999, -9999, -9999, -9999, -9999])
        while len(field_to_robot_poses) < 10:
            field_to_robot_poses.append([-9999, -9999, -9999, -9999, -9999, -9999])
        while len(errors) < 10:
            errors.append([-9999])
        while len(tag_ids) < 10:
            tag_ids.append([-9999])
            
        return {
            'tag_ids': tag_ids,
            'camera_to_tag_poses': camera_to_tag_poses,
            'robot_to_tag_poses': robot_to_tag_poses,
            'field_to_robot_poses': field_to_robot_poses,
            'errors': errors
        }

class MultiTagPoseEstimator:
    """Multi-tag pose estimator using all visible tags"""
    
    def __init__(self, tag_size: float, tag_layout: list, camera_matrix: np.ndarray, 
                 distortion_coeffs: np.ndarray, camera_pose: Transform3d):
        """Initialize multi-tag pose estimator
        
        Args:
            tag_size: Size of AprilTag in meters
            tag_layout: List of tag poses in field coordinate system
            camera_matrix: Camera intrinsic matrix
            distortion_coeffs: Camera distortion coefficients
            camera_pose: Camera pose relative to robot
        """
        self.tag_size = tag_size
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.camera_to_robot = camera_pose.inverse()
        
        # Create corner poses for each tag
        self.corner_poses = {}
        for tag in tag_layout:
            tag_pose = convertor.poseDictToWPIPose3d(tag["pose"])
            # Calculate corner positions in field coordinate system
            corner0 = tag_pose + Transform3d(Translation3d(0.0, self.tag_size / 2.0, -self.tag_size / 2.0), Rotation3d())
            corner1 = tag_pose + Transform3d(Translation3d(0.0, -self.tag_size / 2.0, -self.tag_size / 2.0), Rotation3d())
            corner2 = tag_pose + Transform3d(Translation3d(0.0, -self.tag_size / 2.0, self.tag_size / 2.0), Rotation3d())
            corner3 = tag_pose + Transform3d(Translation3d(0.0, self.tag_size / 2.0, self.tag_size / 2.0), Rotation3d())
            
            tag_object_points = [
                convertor.wpilibTranslationtoOpenCv(corner0.translation()),
                convertor.wpilibTranslationtoOpenCv(corner1.translation()),
                convertor.wpilibTranslationtoOpenCv(corner2.translation()),
                convertor.wpilibTranslationtoOpenCv(corner3.translation())
            ]
            self.corner_poses[str(tag["ID"])] = tag_object_points
    
    def calculate_max_area(self, corners):
        """Calculate maximum area of detected tag corners"""
        points = np.vstack(corners)
        hull = ConvexHull(points[0])
        return hull.area
    
    def estimate_pose(self, ids, corners):
        """Estimate robot pose using multiple tags
        
        Args:
            ids: Array of detected tag IDs
            corners: Array of corner coordinates
            
        Returns:
            tuple: (field_to_robot_pose, estimation_error) or (None, None) if no tags detected
        """
        if len(corners) == 0:
            return None, None
            
        object_points = []
        observed_points = []
        
        # Create object points and observed points from all detected tags
        for i in range(len(corners)):
            tag_id = str(ids[i][0])
            if tag_id in self.corner_poses:
                observed_points.extend(corners[i][0])
                object_points.extend(self.corner_poses[tag_id])
        
        if len(object_points) == 0:
            return None, None
            
        try:
            # Solve PnP to get camera pose in field coordinate system
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                np.array(object_points), 
                np.array(observed_points),
                self.camera_matrix,
                self.distortion_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )
            
            # Convert OpenCV pose to WPILib pose
            camera_to_field_pose = convertor.openCvPoseToWpilib(tvecs[0], rvecs[0])
            
            # Transform to get field-to-robot pose
            camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
            field_to_camera = camera_to_field.inverse()
            field_to_camera_pose = Pose3d(field_to_camera.translation(), field_to_camera.rotation())
            
            field_to_robot_pose = field_to_camera_pose.transformBy(self.camera_to_robot)
            estimation_error = 1 / self.calculate_max_area(corners)
            
            return field_to_robot_pose, estimation_error
            
        except Exception as e:
            print(f"Error in multi-tag pose estimation: {e}")
            return None, None 