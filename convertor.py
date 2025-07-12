import math
from wpimath.geometry import * 
import numpy as np
import torch

def poseDictToWPIPose3d(poseDict: dict):
    """Convert dictionary pose to WPILib Pose3d"""
    try:
        return Pose3d(poseDict["x"], poseDict["y"], poseDict["z"],
                    Rotation3d(poseDict["roll"], poseDict["pitch"], poseDict["yaw"]))
    except:
        rotation = poseDict["rotation"]["quaternion"]
        return Pose3d(poseDict["translation"]["x"], poseDict["translation"]["y"], poseDict["translation"]["z"],
                    Rotation3d(Quaternion(rotation["W"], rotation["X"], rotation["Y"], rotation["Z"])))
    
def poseDictToWPITransform3d(poseDict: dict):
    """Convert dictionary pose to WPILib Transform3d"""
    return Transform3d(Translation3d(poseDict["x"], poseDict["y"], poseDict["z"]),
                  Rotation3d(poseDict["roll"], poseDict["pitch"], poseDict["yaw"]))
    
def pose3dToTransform3d(pose):
    """Convert Pose3d to Transform3d"""
    return Transform3d(pose.translation(), pose.rotation())

def pose2dToTransform2d(pose):
    """Convert Pose2d to Transform2d"""
    return Transform2d(pose.translation(), pose.rotation())

def wpilibTranslationtoOpenCv(translation: Translation3d):
    """Convert WPILib Translation3d to OpenCV coordinate system"""
    return [-translation.Y(), -translation.Z(), translation.X()]

def openCvPoseToWpilib(tvec, rvec) -> Pose3d:
    """Convert OpenCV pose (tvec, rvec) to WPILib Pose3d"""
    return Pose3d(
        Translation3d(tvec[2][0], -tvec[0][0], -tvec[1][0]),
        Rotation3d(
            np.array([rvec[2][0], -rvec[0][0], -rvec[1][0]]),
            math.sqrt(math.pow(rvec[0][0], 2) + math.pow(rvec[1][0], 2) + math.pow(rvec[2][0], 2))
        )
    )

def robotPoseToList(robotPose):
    """Convert robot pose to list [x, y, z, rx, ry, rz]"""
    if robotPose is not None:
        pose = robotPose
        return np.array([pose.X(), pose.Y(), pose.Z(), pose.rotation().X(), pose.rotation().Y(), pose.rotation().Z()])
    return np.array([0, 0, 0, 0, 0, 0])

def robotPoseToTensor(robotPose):
    """Convert robot pose to PyTorch tensor"""
    if robotPose is not None:
        pose = robotPose
        return torch.tensor([pose.X(), pose.Y(), pose.Z(), pose.rotation().X(), pose.rotation().Y(), pose.rotation().Z()])
    return torch.tensor([0, 0, 0, 0, 0, 0])
    
def listToRobotPose(robotPose):
    """Convert list to robot pose"""
    try:
        return Pose3d(
            Translation3d(robotPose[0], robotPose[1], robotPose[2]),
            Rotation3d(robotPose[3], robotPose[4], robotPose[5]))
    except:
        return None

def inverseRotation(rotation):
    """Get inverse rotation"""
    quaternion = rotation.getQuaternion()
    quaternion = quaternion.inverse()
    inversedRotation = Rotation3d(quaternion)
    return inversedRotation 