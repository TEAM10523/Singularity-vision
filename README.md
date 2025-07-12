# Singularity Vision

A real-time computer vision system supporting both YOLO object detection and AprilTag pose estimation with coordinate transformation.

## Features

- **YOLO Object Detection**: Support for both PyTorch and CoreML backends
- **AprilTag Detection**: Real-time AprilTag detection and pose estimation
- **Coordinate Transformation**: Convert AprilTag poses to world coordinates using WPILib geometry
- **Multi-tag Pose Estimation**: Improved accuracy using multiple AprilTags
- **Real-time Streaming**: Live video feed with overlay annotations
- **REST API**: Complete API for accessing detection and pose data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python camera_server.py
```

## AprilTag Coordinate Transformation

This system implements a complete AprilTag pose estimation pipeline similar to BlackholeVision2, including:

### Key Components

1. **AprilTag Detector** (`apriltag_detector.py`): Detects AprilTags using OpenCV ArUco
2. **Pose Estimators** (`pose_estimator.py`): 
   - Single tag pose estimation
   - Multi-tag pose estimation for improved accuracy
3. **Coordinate Converter** (`convertor.py`): Utilities for coordinate system transformations

### Configuration

Configure AprilTag settings in `config.json`:

```json
{
  "apriltag": {
    "enabled": true,
    "tag_size": 0.165,
    "camera_matrix": [...],
    "distortion_coeffs": [...],
    "camera_pose": {
      "x": 0.0, "y": 0.0, "z": 0.5,
      "roll": 0.0, "pitch": 0.0, "yaw": 0.0
    },
    "tag_layout": [
      {
        "ID": 1,
        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
      }
    ]
  }
}
```

### API Endpoints

- `GET /api/apriltag_poses`: Get pose estimation results
- `GET /api/apriltag_config`: Get AprilTag configuration
- `GET /api/detections`: Get basic detection results

### Coordinate Systems

The system provides poses in multiple coordinate frames:
- **Camera to Tag**: Tag position relative to camera
- **Robot to Tag**: Tag position relative to robot
- **Field to Robot**: Robot position in field coordinates (using known tag positions)

### Example Usage

```python
import requests

# Get pose estimation results
response = requests.get('http://localhost:5000/api/apriltag_poses')
poses = response.json()

# Access single tag poses
for tag_pose in poses['single_tag_poses']:
    print(f"Tag {tag_pose['tag_id']} field-to-robot pose: {tag_pose['field_to_robot_pose']}")

# Access multi-tag pose (more accurate)
if poses['multi_tag_pose']:
    robot_pose = poses['multi_tag_pose']['field_to_robot_pose']
    print(f"Robot position: x={robot_pose['x']:.3f}, y={robot_pose['y']:.3f}, yaw={robot_pose['yaw']:.3f}")
```

## Development

The implementation is based on the BlackholeVision2 codebase but adapted for the Singularity-vision architecture with:
- Simplified threading model
- REST API integration
- Real-time coordinate transformation
- Multi-backend support (PyTorch/CoreML)

## Dependencies

- OpenCV for computer vision
- WPILib geometry for coordinate transformations
- SciPy for convex hull calculations
- NumPy for numerical operations
- Flask for web API 