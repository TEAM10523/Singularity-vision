# AI Camera Dashboard

A smart camera dashboard supporting real-time object detection and AprilTag recognition, with both PyTorch and CoreML inference backends.

## Features

- 🎯 **Real-time Object Detection**: Supports YOLOv8 models, capable of detecting 80 common objects
- 🏷️ **AprilTag Recognition**: Real-time recognition and localization of AprilTag markers
- 🔄 **Dynamic Switching**: Switch between PyTorch and CoreML inference backends at runtime
- 📦 **Model Management**: Supports multiple model files, switchable on the fly
- 📊 **Live Monitoring**: Displays FPS, detection counts, and other real-time stats
- 🎨 **Modern UI**: Beautiful web interface with responsive design

## Project Structure

```
├── camera_server.py          # Main server file
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── models/                   # Model directory
│   ├── torch/                # PyTorch models
│   │   ├── yolov8n.pt
│   │   └── Reefscape_FRC_2-640-640-yolo11n.pt
│   └── coreml/               # CoreML models
│       └── yolov8n.mlpackage/
├── templates/                # HTML templates
│   └── dashboard.html
└── static/                   # Static resources
```

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit the `config.json` file to configure system parameters:

```json
{
  "inference": {
    "backend": "coreml",        // Inference backend: "torch" or "coreml"
    "fps": 30,                  // Target FPS
    "confidence_threshold": 0.5, // Confidence threshold
    "nms_threshold": 0.4        // NMS threshold
  },
  "models": {
    "torch": {
      "default": "yolov8n.pt",  // Default PyTorch model
      "available": [...]         // Available model list
    },
    "coreml": {
      "default": "yolov8n.mlpackage", // Default CoreML model
      "available": [...]         // Available model list
    }
  },
  "camera": {
    "device_id": 0,            // Camera device ID
    "width": 640,              // Camera width
    "height": 480              // Camera height
  }
}
```

## Usage

### 1. Start the Server

```bash
python camera_server.py
```

The server will start at `http://localhost:5001`.

### 2. Access the Web Interface

Open your browser and go to `http://localhost:5001` to see:

- **Control Panel**: Switch inference backend and models
- **Live Video Stream**: Shows detection results
- **Detection Info**: Displays AprilTag and object detection details
- **Stats**: Real-time FPS, detection counts, etc.

### 3. Dynamic Switching

#### Switch Inference Backend
- Select "PyTorch" or "CoreML" in the control panel
- Click the "Switch Backend" button
- The system will automatically reload the corresponding model

#### Switch Model
- Select a model from the dropdown menu
- Click the "Switch Model" button
- The system will load the new model file

## Model Files

### PyTorch Models
Place `.pt` files in the `models/torch/` directory and add their configuration in `config.json`:

```json
{
  "name": "Model Name",
  "file": "model_filename.pt",
  "description": "Model description"
}
```

### CoreML Models
Place `.mlpackage` folders in the `models/coreml/` directory and add their configuration in `config.json`.

## API Endpoints

### Get Configuration
```
GET /api/config
```

### Switch Inference Backend
```
POST /api/switch_backend
Content-Type: application/json

{
  "backend": "torch"  // or "coreml"
}
```

### Switch Model
```
POST /api/switch_model
Content-Type: application/json

{
  "backend": "torch",  // or "coreml"
  "model": "yolov8n.pt"
}
```

### Get Detection Results
```
GET /api/detections
```

### Video Stream
```
GET /video_feed
```

## Performance Optimization

### PyTorch Backend
- Supports Apple Silicon MPS acceleration
- Automatically detects and uses Metal Performance Shaders
- CPU fallback supported

### CoreML Backend
- Optimized for Apple Silicon
- Faster inference speed
- Lower power consumption

## Troubleshooting

### Common Issues

1. **Camera cannot be opened**
   - Check `device_id` in `config.json`
   - Make sure the camera is not used by another program

2. **Model loading failed**
   - Check if the model file path is correct
   - Ensure the model file is complete and not corrupted

3. **Slow inference speed**
   - Try lowering the FPS setting
   - Use a smaller model file
   - Check if hardware acceleration is enabled

4. **Inaccurate detection results**
   - Adjust `confidence_threshold` and `nms_threshold`
   - Try different model files

## Development Notes

### Add New Models
1. Place the model file in the appropriate directory
2. Add the model configuration to `config.json`
3. Restart the server or switch via API

### Customize Detection Logic
Modify the inference functions in `camera_server.py`:
- `yolo_inferencer()`: YOLO inference thread
- `apriltag_inferencer()`: AprilTag inference thread

### Extend Frontend Features
Edit `templates/dashboard.html` to add new UI components and features.

## License

This project is licensed under the MIT License.

## Contributing

Feel free to submit Issues and Pull Requests to improve the project! 