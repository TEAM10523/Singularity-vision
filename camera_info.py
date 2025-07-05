import cv2

print("Scanning for available cameras and their default resolution/FPS...\n")

max_cameras = 10
found_any = False

for cam_idx in range(max_cameras):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cap.release()
        continue
    found_any = True
    print(f"Camera {cam_idx} is available.")
    default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_fps = cap.get(cv2.CAP_PROP_FPS)
    if default_fps <= 0 or default_fps > 120:
        default_fps_str = "Unknown"
    else:
        default_fps_str = f"{default_fps:.2f}"
    print(f"  Default: {default_width}x{default_height} @ {default_fps_str} FPS\n")
    ret, frame = cap.read()
    if ret:
        win_name = f"Camera {cam_idx} Preview"
        cv2.imshow(win_name, frame)
        print(f"  Showing preview window for 2 seconds...")
        cv2.waitKey(2000)
        cv2.destroyWindow(win_name)
    else:
        print("  Warning: Could not read a frame for preview.\n")
    cap.release()

if not found_any:
    print("No cameras found.")
else:
    print("Scan complete.") 