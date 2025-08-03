import platform, AVFoundation
if platform.system() != "Darwin":
    raise SystemExit("This script only works on macOS.")
devices = sorted(
    AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo),
    key=lambda d: str(d.uniqueID())
)
for idx, dev in enumerate(devices):
    print(f"index={idx:2d}  uniqueID={dev.uniqueID()}  name={dev.localizedName()}")