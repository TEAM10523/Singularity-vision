from __future__ import annotations

import atexit
import signal
import sys
from pathlib import Path
from flask import Flask

from .routes import api_bp
from .camera import start_camera_thread, cleanup_camera
from .yolo import start_yolo_thread
from .apriltag_module import start_apriltag_thread

# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_app() -> Flask:
    # Explicitly set template_folder to the shared templates directory in the project root.
    app = Flask(__name__, template_folder=str(_PROJECT_ROOT / "templates"))
    app.register_blueprint(api_bp)
    return app


# ---------------------------------------------------------------------------
# Background thread startup
# ---------------------------------------------------------------------------


_camera_thread = None
_yolo_thread = None
_apriltag_thread = None


def _start_background_threads():
    global _camera_thread, _yolo_thread, _apriltag_thread
    _camera_thread = start_camera_thread()
    _yolo_thread = start_yolo_thread()
    _apriltag_thread = start_apriltag_thread()


# ---------------------------------------------------------------------------
# Cleanup handling (signal + atexit)
# ---------------------------------------------------------------------------


def _cleanup(*_args):
    print("\nCleaning up resources (signal/exit handler)...")
    cleanup_camera()
    # No explicit clean-up necessary for YOLO/CoreML threads; they are daemons.

    sys.stdout.flush()
    # Force process termination to avoid werkzeug hanging
    import os
    os._exit(0)


# Register handlers
signal.signal(signal.SIGINT, _cleanup)  # CTRL+C
signal.signal(signal.SIGTERM, _cleanup)
atexit.register(_cleanup)


# ---------------------------------------------------------------------------
# Main entry point when executed via `python -m singularity_vision.app`
# ---------------------------------------------------------------------------


def main():
    app = create_app()
    _start_background_threads()
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main() 