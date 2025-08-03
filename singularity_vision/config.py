import json
from pathlib import Path
from typing import Any, Dict

# Resolve the config.json file located in the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def _load_config(path: Path = _CONFIG_PATH) -> Dict[str, Any]:
    """Load the JSON configuration file from *path* and return it as a dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# The immutable configuration dictionary (feel free to mutate nested values
# if needed)
config: Dict[str, Any] = _load_config()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_cam_fps() -> float:
    """Return the desired camera FPS as configured."""
    return config["inference"]["fps"]


def get_confidence_threshold() -> float:
    return config["inference"]["confidence_threshold"]


def get_nms_threshold() -> float:
    return config["inference"]["nms_threshold"]


__all__ = [
    "config",
    "get_cam_fps",
    "get_confidence_threshold",
    "get_nms_threshold",
] 