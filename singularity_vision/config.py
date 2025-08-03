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
# Dynamic override: load season-specific AprilTag layout if a file matching
# "*reefscape*.json" exists in the project root (e.g. "2025-reefscape-welded.json").
# ---------------------------------------------------------------------------


def _apply_dynamic_apriltag_layout(cfg: Dict[str, Any]):
    from glob import glob

    candidates = glob(str(_PROJECT_ROOT / "*reefscape*.json"))
    if not candidates:
        return cfg  # nothing to do

    tag_file = Path(candidates[0])
    try:
        with tag_file.open("r", encoding="utf-8") as f:
            tag_data = json.load(f)
        layout = [
            {"ID": t["ID"], "pose": t["pose"]}
            for t in tag_data.get("tags", [])
        ]

        cfg.setdefault("apriltag", {})["tag_layout"] = layout
        cfg["apriltag"]["source_file"] = tag_file.name
    except Exception as exc:
        # Fail silently but log for debugging
        print(f"Warning: failed to load dynamic AprilTag layout from {tag_file}: {exc}")

    return cfg


config = _apply_dynamic_apriltag_layout(config)


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