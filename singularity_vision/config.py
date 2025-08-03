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
# Dynamic override: load AprilTag layout from a user-specified JSON file.
# If config["apriltag"]["layout_file"] is set, it takes precedence.
# Otherwise fall back to auto-detecting a *reefscape*.json file (legacy behaviour).
# ---------------------------------------------------------------------------


def _apply_dynamic_apriltag_layout(cfg: Dict[str, Any]):
    """Populate cfg['apriltag']['tag_layout'] from an external JSON layout file.

    Precedence:
    1. If cfg['apriltag']['layout_file'] is provided, load from that path (relative
       to project root if not absolute).
    2. Otherwise, fall back to the legacy behaviour that auto-detects the first
       "*reefscape*.json" file in the project root.
    """

    apriltag_cfg = cfg.setdefault("apriltag", {})

    layout_path = apriltag_cfg.get("layout_file")

    tag_file: Path | None = None

    if layout_path:
        tag_file = Path(layout_path)
        if not tag_file.is_absolute():
            tag_file = _PROJECT_ROOT / tag_file
    else:
        # Legacy fallback – look for any reefscape JSON in the root
        from glob import glob
        candidates = glob(str(_PROJECT_ROOT / "*reefscape*.json"))
        if candidates:
            tag_file = Path(candidates[0])

    if tag_file and tag_file.exists():
        try:
            with tag_file.open("r", encoding="utf-8") as f:
                tag_data = json.load(f)
            layout = [{"ID": t["ID"], "pose": t["pose"]} for t in tag_data.get("tags", [])]

            apriltag_cfg["tag_layout"] = layout
            apriltag_cfg["source_file"] = str(tag_file)
        except Exception as exc:
            print(f"Warning: failed to load AprilTag layout from {tag_file}: {exc}")

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