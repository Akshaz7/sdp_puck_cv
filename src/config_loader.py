import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {
    "camera": ["source", "width", "height", "fps"],
    "detection": ["hsv_lower", "hsv_upper", "min_contour_area", "blur_kernel_size"],
    "table": ["width_mm", "height_mm", "corners_px", "defence_y_mm"],
    "tracker": ["history_length", "smoothing_window", "max_missed_frames"],
    "predictor": ["max_bounces"],
    "serial": ["port", "baud_rate", "min_send_interval"],
    "debug": ["show_windows", "show_fps", "show_trajectory"],
}


def load_config(config_path: str = "config/settings.json") -> dict:
    """Load and validate the configuration file.

    Args:
        config_path: Path to settings.json relative to project root.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If config file does not exist.
        KeyError: If required keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = json.load(f)

    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise KeyError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing config key: {section}.{key}")

    logger.info(f"Loaded config from {config_path}")
    return config


def get_hsv_range(config: dict) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Extract HSV lower and upper bounds from config.

    Returns:
        Tuple of (lower, upper) HSV bounds as tuples of 3 ints.
    """
    lower = tuple(config["detection"]["hsv_lower"])
    upper = tuple(config["detection"]["hsv_upper"])
    return (lower, upper)


def get_table_dimensions(config: dict) -> tuple[float, float]:
    """Extract table width and height in mm from config.

    Returns:
        Tuple of (width_mm, height_mm).
    """
    return (
        float(config["table"]["width_mm"]),
        float(config["table"]["height_mm"]),
    )
