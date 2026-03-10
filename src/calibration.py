import cv2
import json
import logging
from pathlib import Path
from src.camera import Camera
from src.detector import PuckDetector
from src.models import HSVRange

logger = logging.getLogger(__name__)


def _nothing(x: int) -> None:
    """Trackbar callback (OpenCV requires one)."""
    pass


def _save_hsv_to_config(hsv_range: HSVRange, config_path: str) -> None:
    """Save HSV range to config file."""
    path = Path(config_path)
    with open(path, "r") as f:
        config = json.load(f)

    config["detection"]["hsv_lower"] = list(hsv_range.lower)
    config["detection"]["hsv_upper"] = list(hsv_range.upper)

    with open(path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(
        f"Saved HSV range: lower={hsv_range.lower}, upper={hsv_range.upper}"
    )


def run_calibration(
    source: int | str = 0,
    config_path: str = "config/settings.json",
) -> HSVRange:
    """Run the interactive HSV calibration tool.

    Opens two OpenCV windows:
        1. "Original" - the raw camera frame with detected contour drawn in green
        2. "Mask" - the binary threshold result

    Creates a third window "Controls" with 6 trackbars:
        H_min (0-179), S_min (0-255), V_min (0-255)
        H_max (0-179), S_max (0-255), V_max (0-255)

    Trackbar initial values are loaded from config/settings.json.

    Key bindings:
        'q' or ESC - quit and save current values to settings.json
        's' - save current values to settings.json without quitting
        'r' - reset trackbars to defaults (H: 35-85, S: 100-255, V: 100-255)

    Args:
        source: Camera source (device index or file path).
        config_path: Path to settings.json to load defaults and save results.

    Returns:
        The final HSVRange that was saved.
    """
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            config = json.load(f)
        h_min = config["detection"]["hsv_lower"][0]
        s_min = config["detection"]["hsv_lower"][1]
        v_min = config["detection"]["hsv_lower"][2]
        h_max = config["detection"]["hsv_upper"][0]
        s_max = config["detection"]["hsv_upper"][1]
        v_max = config["detection"]["hsv_upper"][2]
    else:
        h_min, s_min, v_min = 35, 100, 100
        h_max, s_max, v_max = 85, 255, 255

    hsv_range = HSVRange(
        lower=(h_min, s_min, v_min), upper=(h_max, s_max, v_max)
    )
    detector = PuckDetector(hsv_range=hsv_range)
    cam = Camera(source=source)

    cv2.namedWindow("Controls")
    cv2.createTrackbar("H_min", "Controls", h_min, 179, _nothing)
    cv2.createTrackbar("S_min", "Controls", s_min, 255, _nothing)
    cv2.createTrackbar("V_min", "Controls", v_min, 255, _nothing)
    cv2.createTrackbar("H_max", "Controls", h_max, 179, _nothing)
    cv2.createTrackbar("S_max", "Controls", s_max, 255, _nothing)
    cv2.createTrackbar("V_max", "Controls", v_max, 255, _nothing)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                break

            h_min = cv2.getTrackbarPos("H_min", "Controls")
            s_min = cv2.getTrackbarPos("S_min", "Controls")
            v_min = cv2.getTrackbarPos("V_min", "Controls")
            h_max = cv2.getTrackbarPos("H_max", "Controls")
            s_max = cv2.getTrackbarPos("S_max", "Controls")
            v_max = cv2.getTrackbarPos("V_max", "Controls")

            hsv_range = HSVRange(
                lower=(h_min, s_min, v_min),
                upper=(h_max, s_max, v_max),
            )
            detector.update_hsv_range(hsv_range)

            detection = detector.detect(frame)
            mask = detector.get_mask(frame)

            display = frame.copy()
            if detection.found and detection.bounding_rect is not None:
                x, y, w, h = detection.bounding_rect
                cv2.rectangle(
                    display, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                if detection.position_px is not None:
                    cv2.circle(
                        display,
                        (
                            int(detection.position_px.x),
                            int(detection.position_px.y),
                        ),
                        5,
                        (0, 255, 0),
                        -1,
                    )

            cv2.imshow("Original", display)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                _save_hsv_to_config(hsv_range, config_path)
                break
            elif key == ord("s"):
                _save_hsv_to_config(hsv_range, config_path)
            elif key == ord("r"):
                cv2.setTrackbarPos("H_min", "Controls", 35)
                cv2.setTrackbarPos("S_min", "Controls", 100)
                cv2.setTrackbarPos("V_min", "Controls", 100)
                cv2.setTrackbarPos("H_max", "Controls", 85)
                cv2.setTrackbarPos("S_max", "Controls", 255)
                cv2.setTrackbarPos("V_max", "Controls", 255)
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return hsv_range


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HSV Calibration Tool")
    parser.add_argument(
        "--source", default=0,
        help="Camera source: integer for webcam index, or file path",
    )
    parser.add_argument(
        "--config", default="config/settings.json",
        help="Path to settings.json",
    )
    args = parser.parse_args()
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    run_calibration(source=source, config_path=args.config)
