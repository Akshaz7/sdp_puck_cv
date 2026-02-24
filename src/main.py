"""Main pipeline for GOAL-E puck tracking system.

Usage:
    python -m src.main                          # webcam, default config
    python -m src.main --source video.mp4       # video file
    python -m src.main --source 0               # explicit webcam index
    python -m src.main --no-serial              # disable Arduino communication
    python -m src.main --no-display             # headless mode, no OpenCV windows

The pipeline runs the following loop at each frame:
    1. Capture frame from camera
    2. Detect puck (pixel coordinates)
    3. Convert pixel position to mm coordinates via homography
    4. Update tracker with mm position
    5. If tracker has velocity and puck is approaching, predict trajectory
    6. Send interception x coordinate to Arduino via serial
    7. Draw debug overlays if display is enabled
    8. Show frame in OpenCV window if display is enabled
    9. Calculate and display FPS
    10. Exit on 'q' key press
"""

import argparse
import logging
import time
import cv2
import numpy as np
from collections import deque
from src.config_loader import load_config, get_hsv_range, get_table_dimensions
from src.camera import Camera
from src.detector import PuckDetector
from src.homography import HomographyMapper
from src.tracker import PuckTracker
from src.predictor import TrajectoryPredictor
from src.serial_comms import ArduinoComms
from src.visualiser import Visualiser
from src.models import HSVRange, Position

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Arguments:
        --source: Camera source. Integer for webcam index, string for file path.
                  Default: value from config/settings.json camera.source.
        --config: Path to settings.json. Default: config/settings.json.
        --no-serial: Disable Arduino serial communication.
        --no-display: Disable OpenCV window display (headless mode).
    """
    parser = argparse.ArgumentParser(
        description="GOAL-E Air Hockey Puck Tracking System"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Camera source: integer for webcam index, or file path",
    )
    parser.add_argument(
        "--config",
        default="config/settings.json",
        help="Path to settings.json",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Disable Arduino serial communication",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window display (headless mode)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the main tracking pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    # Determine source
    source = config["camera"]["source"]
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    # Initialise camera
    cam = Camera(
        source=source,
        width=config["camera"]["width"],
        height=config["camera"]["height"],
        fps=config["camera"]["fps"],
    )

    # Initialise detector
    hsv_lower, hsv_upper = get_hsv_range(config)
    hsv_range = HSVRange(lower=hsv_lower, upper=hsv_upper)
    detector = PuckDetector(
        hsv_range=hsv_range,
        min_contour_area=config["detection"]["min_contour_area"],
        blur_kernel=config["detection"]["blur_kernel_size"],
    )

    # Initialise homography mapper
    corners = np.array(
        config["table"]["corners_px"], dtype=np.float32
    )
    table_width, table_height = get_table_dimensions(config)
    mapper = HomographyMapper(
        pixel_corners=corners,
        table_width_mm=table_width,
        table_height_mm=table_height,
    )

    # Initialise tracker
    tracker = PuckTracker(
        history_length=config["tracker"]["history_length"],
        smoothing_window=config["tracker"]["smoothing_window"],
        max_missed_frames=config["tracker"]["max_missed_frames"],
    )

    # Initialise predictor
    predictor = TrajectoryPredictor(
        table_width=table_width,
        table_height=table_height,
        defence_y=config["table"]["defence_y_mm"],
        max_bounces=config["predictor"]["max_bounces"],
    )

    # Initialise serial comms
    arduino: ArduinoComms | None = None
    if not args.no_serial:
        arduino = ArduinoComms(
            port=config["serial"]["port"],
            baud_rate=config["serial"]["baud_rate"],
            min_send_interval=config["serial"]["min_send_interval"],
        )
        if not arduino.connect():
            logger.warning("Arduino connection failed, continuing without serial")
            arduino = None

    # Initialise visualiser
    visualiser: Visualiser | None = None
    show_display = not args.no_display and config["debug"]["show_windows"]
    if show_display:
        visualiser = Visualiser(
            show_fps=config["debug"]["show_fps"],
            show_trajectory=config["debug"]["show_trajectory"],
        )

    # FPS tracking
    frame_times: deque[float] = deque(maxlen=30)

    logger.info("Starting main tracking loop")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                break

            # Detect puck
            detection = detector.detect(frame)

            # Convert to mm and update tracker
            position_mm: Position | None = None
            if detection.found and detection.position_px is not None:
                position_mm = mapper.pixel_to_mm(detection.position_px)

            state = tracker.update(position_mm)

            # Predict trajectory
            prediction = None
            if tracker.has_velocity and state.velocity is not None:
                if state.current_position is not None:
                    prediction = predictor.predict(
                        state.current_position, state.velocity
                    )

            # Send to Arduino
            if (
                arduino is not None
                and prediction is not None
                and prediction.is_approaching
            ):
                arduino.send_target(prediction.interception_x)

            # Calculate FPS
            now = time.time()
            frame_times.append(now)
            fps = 0.0
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    fps = (len(frame_times) - 1) / elapsed

            # Visualisation
            if visualiser is not None:
                visualiser.draw_detection(frame, detection)
                visualiser.draw_tracking_state(
                    frame, state, mapper.mm_to_pixel
                )
                if prediction is not None and prediction.is_approaching:
                    visualiser.draw_trajectory(
                        frame, prediction, mapper.mm_to_pixel
                    )
                visualiser.draw_fps(frame, fps)

                mask = detector.get_mask(frame)
                cv2.imshow("GOAL-E Tracker", frame)
                cv2.imshow("Mask", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested")
                    break
    finally:
        cam.release()
        if arduino is not None:
            arduino.disconnect()
        if show_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
