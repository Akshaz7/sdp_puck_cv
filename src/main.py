"""Main pipeline for GOAL-E puck tracking system.

Usage:
    python -m src.main                          # webcam, default config
    python -m src.main --source video.mp4       # video file
    python -m src.main --source 0               # explicit webcam index
    python -m src.main --no-serial              # disable Arduino communication
    python -m src.main --no-display             # headless mode, no OpenCV windows
    python -m src.main --list-cameras           # list available cameras and exit

The pipeline runs the following loop at each frame:
    1. Capture frame from camera
    2. Detect puck (pixel coordinates)
    3. Convert pixel position to mm coordinates via homography
    4. Update tracker with mm position
    5. If tracker has velocity, predict trajectory
    6. Send interception x coordinate to Arduino via serial
    7. Draw debug overlays if display is enabled
    8. Show frame in OpenCV window if display is enabled
    9. Calculate and display FPS
    10. Exit on 'q' key press
"""

import argparse
import json
import logging
import time
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Union
from src.config_loader import load_config, get_hsv_range, get_table_dimensions
from src.camera import Camera, list_cameras
from src.detector import PuckDetector
from src.homography import HomographyMapper
from src.tracker import PuckTracker
from src.predictor import TrajectoryPredictor
from src.serial_comms import ArduinoComms
from src.display_comms import DisplayComms
from src.goal_detector import GoalDetector
from src.game_manager import GameManager, GameMode, GameState
from src.visualiser import Visualiser
from src.models import HSVRange, Position, Prediction

logger = logging.getLogger(__name__)


def run_defence_calibration(
    source: Union[int, str] = 0,
    config_path: str = "config/settings.json",
) -> list[list[int]]:
    """Interactive tool to click two points defining the defence line.

    Opens a window showing the camera feed with table boundary overlay.
    The user clicks two points to define the defence line.

    Key bindings:
        'z' - undo last click
        'r' - reset all clicks
        'q' - quit without saving
        ENTER - confirm and save (only after 2 points)

    Args:
        source: Camera source.
        config_path: Path to settings.json.

    Returns:
        List of two [x, y] pixel coordinate pairs.
    """
    path = Path(config_path)
    config = {}
    if path.exists():
        with open(path, "r") as f:
            config = json.load(f)

    corners = config.get("table", {}).get("corners_px", [])

    points: list[tuple[int, int]] = []

    def mouse_callback(
        event: int, x: int, y: int, flags: int, param
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            logger.info(f"Defence point {len(points)}: ({x}, {y})")

    cam = Camera(source=source)
    cv2.namedWindow("Defence Line Calibration")
    cv2.setMouseCallback("Defence Line Calibration", mouse_callback)

    result: list[list[int]] = []

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                break

            display = frame.copy()

            # Draw table boundary if corners exist
            if corners:
                pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    display, [pts], isClosed=True,
                    color=(255, 255, 0), thickness=2,
                )

            # Draw clicked points
            for i, (cx, cy) in enumerate(points):
                cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)
                cv2.putText(
                    display, f"P{i + 1}",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2,
                )

            # Draw line between the two points
            if len(points) == 2:
                cv2.line(
                    display, points[0], points[1], (0, 0, 255), 2,
                )

            status = f"Click defence line point {len(points) + 1}/2"
            if len(points) == 2:
                status = "Press ENTER to save, 'r' to reset"
            cv2.putText(
                display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("Defence Line Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z") and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key == 13 and len(points) == 2:
                result = [[p[0], p[1]] for p in points]
                config.setdefault("table", {})["defence_line_px"] = result
                with open(path, "w") as f:
                    json.dump(config, f, indent=4)
                logger.info(f"Saved defence line to {config_path}")
                print(f"Defence line saved: {result}")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Arguments:
        --source: Camera source. Integer for webcam index, string for file path.
                  Default: value from config/settings.json camera.source.
        --config: Path to settings.json. Default: config/settings.json.
        --no-serial: Disable Arduino serial communication.
        --no-display: Disable OpenCV window display (headless mode).
        --list-cameras: List available cameras and exit.
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
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit",
    )
    parser.add_argument(
        "--calibrate-defence",
        action="store_true",
        help="Calibrate defence line by clicking two points",
    )
    parser.add_argument(
        "--no-tft",
        action="store_true",
        help="Disable TFT display Arduino communication",
    )
    return parser.parse_args()


def main() -> None:
    """Run the main tracking pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Handle --list-cameras
    if args.list_cameras:
        cameras = list_cameras()
        if not cameras:
            print("No cameras found.")
        else:
            for cam in cameras:
                print(
                    f"Index {cam['index']}: Camera {cam['index']} "
                    f"({cam['backend']}) {cam['width']}x{cam['height']}"
                )
        return

    config = load_config(args.config)

    # Determine source
    source = config["camera"]["source"]
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    # Handle --calibrate-defence
    if args.calibrate_defence:
        run_defence_calibration(source=source, config_path=args.config)
        return

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
    defence_y = config["table"]["defence_y_mm"]
    predictor = TrajectoryPredictor(
        table_width=table_width,
        table_height=table_height,
        defence_y=defence_y,
        max_bounces=config["predictor"]["max_bounces"],
    )

    # Initialise serial comms (motor Arduino)
    arduino: Optional[ArduinoComms] = None
    if not args.no_serial:
        arduino = ArduinoComms(
            port=config["serial"]["port"],
            baud_rate=config["serial"]["baud_rate"],
            min_send_interval=config["serial"]["min_send_interval"],
        )
        if not arduino.connect():
            logger.warning("Arduino connection failed, continuing without serial")
            arduino = None

    # Initialise TFT display comms
    display: Optional[DisplayComms] = None
    if not args.no_tft:
        display_cfg = config.get("display", {})
        display = DisplayComms(
            port=display_cfg.get("port", "/dev/cu.usbmodem2201"),
            baud_rate=display_cfg.get("baud_rate", 9600),
        )
        if not display.connect():
            logger.warning("TFT display connection failed, continuing without it")
            display = None

    # Initialise goal detector and game manager
    goal_detector = GoalDetector(table_height=table_height)
    game = GameManager()

    # Initialise visualiser
    debug_cfg = config["debug"]
    visualiser: Optional[Visualiser] = None
    show_display = not args.no_display and debug_cfg["show_windows"]
    if show_display:
        visualiser = Visualiser(
            show_fps=debug_cfg["show_fps"],
            show_trajectory=debug_cfg["show_trajectory"],
            show_table_boundary=debug_cfg.get("show_table_boundary", True),
            show_bounce_markers=debug_cfg.get("show_bounce_markers", True),
            show_prediction_info=debug_cfg.get("show_prediction_info", True),
            trajectory_persistence_frames=debug_cfg.get(
                "trajectory_persistence_frames", 15
            ),
            table_width_mm=table_width,
        )

    # Pre-compute table boundary pixel coordinates for visualisation
    table_corner_pixels = corners.copy()

    # Load defence line pixel coordinates (set via --calibrate-defence)
    defence_line_px = config["table"].get("defence_line_px", None)

    # FPS tracking
    frame_times: deque[float] = deque(maxlen=30)

    # Trajectory lock-in state
    locked_prediction: Optional[Prediction] = None
    locked_angle: Optional[float] = None
    min_speed_mm_s = 50.0        # ignore stationary noise
    angle_change_deg = 15.0      # only re-predict if direction changes this much
    frames_since_lock = 0
    max_lock_frames = 60         # force re-evaluate after this many frames

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
            position_mm: Optional[Position] = None
            if detection.found and detection.position_px is not None:
                position_mm = mapper.pixel_to_mm(detection.position_px)

            state = tracker.update(position_mm)

            # Locked trajectory prediction — only recompute when direction
            # changes significantly, not every frame
            if tracker.has_velocity and state.velocity is not None:
                speed = np.sqrt(
                    state.velocity.vx ** 2 + state.velocity.vy ** 2
                )
                if state.current_position is not None and speed >= min_speed_mm_s:
                    current_angle = np.degrees(
                        np.arctan2(state.velocity.vy, state.velocity.vx)
                    )
                    should_update = False

                    if locked_prediction is None or locked_angle is None:
                        should_update = True
                    else:
                        angle_diff = abs(current_angle - locked_angle)
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        if angle_diff > angle_change_deg:
                            should_update = True
                        elif frames_since_lock > max_lock_frames:
                            should_update = True

                    if should_update:
                        locked_prediction = predictor.predict(
                            state.current_position, state.velocity
                        )
                        locked_angle = current_angle
                        frames_since_lock = 0
                    else:
                        frames_since_lock += 1
                else:
                    # Puck too slow — clear the lock
                    locked_prediction = None
                    locked_angle = None
                    frames_since_lock = 0
            else:
                # No velocity — clear the lock
                if state.frames_since_detection > 30:
                    locked_prediction = None
                    locked_angle = None
                    frames_since_lock = 0

            # Send to Arduino
            if arduino is not None:
                if (
                    locked_prediction is not None
                    and locked_prediction.is_approaching
                ):
                    # Move to interception point (X along defence line, Y = defence_y)
                    arduino.send_target(
                        locked_prediction.interception_x,
                        locked_prediction.interception_y,
                    )
                elif state.current_position is not None:
                    # Puck not approaching — shadow puck's X, hold at defence line
                    arduino.send_target(state.current_position.x, defence_y)

            # --- Goal detection and game management ---
            if game.state == GameState.PLAYING:
                goal = goal_detector.check(position_mm)
                if goal is not None:
                    game.record_goal(goal)
                    if display is not None:
                        display.send_goal(goal)
                        display.send_score(game.human_score, game.robot_score)
                    logger.info(
                        f"GOAL! {goal} scored. "
                        f"Score: {game.human_score}-{game.robot_score}"
                    )

                game.update()  # check timer-based win conditions

                # Send timer to display for timed modes
                if display is not None and game.remaining_seconds > 0:
                    display.send_timer(game.remaining_seconds)

                # Handle game over
                if game.state == GameState.FINISHED and display is not None:
                    display.send_state(GameState.FINISHED)
                    if game.winner is not None:
                        display.send_winner(game.winner)

            # --- Process commands from TFT touchscreen ---
            if display is not None:
                for cmd in display.get_commands():
                    if cmd.startswith("MODE:"):
                        mode_str = cmd[5:]
                        try:
                            game.set_mode(GameMode(mode_str))
                        except ValueError:
                            logger.warning(f"Unknown game mode: {mode_str}")
                    elif cmd == "START":
                        game.start()
                        goal_detector.reset()
                        display.send_state(GameState.PLAYING)
                        display.send_score(0, 0)
                        logger.info("Game started from TFT")
                    elif cmd == "RESET":
                        game.reset()
                        goal_detector.reset()
                        display.send_state(GameState.WAITING)
                        logger.info("Game reset from TFT")

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
                visualiser.draw_table_boundary(
                    frame, table_corner_pixels, defence_line_px,
                )
                visualiser.draw_detection(frame, detection)
                visualiser.draw_tracking_state(
                    frame, state, mapper.mm_to_pixel
                )

                if locked_prediction is not None:
                    visualiser.draw_trajectory(
                        frame, locked_prediction, mapper.mm_to_pixel
                    )
                    visualiser.draw_prediction_info(
                        frame, locked_prediction
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
        if display is not None:
            display.disconnect()
        if show_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
