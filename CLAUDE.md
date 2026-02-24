# GOAL-E: Air Hockey Puck Tracking System

## Project Overview

This is the computer vision subsystem for GOAL-E, an autonomous air hockey robot. A camera is mounted overhead looking down at an air hockey table. This system detects a brightly coloured puck, tracks its position across frames, predicts its trajectory (including wall bounces), and sends target interception coordinates to an Arduino via serial. The Arduino controls an H-bot rail system beneath the table that moves a magnetic pusher.

This codebase is developed locally on a laptop and will later be deployed to a Raspberry Pi 4.

---

## System Constants

- Table playing surface: 610mm wide x 1220mm long
- Camera resolution (development): 640x480 at 60fps
- Camera resolution (Pi deployment): 320x240 at 60fps
- Puck: bright neon green, 3D-printed PLA, approximately 50mm diameter
- Robot controls the BOTTOM half of the table (y = 610mm to y = 1220mm)
- Human plays on the TOP half (y = 0mm to y = 610mm)
- Origin (0, 0) is the top-left corner of the table in real-world coordinates
- X axis runs left to right (0 to 610mm)
- Y axis runs top to bottom (0 to 1220mm)
- The robot's goal line is at y = 1220mm
- The human's goal line is at y = 0mm

---

## Tech Stack

- Python 3.10+
- OpenCV (opencv-python >= 4.8.0)
- NumPy (>= 1.24.0)
- pyserial (>= 3.5)
- pytest (>= 7.4.0)

---

## Project Structure

```
sdp_cv/
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── config/
│   └── settings.json
├── src/
│   ├── __init__.py
│   ├── models.py          # Shared data structures
│   ├── config_loader.py   # Load and validate settings.json
│   ├── camera.py          # Camera/video input abstraction
│   ├── detector.py        # Puck detection via HSV thresholding
│   ├── calibration.py     # Interactive HSV calibration tool
│   ├── homography.py      # Pixel-to-millimetre coordinate mapping
│   ├── tracker.py         # Position history and velocity estimation
│   ├── predictor.py       # Trajectory prediction with wall bounces
│   ├── serial_comms.py    # Non-blocking serial to Arduino
│   ├── visualiser.py      # Debug overlay drawing
│   └── main.py            # Main pipeline loop
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_predictor.py
│   ├── test_homography.py
│   └── fixtures/          # Test images go here
└── docs/
    └── calibration.md
```

---

## Shared Data Structures — src/models.py

All modules import data structures from this file. This is the contract between modules.

```python
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Position:
    """A 2D position with timestamp."""
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Velocity:
    """Velocity vector in units per second."""
    vx: float
    vy: float


@dataclass
class Detection:
    """Result of puck detection on a single frame."""
    found: bool
    position_px: Optional[Position] = None  # pixel coordinates
    contour_area: float = 0.0
    bounding_rect: Optional[tuple[int, int, int, int]] = None  # x, y, w, h


@dataclass
class TrackingState:
    """Current tracking state with position history."""
    current_position: Optional[Position] = None  # real-world mm coordinates
    velocity: Optional[Velocity] = None
    positions: list[Position] = field(default_factory=list)
    frames_since_detection: int = 0


@dataclass
class Prediction:
    """Predicted trajectory result."""
    interception_x: float          # mm, where puck will cross robot's defence line
    interception_y: float          # mm, should be close to robot's y position
    time_to_intercept: float       # seconds until puck reaches defence line
    trajectory_points: list[Position] = field(default_factory=list)  # for visualisation
    is_approaching: bool = False   # True if puck is moving towards robot's goal


@dataclass
class HSVRange:
    """HSV colour range for thresholding."""
    lower: tuple[int, int, int] = (35, 100, 100)
    upper: tuple[int, int, int] = (85, 255, 255)
```

---

## Module Specifications

### src/config_loader.py

Loads config/settings.json and returns a dictionary. Validates that all required keys exist.

```python
import json
from pathlib import Path


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
    ...


def get_hsv_range(config: dict) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Extract HSV lower and upper bounds from config.

    Returns:
        Tuple of (lower, upper) HSV bounds as tuples of 3 ints.
    """
    ...


def get_table_dimensions(config: dict) -> tuple[float, float]:
    """Extract table width and height in mm from config.

    Returns:
        Tuple of (width_mm, height_mm).
    """
    ...
```

---

### src/camera.py

Abstracts camera input. Supports three modes: live webcam, video file, and single image. Uses OpenCV VideoCapture internally.

```python
import cv2
import numpy as np
from enum import Enum


class InputMode(Enum):
    WEBCAM = "webcam"
    VIDEO = "video"
    IMAGE = "image"


class Camera:
    """Camera input abstraction.

    Usage:
        cam = Camera(source=0)              # webcam (device index)
        cam = Camera(source="test.mp4")     # video file
        cam = Camera(source="frame.png")    # single image (loops same frame)

    The class auto-detects the input mode based on the source type.
    - int -> WEBCAM
    - string ending in .mp4/.avi/.mov -> VIDEO
    - string ending in .png/.jpg/.jpeg/.bmp -> IMAGE
    """

    def __init__(self, source: int | str = 0, width: int = 640, height: int = 480, fps: int = 60):
        """Initialise camera.

        Args:
            source: Device index (int) for webcam, or file path (str) for video/image.
            width: Desired frame width. Applied to webcam only.
            height: Desired frame height. Applied to webcam only.
            fps: Desired FPS. Applied to webcam only.
        """
        ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame.

        For VIDEO mode: loops back to start when video ends.
        For IMAGE mode: returns the same image every call.

        Returns:
            Tuple of (success: bool, frame: np.ndarray or None).
            Frame is in BGR colour space, dtype uint8.
        """
        ...

    def release(self) -> None:
        """Release the camera/video resource."""
        ...

    @property
    def mode(self) -> InputMode:
        """Return the current input mode."""
        ...

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return (width, height) of frames."""
        ...
```

---

### src/detector.py

Detects the puck in a single BGR frame using HSV colour thresholding. Stateless. No frame history.

```python
import cv2
import numpy as np
from src.models import Detection, Position, HSVRange


class PuckDetector:
    """Detects a coloured puck in a BGR frame using HSV thresholding.

    Pipeline:
        1. Apply Gaussian blur (kernel size from config, must be odd)
        2. Convert BGR to HSV
        3. Apply inRange threshold using HSV bounds
        4. Apply morphological opening (3x3 kernel) to remove noise
        5. Apply morphological closing (3x3 kernel) to fill gaps
        6. Find contours
        7. Filter contours by minimum area
        8. Select the largest valid contour
        9. Compute centroid using cv2.moments
        10. Compute bounding rect using cv2.boundingRect
    """

    def __init__(self, hsv_range: HSVRange, min_contour_area: int = 100, blur_kernel: int = 5):
        """Initialise detector.

        Args:
            hsv_range: HSV colour range for the puck.
            min_contour_area: Minimum contour area in pixels to count as a detection.
            blur_kernel: Gaussian blur kernel size. Must be odd.
        """
        ...

    def detect(self, frame: np.ndarray) -> Detection:
        """Detect the puck in a single frame.

        Args:
            frame: BGR image as numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            Detection result with pixel coordinates if found.
        """
        ...

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the binary threshold mask for a frame. Useful for debugging.

        Args:
            frame: BGR image.

        Returns:
            Binary mask as numpy array, shape (H, W), dtype uint8, values 0 or 255.
        """
        ...

    def update_hsv_range(self, hsv_range: HSVRange) -> None:
        """Update the HSV range at runtime (used by calibration tool)."""
        ...
```

---

### src/calibration.py

Interactive tool for finding the correct HSV range for the puck. Opens two windows: original frame and threshold mask. Provides trackbars for H_min, S_min, V_min, H_max, S_max, V_max. Saves the final values to config/settings.json on exit.

```python
import cv2
import json
from src.camera import Camera
from src.detector import PuckDetector
from src.models import HSVRange


def run_calibration(source: int | str = 0, config_path: str = "config/settings.json") -> HSVRange:
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
    ...


if __name__ == "__main__":
    run_calibration()
```

---

### src/homography.py

Maps pixel coordinates to real-world table coordinates in millimetres using a perspective transform.

```python
import cv2
import numpy as np
from src.models import Position


class HomographyMapper:
    """Maps between pixel coordinates and real-world millimetre coordinates.

    Calibration requires four corner points of the table playing surface,
    specified in both pixel and real-world coordinates.

    The default real-world corners assume:
        top-left:     (0, 0)
        top-right:    (610, 0)
        bottom-right: (610, 1220)
        bottom-left:  (0, 1220)
    """

    def __init__(self, pixel_corners: np.ndarray, table_width_mm: float = 610.0, table_height_mm: float = 1220.0):
        """Initialise the mapper.

        Args:
            pixel_corners: Array of shape (4, 2) containing the four table corner
                           positions in pixel coordinates. Order: top-left, top-right,
                           bottom-right, bottom-left. dtype float32.
            table_width_mm: Table width in millimetres.
            table_height_mm: Table height in millimetres.

        The constructor computes the homography matrix using cv2.getPerspectiveTransform.
        """
        ...

    def pixel_to_mm(self, position_px: Position) -> Position:
        """Convert a pixel position to real-world millimetres.

        Args:
            position_px: Position in pixel coordinates.

        Returns:
            Position in millimetre coordinates. Timestamp is preserved.
        """
        ...

    def mm_to_pixel(self, position_mm: Position) -> Position:
        """Convert a real-world mm position back to pixel coordinates.

        Args:
            position_mm: Position in millimetre coordinates.

        Returns:
            Position in pixel coordinates. Timestamp is preserved.
        """
        ...


def run_corner_calibration(source: int | str = 0, config_path: str = "config/settings.json") -> np.ndarray:
    """Interactive tool to select the four table corners in a camera frame.

    Opens a window showing the camera feed. The user clicks the four corners
    of the table playing surface in order: top-left, top-right, bottom-right, bottom-left.

    Each click is marked with a circle and numbered label.
    After 4 clicks, the corners are saved to config/settings.json under table.corners_px.

    Key bindings:
        'z' - undo last click
        'r' - reset all clicks
        'q' - quit without saving
        ENTER - confirm and save (only available after 4 points selected)

    Args:
        source: Camera source.
        config_path: Path to settings.json.

    Returns:
        Array of shape (4, 2) with pixel corner coordinates.
    """
    ...


if __name__ == "__main__":
    run_corner_calibration()
```

---

### src/tracker.py

Maintains position history and estimates velocity. Uses a sliding window of recent positions.

```python
import numpy as np
from src.models import Position, Velocity, TrackingState


class PuckTracker:
    """Tracks puck position over time and estimates velocity.

    Maintains a rolling history of the last N positions.
    Estimates velocity using linear regression over the smoothing window.
    Handles missed detections by incrementing frames_since_detection.
    After max_missed_frames consecutive misses, resets tracking state.
    """

    def __init__(self, history_length: int = 10, smoothing_window: int = 5, max_missed_frames: int = 15):
        """Initialise the tracker.

        Args:
            history_length: Maximum number of positions to store in history.
            smoothing_window: Number of recent positions used for velocity estimation.
                              Must be >= 2. Uses the last N positions from history.
            max_missed_frames: After this many consecutive frames with no detection,
                               reset the tracking state entirely.
        """
        ...

    def update(self, position: Position | None) -> TrackingState:
        """Update the tracker with a new position (or None if not detected).

        If position is not None:
            - Append to history (drop oldest if at capacity)
            - Reset frames_since_detection to 0
            - Estimate velocity if enough history exists

        If position is None:
            - Increment frames_since_detection
            - If exceeded max_missed_frames, call reset()
            - Velocity remains as last known estimate

        Velocity estimation method:
            Take the last smoothing_window positions from history.
            For each consecutive pair, compute dx/dt and dy/dt where dt is
            the timestamp difference. Average all dx/dt and dy/dt values
            to get the velocity vector. Units: mm/second.

        Returns:
            Current TrackingState.
        """
        ...

    def reset(self) -> None:
        """Clear all history and reset state."""
        ...

    @property
    def state(self) -> TrackingState:
        """Return the current tracking state."""
        ...

    @property
    def has_velocity(self) -> bool:
        """Return True if a velocity estimate is available (need >= 2 positions)."""
        ...
```

---

### src/predictor.py

Predicts where the puck will cross the robot's defence line, accounting for wall bounces.

```python
import numpy as np
from src.models import Position, Velocity, Prediction


class TrajectoryPredictor:
    """Predicts puck trajectory with wall bounce reflections.

    Given a position and velocity, simulates the puck's path forward in time.
    The puck bounces off the left wall (x=0) and right wall (x=table_width).
    Prediction ends when the puck crosses the defence_y line (robot's side)
    or when max_bounces is reached.

    Algorithm:
        1. Start at current position with current velocity.
        2. Compute time to hit each boundary:
           - Left wall (x=0): t = -x / vx (only if vx < 0)
           - Right wall (x=table_width): t = (table_width - x) / vx (only if vx > 0)
           - Defence line (y=defence_y): t = (defence_y - y) / vy (only if vy > 0, puck moving toward robot)
           - Top wall (y=0): t = -y / vy (only if vy < 0, puck moving away from robot)
        3. Take the smallest positive t.
        4. If the nearest boundary is the defence line, record interception point and stop.
        5. If the nearest boundary is a side wall, move to that point, reflect vx (multiply by -1), continue.
        6. If the nearest boundary is the top wall, move to that point, reflect vy (multiply by -1), continue.
        7. Record each waypoint for visualisation.
        8. Repeat until defence line is crossed or max_bounces is reached.

    If the puck is moving away from the robot (vy <= 0), set is_approaching to False
    and return a prediction with the current x position as a default interception.
    """

    def __init__(self, table_width: float = 610.0, table_height: float = 1220.0,
                 defence_y: float = 1100.0, max_bounces: int = 10):
        """Initialise the predictor.

        Args:
            table_width: Table width in mm.
            table_height: Table height in mm.
            defence_y: Y coordinate of the robot's defence line in mm.
                       This is where the robot tries to intercept.
                       Slightly above the goal line at y=1220.
            max_bounces: Maximum number of wall bounces to simulate before giving up.
        """
        ...

    def predict(self, position: Position, velocity: Velocity) -> Prediction:
        """Predict the puck's trajectory from current state.

        Args:
            position: Current puck position in mm coordinates.
            velocity: Current puck velocity in mm/second.

        Returns:
            Prediction with interception point and trajectory waypoints.
            If puck is not approaching, returns is_approaching=False with
            interception_x set to current x position.
        """
        ...

    def _clamp_x(self, x: float) -> float:
        """Clamp x to table bounds [0, table_width]."""
        ...
```

---

### src/serial_comms.py

Non-blocking serial communication with the Arduino. Sends target X position as a simple text command.

```python
import serial
import threading
import time
from typing import Optional


class ArduinoComms:
    """Non-blocking serial communication with the Arduino.

    Protocol:
        Send: "X{value}\\n" where value is the target x position in mm (integer).
        Example: "X305\\n" to move to centre of table.

        Receive: Arduino sends "OK\\n" after reaching position (not waited for).

    The class runs a background thread for reading responses.
    Writing is done from the main thread.
    Messages are sent at most once every min_send_interval seconds
    to avoid flooding the Arduino.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud_rate: int = 115200,
                 min_send_interval: float = 0.02, timeout: float = 0.1):
        """Initialise serial connection.

        Args:
            port: Serial port path.
            baud_rate: Baud rate. Must match Arduino sketch.
            min_send_interval: Minimum seconds between consecutive sends.
            timeout: Serial read timeout in seconds.
        """
        ...

    def connect(self) -> bool:
        """Open the serial connection and start the reader thread.

        Returns:
            True if connected successfully, False otherwise.
        """
        ...

    def send_target(self, x_mm: float) -> bool:
        """Send a target x position to the Arduino.

        The value is rounded to the nearest integer.
        Skips sending if min_send_interval has not elapsed since last send.

        Args:
            x_mm: Target x position in millimetres.

        Returns:
            True if message was sent, False if skipped or connection error.
        """
        ...

    def disconnect(self) -> None:
        """Close the serial connection and stop the reader thread."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return True if serial port is open and reader thread is alive."""
        ...

    @property
    def last_response(self) -> Optional[str]:
        """Return the last response received from Arduino, or None."""
        ...
```

---

### src/visualiser.py

Draws debug overlays on the frame. All drawing is optional and controlled by debug flags.

```python
import cv2
import numpy as np
from src.models import Detection, TrackingState, Prediction, Position


class Visualiser:
    """Draws debug overlays on camera frames.

    All methods modify the frame in-place and return it for chaining.
    """

    def __init__(self, show_fps: bool = True, show_trajectory: bool = True):
        """Initialise the visualiser.

        Args:
            show_fps: Whether to draw FPS counter.
            show_trajectory: Whether to draw predicted trajectory lines.
        """
        ...

    def draw_detection(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """Draw detection result on frame.

        If detected:
            - Green circle (radius 10) at centroid
            - Green bounding rectangle around contour
            - White text "PUCK (x, y)" above bounding rect

        If not detected:
            - Red text "NO PUCK" in top-left corner

        Args:
            frame: BGR image to draw on. Modified in-place.
            detection: Detection result.

        Returns:
            The same frame (modified in-place).
        """
        ...

    def draw_trajectory(self, frame: np.ndarray, prediction: Prediction,
                        mm_to_pixel_func: callable) -> np.ndarray:
        """Draw predicted trajectory on frame.

        Draws lines connecting trajectory_points in yellow (0, 255, 255).
        Draws a red circle at the interception point.
        Draws text showing time_to_intercept.

        Each point must be converted from mm to pixel using mm_to_pixel_func.

        Args:
            frame: BGR image.
            prediction: Trajectory prediction.
            mm_to_pixel_func: Function that converts Position (mm) to Position (px).

        Returns:
            The same frame.
        """
        ...

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter in top-right corner.

        White text on black background rectangle for readability.
        Format: "FPS: {fps:.1f}"

        Args:
            frame: BGR image.
            fps: Current frames per second.

        Returns:
            The same frame.
        """
        ...

    def draw_tracking_state(self, frame: np.ndarray, state: TrackingState,
                            mm_to_pixel_func: callable) -> np.ndarray:
        """Draw position history as fading dots.

        Draws small circles at each historical position. Older positions
        are more transparent (use decreasing brightness from white to grey).

        If velocity is available, draw a short blue arrow from current
        position in the velocity direction.

        Args:
            frame: BGR image.
            state: Current tracking state.
            mm_to_pixel_func: Coordinate conversion function.

        Returns:
            The same frame.
        """
        ...
```

---

### src/main.py

The main pipeline loop. Ties all modules together.

```python
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
import time
import cv2
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
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Arguments:
        --source: Camera source. Integer for webcam index, string for file path.
                  Default: value from config/settings.json camera.source.
        --config: Path to settings.json. Default: config/settings.json.
        --no-serial: Disable Arduino serial communication.
        --no-display: Disable OpenCV window display (headless mode).
    """
    ...


def main() -> None:
    """Run the main tracking pipeline.

    Implementation:
        1. Parse args and load config.
        2. Initialise all modules:
           - Camera(source, width, height, fps)
           - PuckDetector(hsv_range, min_contour_area, blur_kernel)
           - HomographyMapper(pixel_corners from config, table dims)
           - PuckTracker(history_length, smoothing_window)
           - TrajectoryPredictor(table_width, table_height, defence_y)
           - ArduinoComms(port, baud_rate) only if serial enabled
           - Visualiser(show_fps, show_trajectory) only if display enabled
        3. FPS tracking: use a deque of timestamps for last 30 frames.
           FPS = len(deque) / (newest - oldest).
        4. Main loop:
           - Read frame from camera
           - Detect puck
           - Convert pixel coords to mm via homography (if detected)
           - Update tracker
           - Predict trajectory (if velocity available)
           - Send target to Arduino (if approaching and serial enabled)
           - Draw visualisations (if display enabled)
           - Show frame and mask windows
           - Check for 'q' key to quit
        5. Cleanup in finally block:
           - Release camera
           - Disconnect Arduino
           - Destroy OpenCV windows
    """
    ...


if __name__ == "__main__":
    main()
```

---

## config/settings.json

```json
{
    "camera": {
        "source": 0,
        "width": 640,
        "height": 480,
        "fps": 60
    },
    "detection": {
        "hsv_lower": [35, 100, 100],
        "hsv_upper": [85, 255, 255],
        "min_contour_area": 100,
        "blur_kernel_size": 5
    },
    "table": {
        "width_mm": 610,
        "height_mm": 1220,
        "corners_px": [[0, 0], [640, 0], [640, 480], [0, 480]],
        "defence_y_mm": 1100
    },
    "tracker": {
        "history_length": 10,
        "smoothing_window": 5,
        "max_missed_frames": 15
    },
    "predictor": {
        "max_bounces": 10
    },
    "serial": {
        "port": "/dev/ttyUSB0",
        "baud_rate": 115200,
        "min_send_interval": 0.02
    },
    "debug": {
        "show_windows": true,
        "show_fps": true,
        "show_trajectory": true
    }
}
```

---

## Test Specifications

### tests/test_detector.py

Test cases for PuckDetector:

1. **test_detect_green_circle**: Create a 640x480 black image. Draw a green filled circle (BGR: 0, 255, 0) at position (320, 240) with radius 25. Run detect(). Assert found is True. Assert position_px.x is within 2 pixels of 320. Assert position_px.y is within 2 pixels of 240. Assert contour_area > 100.

2. **test_detect_no_puck**: Create a 640x480 black image. Run detect(). Assert found is False. Assert position_px is None.

3. **test_detect_ignores_small_noise**: Create a 640x480 black image. Draw a tiny green dot (radius 2) at (100, 100). Run detect() with min_contour_area=100. Assert found is False.

4. **test_detect_selects_largest**: Create a 640x480 black image. Draw a small green circle (radius 10) at (100, 100) and a large green circle (radius 30) at (400, 300). Run detect(). Assert position_px is closer to (400, 300).

5. **test_get_mask_returns_binary**: Create any frame. Call get_mask(). Assert result shape is (H, W). Assert dtype is uint8. Assert all values are either 0 or 255.

### tests/test_tracker.py

Test cases for PuckTracker:

1. **test_update_single_position**: Update with one position. Assert current_position matches. Assert velocity is None (need >= 2 points). Assert frames_since_detection is 0.

2. **test_velocity_estimation**: Create 5 positions moving right at constant speed: (0, 0, t=0.0), (10, 0, t=0.1), (20, 0, t=0.2), (30, 0, t=0.3), (40, 0, t=0.4). Update tracker with each. Assert velocity.vx is approximately 100 mm/s (within 5%). Assert velocity.vy is approximately 0.

3. **test_missed_frames_increment**: Update with a position, then update with None three times. Assert frames_since_detection is 3.

4. **test_reset_after_max_missed**: Create tracker with max_missed_frames=3. Update with a position, then update with None 4 times. Assert current_position is None. Assert positions list is empty.

5. **test_history_length_limit**: Create tracker with history_length=5. Update with 10 positions. Assert len(state.positions) == 5.

### tests/test_predictor.py

Test cases for TrajectoryPredictor:

1. **test_straight_down**: Position (305, 0), velocity (0, 1000). Predict. Assert interception_x is approximately 305. Assert is_approaching is True. Assert time_to_intercept is approximately 1.1 seconds (defence_y=1100, distance=1100, speed=1000).

2. **test_moving_away**: Position (305, 600), velocity (0, -500). Predict. Assert is_approaching is False.

3. **test_single_wall_bounce**: Position (500, 0), velocity (500, 1000). The puck should hit the right wall (x=610) after 0.22 seconds, then bounce back. Assert is_approaching is True. Assert interception_x is between 0 and 610.

4. **test_stationary_puck**: Position (305, 600), velocity (0, 0). Predict. Assert is_approaching is False.

5. **test_interception_within_bounds**: Run 100 random trajectories with random positions (x in 0-610, y in 0-600) and random velocities (vx in -1000 to 1000, vy in 100 to 2000). For all predictions where is_approaching is True, assert 0 <= interception_x <= 610.

### tests/test_homography.py

Test cases for HomographyMapper:

1. **test_identity_mapping**: Set pixel_corners to [[0,0],[610,0],[610,1220],[0,1220]]. Map pixel (305, 610) to mm. Assert result is approximately (305, 610).

2. **test_corner_mapping**: Map each pixel corner. Assert it maps to the corresponding real-world corner within 1mm tolerance.

3. **test_round_trip**: Map a point pixel to mm, then mm back to pixel. Assert the result matches the original within 1 pixel.

---

## Coding Conventions

- Type hints on all function signatures and return types.
- Google-style docstrings on all public classes and methods.
- No hardcoded values. All constants come from config/settings.json or constructor args.
- Use dataclasses from models.py for all structured data passing between modules.
- Imports use the full path from project root: from src.models import Position.
- No global state. All state lives in class instances.
- All OpenCV window operations are guarded behind debug/display flags.
- Use logging module (level INFO by default) instead of print statements.
- f-strings for string formatting.
- Line length: 100 characters max.
- No wildcard imports.

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Calibrate HSV values (do this first)
python -m src.calibration

# Calibrate table corners
python -m src.homography

# Run the main tracker
python -m src.main

# Run with a video file
python -m src.main --source test_video.mp4

# Run headless (no display)
python -m src.main --no-display --no-serial

# Run tests
pytest tests/ -v
```

---

## Agent Task Assignments

If running parallel agents, each agent should build one or more of these modules. Dependencies are listed so agents know what interfaces to code against.

| Agent | Files | Dependencies |
|-------|-------|-------------|
| Agent 1 | models.py, config_loader.py | None (build first or in parallel, no external deps) |
| Agent 2 | camera.py | None (only uses OpenCV and numpy) |
| Agent 3 | detector.py, calibration.py | models.py (import Position, Detection, HSVRange) |
| Agent 4 | homography.py | models.py (import Position) |
| Agent 5 | tracker.py | models.py (import Position, Velocity, TrackingState) |
| Agent 6 | predictor.py | models.py (import Position, Velocity, Prediction) |
| Agent 7 | serial_comms.py | None (only uses pyserial and threading) |
| Agent 8 | visualiser.py | models.py (import Detection, TrackingState, Prediction, Position) |
| Agent 9 | main.py | All of the above (build last, or stub imports initially) |
| Agent 10 | All test files | models.py and the module being tested |

Models.py has the exact code above. All agents must copy it verbatim. All agents must use the exact class and method signatures defined in this document. Do not rename classes, methods, parameters, or change return types. The interfaces are frozen.