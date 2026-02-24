import cv2
import logging
import numpy as np
from src.models import Detection, TrackingState, Prediction, Position

logger = logging.getLogger(__name__)


class Visualiser:
    """Draws debug overlays on camera frames.

    All methods modify the frame in-place and return it for chaining.
    """

    def __init__(
        self,
        show_fps: bool = True,
        show_trajectory: bool = True,
    ):
        """Initialise the visualiser.

        Args:
            show_fps: Whether to draw FPS counter.
            show_trajectory: Whether to draw predicted trajectory lines.
        """
        self._show_fps = show_fps
        self._show_trajectory = show_trajectory

    def draw_detection(
        self, frame: np.ndarray, detection: Detection
    ) -> np.ndarray:
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
        if detection.found and detection.position_px is not None:
            cx = int(detection.position_px.x)
            cy = int(detection.position_px.y)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            if detection.bounding_rect is not None:
                x, y, w, h = detection.bounding_rect
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                text = f"PUCK ({cx}, {cy})"
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
        else:
            cv2.putText(
                frame,
                "NO PUCK",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return frame

    def draw_trajectory(
        self,
        frame: np.ndarray,
        prediction: Prediction,
        mm_to_pixel_func: callable,
    ) -> np.ndarray:
        """Draw predicted trajectory on frame.

        Draws lines connecting trajectory_points in yellow (0, 255, 255).
        Draws a red circle at the interception point.
        Draws text showing time_to_intercept.

        Each point must be converted from mm to pixel using mm_to_pixel_func.

        Args:
            frame: BGR image.
            prediction: Trajectory prediction.
            mm_to_pixel_func: Function that converts Position (mm) to
                              Position (px).

        Returns:
            The same frame.
        """
        if not self._show_trajectory:
            return frame

        if not prediction.trajectory_points:
            return frame

        pixel_points: list[tuple[int, int]] = []
        for pt in prediction.trajectory_points:
            px = mm_to_pixel_func(pt)
            pixel_points.append((int(px.x), int(px.y)))

        for i in range(1, len(pixel_points)):
            cv2.line(
                frame, pixel_points[i - 1], pixel_points[i],
                (0, 255, 255), 2,
            )

        intercept_mm = Position(
            x=prediction.interception_x, y=prediction.interception_y
        )
        intercept_px = mm_to_pixel_func(intercept_mm)
        cv2.circle(
            frame,
            (int(intercept_px.x), int(intercept_px.y)),
            8,
            (0, 0, 255),
            -1,
        )

        text = f"T: {prediction.time_to_intercept:.2f}s"
        cv2.putText(
            frame,
            text,
            (int(intercept_px.x) + 10, int(intercept_px.y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        return frame

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
        if not self._show_fps:
            return frame

        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )

        h, w = frame.shape[:2]
        x = w - text_w - 10
        y = 25

        cv2.rectangle(
            frame,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame, text, (x, y), font, scale, (255, 255, 255), thickness
        )

        return frame

    def draw_tracking_state(
        self,
        frame: np.ndarray,
        state: TrackingState,
        mm_to_pixel_func: callable,
    ) -> np.ndarray:
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
        if not state.positions:
            return frame

        n = len(state.positions)
        for i, pos in enumerate(state.positions):
            px = mm_to_pixel_func(pos)
            brightness = int(100 + (155 * i / max(n - 1, 1)))
            colour = (brightness, brightness, brightness)
            cv2.circle(frame, (int(px.x), int(px.y)), 4, colour, -1)

        if (
            state.velocity is not None
            and state.current_position is not None
        ):
            current_px = mm_to_pixel_func(state.current_position)
            arrow_scale = 0.05
            end_mm = Position(
                x=state.current_position.x
                + state.velocity.vx * arrow_scale,
                y=state.current_position.y
                + state.velocity.vy * arrow_scale,
            )
            end_px = mm_to_pixel_func(end_mm)
            cv2.arrowedLine(
                frame,
                (int(current_px.x), int(current_px.y)),
                (int(end_px.x), int(end_px.y)),
                (255, 0, 0),
                2,
                tipLength=0.3,
            )

        return frame
