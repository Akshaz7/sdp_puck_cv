import cv2
import json
import logging
import numpy as np
from pathlib import Path
from src.models import Position

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        pixel_corners: np.ndarray,
        table_width_mm: float = 610.0,
        table_height_mm: float = 1220.0,
    ):
        """Initialise the mapper.

        Args:
            pixel_corners: Array of shape (4, 2) containing the four table corner
                           positions in pixel coordinates. Order: top-left, top-right,
                           bottom-right, bottom-left. dtype float32.
            table_width_mm: Table width in millimetres.
            table_height_mm: Table height in millimetres.

        The constructor computes the homography matrix using cv2.getPerspectiveTransform.
        """
        self._pixel_corners = np.array(pixel_corners, dtype=np.float32)
        self._table_width = table_width_mm
        self._table_height = table_height_mm

        self._world_corners = np.array(
            [
                [0, 0],
                [table_width_mm, 0],
                [table_width_mm, table_height_mm],
                [0, table_height_mm],
            ],
            dtype=np.float32,
        )

        self._h_matrix = cv2.getPerspectiveTransform(
            self._pixel_corners, self._world_corners
        )
        self._h_matrix_inv = cv2.getPerspectiveTransform(
            self._world_corners, self._pixel_corners
        )

        logger.info("Homography matrix computed")

    def pixel_to_mm(self, position_px: Position) -> Position:
        """Convert a pixel position to real-world millimetres.

        Args:
            position_px: Position in pixel coordinates.

        Returns:
            Position in millimetre coordinates. Timestamp is preserved.
        """
        pt = np.array(
            [[[position_px.x, position_px.y]]], dtype=np.float32
        )
        transformed = cv2.perspectiveTransform(pt, self._h_matrix)
        x_mm = float(transformed[0][0][0])
        y_mm = float(transformed[0][0][1])
        return Position(x=x_mm, y=y_mm, timestamp=position_px.timestamp)

    def mm_to_pixel(self, position_mm: Position) -> Position:
        """Convert a real-world mm position back to pixel coordinates.

        Args:
            position_mm: Position in millimetre coordinates.

        Returns:
            Position in pixel coordinates. Timestamp is preserved.
        """
        pt = np.array(
            [[[position_mm.x, position_mm.y]]], dtype=np.float32
        )
        transformed = cv2.perspectiveTransform(pt, self._h_matrix_inv)
        x_px = float(transformed[0][0][0])
        y_px = float(transformed[0][0][1])
        return Position(x=x_px, y=y_px, timestamp=position_mm.timestamp)


def run_corner_calibration(
    source: int | str = 0,
    config_path: str = "config/settings.json",
) -> np.ndarray:
    """Interactive tool to select the four table corners in a camera frame.

    Opens a window showing the camera feed. The user clicks the four corners
    of the table playing surface in order: top-left, top-right, bottom-right,
    bottom-left.

    Each click is marked with a circle and numbered label.
    After 4 clicks, the corners are saved to config/settings.json under
    table.corners_px.

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
    from src.camera import Camera

    corners: list[tuple[int, int]] = []
    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            logger.info(
                f"Corner {len(corners)}: ({x}, {y}) - "
                f"{labels[len(corners) - 1]}"
            )

    cam = Camera(source=source)
    cv2.namedWindow("Corner Calibration")
    cv2.setMouseCallback("Corner Calibration", mouse_callback)

    result = np.zeros((4, 2), dtype=np.float32)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                break

            display = frame.copy()

            for i, (cx, cy) in enumerate(corners):
                cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"{i + 1}: {labels[i]}",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            if len(corners) >= 2:
                for i in range(len(corners) - 1):
                    cv2.line(
                        display, corners[i], corners[i + 1],
                        (0, 255, 0), 2,
                    )
                if len(corners) == 4:
                    cv2.line(
                        display, corners[3], corners[0],
                        (0, 255, 0), 2,
                    )

            status = f"Click corner {len(corners) + 1}/4"
            if len(corners) == 4:
                status = "Press ENTER to save, 'r' to reset"
            cv2.putText(
                display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("Corner Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z") and corners:
                corners.pop()
            elif key == ord("r"):
                corners.clear()
            elif key == 13 and len(corners) == 4:
                result = np.array(corners, dtype=np.float32)
                path = Path(config_path)
                with open(path, "r") as f:
                    config = json.load(f)
                config["table"]["corners_px"] = [
                    [int(c[0]), int(c[1])] for c in corners
                ]
                with open(path, "w") as f:
                    json.dump(config, f, indent=4)
                logger.info(f"Saved corners to {config_path}")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return result


if __name__ == "__main__":
    run_corner_calibration()
