import cv2
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
    ):
        """Initialise camera.

        Args:
            source: Device index (int) for webcam, or file path (str) for video/image.
            width: Desired frame width. Applied to webcam only.
            height: Desired frame height. Applied to webcam only.
            fps: Desired FPS. Applied to webcam only.
        """
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._mode = self._detect_mode(source)
        self._frame: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None

        if self._mode == InputMode.IMAGE:
            self._frame = cv2.imread(str(source))
            if self._frame is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            self._width = self._frame.shape[1]
            self._height = self._frame.shape[0]
            logger.info(f"Loaded image: {source} ({self._width}x{self._height})")
        else:
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {source}")
            if self._mode == InputMode.WEBCAM:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap.set(cv2.CAP_PROP_FPS, fps)
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                f"Opened {self._mode.value} source: {source} "
                f"({self._width}x{self._height})"
            )

    @staticmethod
    def _detect_mode(source: int | str) -> InputMode:
        """Detect input mode from source type."""
        if isinstance(source, int):
            return InputMode.WEBCAM
        source_lower = str(source).lower()
        if source_lower.endswith((".mp4", ".avi", ".mov")):
            return InputMode.VIDEO
        if source_lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            return InputMode.IMAGE
        raise ValueError(f"Cannot determine input mode for source: {source}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame.

        For VIDEO mode: loops back to start when video ends.
        For IMAGE mode: returns the same image every call.

        Returns:
            Tuple of (success: bool, frame: np.ndarray or None).
            Frame is in BGR colour space, dtype uint8.
        """
        if self._mode == InputMode.IMAGE:
            return (True, self._frame.copy())

        ret, frame = self._cap.read()
        if not ret and self._mode == InputMode.VIDEO:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return (ret, frame)

    def release(self) -> None:
        """Release the camera/video resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera released")

    @property
    def mode(self) -> InputMode:
        """Return the current input mode."""
        return self._mode

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return (width, height) of frames."""
        return (self._width, self._height)
