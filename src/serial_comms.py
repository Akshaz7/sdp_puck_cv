import logging
import serial
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ArduinoComms:
    """Non-blocking serial communication with the Arduino.

    Protocol:
        Send: "X{x}Y{y}\\n" where x, y are target positions in mm (integers).
        Example: "X305Y1100\\n" to move to centre of defence line.

        Also supports X-only: "X{x}\\n" (Y keeps its last target).

        Receive: Arduino sends "OK\\n" after parsing command (not waited for).

    The class runs a background thread for reading responses.
    Writing is done from the main thread.
    Messages are sent at most once every min_send_interval seconds
    to avoid flooding the Arduino.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        min_send_interval: float = 0.02,
        timeout: float = 0.1,
    ):
        """Initialise serial connection.

        Args:
            port: Serial port path.
            baud_rate: Baud rate. Must match Arduino sketch.
            min_send_interval: Minimum seconds between consecutive sends.
            timeout: Serial read timeout in seconds.
        """
        self._port = port
        self._baud_rate = baud_rate
        self._min_send_interval = min_send_interval
        self._timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_send_time: float = 0.0
        self._last_response: Optional[str] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Open the serial connection and start the reader thread.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud_rate,
                timeout=self._timeout,
            )
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True
            )
            self._reader_thread.start()
            logger.info(
                f"Connected to Arduino on {self._port} "
                f"at {self._baud_rate} baud"
            )
            return True
        except (serial.SerialException, OSError) as e:
            logger.error(f"Failed to connect to {self._port}: {e}")
            self._serial = None
            return False

    def _reader_loop(self) -> None:
        """Background thread that reads responses from Arduino."""
        while self._running and self._serial is not None:
            try:
                if self._serial.in_waiting > 0:
                    line = self._serial.readline().decode("utf-8").strip()
                    if line:
                        with self._lock:
                            self._last_response = line
                        logger.debug(f"Arduino response: {line}")
                else:
                    time.sleep(0.01)
            except (serial.SerialException, OSError):
                if self._running:
                    logger.error("Serial read error")
                break

    def send_target(self, x_mm: float, y_mm: float | None = None) -> bool:
        """Send a target position to the Arduino.

        Values are rounded to the nearest integer.
        Skips sending if min_send_interval has not elapsed since last send.

        Args:
            x_mm: Target x position in millimetres.
            y_mm: Target y position in millimetres. If None, sends X-only
                  command and Arduino keeps its current Y target.

        Returns:
            True if message was sent, False if skipped or connection error.
        """
        if self._serial is None or not self._serial.is_open:
            return False

        now = time.time()
        if now - self._last_send_time < self._min_send_interval:
            return False

        try:
            x_val = round(x_mm)
            if y_mm is not None:
                y_val = round(y_mm)
                message = f"X{x_val}Y{y_val}\n"
            else:
                message = f"X{x_val}\n"
            self._serial.write(message.encode("utf-8"))
            self._last_send_time = now
            logger.debug(f"Sent: {message.strip()}")
            return True
        except (serial.SerialException, OSError) as e:
            logger.error(f"Serial write error: {e}")
            return False

    def disconnect(self) -> None:
        """Close the serial connection and stop the reader thread."""
        self._running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            except (serial.SerialException, OSError):
                pass
            self._serial = None
        logger.info("Disconnected from Arduino")

    @property
    def is_connected(self) -> bool:
        """Return True if serial port is open and reader thread is alive."""
        return (
            self._serial is not None
            and self._serial.is_open
            and self._reader_thread is not None
            and self._reader_thread.is_alive()
        )

    @property
    def last_response(self) -> Optional[str]:
        """Return the last response received from Arduino, or None."""
        with self._lock:
            return self._last_response
