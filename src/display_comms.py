import logging
import serial
import socket
import threading
import time
from typing import Optional
from src.game_manager import GameMode, GameState

logger = logging.getLogger(__name__)

# Network bridge: CV Pi sends goal events to Screen Pi over UDP
SCREEN_PI_IP = "192.168.105.183"  # slowbro
SCREEN_PI_PORT = 5555


class DisplayComms:
    """Serial communication with the TFT display Arduino + UDP to Screen Pi.

    Protocol (Python → Display / Screen Pi):
        "G:H\\n"          — human scored
        "G:R\\n"          — robot scored
        "S:h:r\\n"        — score update (e.g. "S:3:2\\n")
        "T:mm:ss\\n"      — timer update (e.g. "T:02:45\\n")
        "STATE:X\\n"      — game state (WAITING, PLAYING, FINISHED)
        "WIN:X\\n"        — winner (HUMAN, ROBOT, DRAW)

    Protocol (Display → Python):
        "MODE:FREE\\n"    — mode selected
        "MODE:FIRST5\\n"
        "MODE:FIRST7\\n"
        "MODE:TIMED3\\n"
        "MODE:TIMED5\\n"
        "START\\n"        — start game
        "RESET\\n"        — reset game
    """

    def __init__(
        self,
        port: str = "/dev/cu.usbmodem2201",
        baud_rate: int = 9600,
        timeout: float = 0.1,
        screen_pi_ip: str = SCREEN_PI_IP,
        screen_pi_port: int = SCREEN_PI_PORT,
    ):
        self._port = port
        self._baud_rate = baud_rate
        self._timeout = timeout
        self._serial: Optional[serial.Serial] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._pending_commands: list[str] = []
        # UDP socket for sending to screen Pi
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._screen_pi_addr = (screen_pi_ip, screen_pi_port)

    def connect(self) -> bool:
        """Open serial connection and start reader thread."""
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
            logger.info(f"Display connected on {self._port}")
            return True
        except (serial.SerialException, OSError) as e:
            logger.error(f"Display connection failed: {e}")
            self._serial = None
            return False

    def _reader_loop(self) -> None:
        """Read commands from the display Arduino (touch input)."""
        while self._running and self._serial is not None:
            try:
                if self._serial.in_waiting > 0:
                    line = self._serial.readline().decode("utf-8").strip()
                    if line:
                        with self._lock:
                            self._pending_commands.append(line)
                        logger.debug(f"Display sent: {line}")
                else:
                    time.sleep(0.01)
            except (serial.SerialException, OSError):
                if self._running:
                    logger.error("Display serial read error")
                break

    def get_commands(self) -> list[str]:
        """Return and clear any pending commands from the display."""
        with self._lock:
            cmds = self._pending_commands.copy()
            self._pending_commands.clear()
        return cmds

    def send_goal(self, scorer: str) -> None:
        """Notify display that a goal was scored."""
        code = "H" if scorer == "human" else "R"
        self._send(f"G:{code}")

    def send_score(self, human: int, robot: int) -> None:
        """Send current score to display."""
        self._send(f"S:{human}:{robot}")

    def send_timer(self, seconds: float) -> None:
        """Send timer value to display."""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        self._send(f"T:{mins:02d}:{secs:02d}")

    def send_state(self, state: GameState) -> None:
        """Send game state to display."""
        self._send(f"STATE:{state.value}")

    def send_winner(self, winner: str) -> None:
        """Send winner to display."""
        self._send(f"WIN:{winner.upper()}")

    def _send(self, message: str) -> bool:
        # Always send over UDP to screen Pi
        try:
            self._udp_sock.sendto(
                f"{message}\n".encode("utf-8"), self._screen_pi_addr
            )
            logger.debug(f"UDP -> screen Pi: {message}")
        except OSError as e:
            logger.warning(f"UDP send failed: {e}")

        # Also send over serial if connected
        if self._serial is None or not self._serial.is_open:
            return False
        try:
            self._serial.write(f"{message}\n".encode("utf-8"))
            logger.debug(f"Sent to display: {message}")
            return True
        except (serial.SerialException, OSError) as e:
            logger.error(f"Display write error: {e}")
            return False

    def disconnect(self) -> None:
        """Close the serial connection."""
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
        logger.info("Display disconnected")

    @property
    def is_connected(self) -> bool:
        return (
            self._serial is not None
            and self._serial.is_open
            and self._reader_thread is not None
            and self._reader_thread.is_alive()
        )
