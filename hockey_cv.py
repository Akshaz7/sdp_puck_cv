"""
Air Hockey Defensive AI  —  hockey_cv.py
Camera : Logitech HD 720p  (CAMERA_INDEX = 1 for external USB)
Serial : "x,y\\n"  →  Arduino CoreXY pusher

TABLE ORIENTATION  (landscape — longer than it is wide)
  ┌─────────────────────────────────────────────────┐
  │  LONG WALL (top)             length = TABLE_W   │
  │                                                 │
  │ SHORT    │  center  │  defense │     SHORT       │
  │ WALL     │   line   │   line   │     WALL        │
  │ x=0      │ x=TW/2   │  x=?     │   x=TABLE_W    │
  │  GOAL    │          │          │     GOAL        │
  │                                                 │
  │  LONG WALL (bottom)          length = TABLE_W   │
  └─────────────────────────────────────────────────┘

  X runs along the LONG axis  (left→right,  0 to TABLE_W)
  Y runs along the SHORT axis (top→bottom,  0 to TABLE_H)

  Goals        : centered on the SHORT sides (x=0 and x=TABLE_W)
                 slit spans TABLE_H/2 ± GOAL_HALF along Y
  Defense line : perpendicular to the SHORT sides (goal walls)
                 = a vertical line at some X value
                 Drawn in BLUE to match the blue acrylic line on
                 the physical table. Turns red in emergency mode.
                 Re-clickable at any time.
  Center line  : thin grey line at x = TABLE_W/2  (matches the
                 white acrylic center marking on the table)

WINDOWS
  "Camera"  — single gameplay window.
              All overlays (border, goals, defense line, center
              line, trail, prediction, scoreboard) are projected
              back onto the raw camera frame via inverse
              perspective transform, so they follow the camera angle.
  "Mask"    — HSV binary detection mask (debug).

CALIBRATION  (all clicks in the Camera window)
  Step 1 – Click the 4 table corners  TL → TR → BR → BL
  Step 2 – Click anywhere inside the table to place the defense
             line.  The blue vertical line appears at that X.
             Re-click during gameplay to reposition it.
"""

import cv2
import logging
import numpy as np
import serial
import threading
import time
from collections import deque
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hockey")

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────

IS_PI             = True          # True = Pi/Linux  |  False = Windows laptop
CAMERA_INDEX      = 0             # Pi: USB webcam always index 0
                                  # Windows laptop: 0=built-in, 1=external USB
FRAME_W           = 640           # Pi 3: keep at 640x480 — 720p will stall
FRAME_H           = 480
WARMUP_FRAMES     = 10

# Warped table dimensions — LANDSCAPE orientation
# TABLE_W is along the LONG axis, TABLE_H along the SHORT axis
# These must match Arduino MAXX and MAXY respectively
TABLE_W           = 600           # long axis  (left short wall → right short wall)
TABLE_H           = 300           # short axis (top long wall   → bottom long wall)

GREEN_LOW         = np.array([30,  60,  60])
GREEN_HIGH        = np.array([95, 255, 255])

MIN_PUCK_AREA     = 80
PUCK_HISTORY      = 14

SEND_HZ           = 40
SERIAL_PORT       = "/dev/ttyUSB0"  # Pi/Linux: ttyUSB0 or ttyACM0
                                    # Windows:  COM5 (or whichever COMx)
BAUD_RATE         = 115200

# Goal slits are on the SHORT sides (left x=0, right x=TABLE_W)
# Goal opening spans TABLE_H/2 ± GOAL_HALF along the Y axis
GOAL_HALF         = 40            # half-height of goal opening in table px
GOAL_COOLDOWN_F   = 45

EMERGENCY_SPEED   = 600           # px/sec
FRICTION          = 0.98          # per 1ms simulation step
DEFENSE_ZONE_HALF = 30            # visual band around pusher target
DEAD_ZONE         = 5             # ignore Y moves smaller than this

# ──────────────────────────────────────────────────────────────
#  THREADED ARDUINO COMMS
# ──────────────────────────────────────────────────────────────

class ArduinoComms:
    """
    Non-blocking 2-D serial link.
    Send:    "x,y\\n"  — table-space integers
    Receive: drained by a background daemon thread.
    Writes rate-limited to SEND_HZ.
    Falls back silently if port unavailable.
    """

    def __init__(
        self,
        port: str                = SERIAL_PORT,
        baud_rate: int           = BAUD_RATE,
        min_send_interval: float = 1.0 / SEND_HZ,
        timeout: float           = 0.1,
    ):
        self._port               = port
        self._baud_rate          = baud_rate
        self._min_send_interval  = min_send_interval
        self._timeout            = timeout
        self._serial: Optional[serial.Serial]          = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running            = False
        self._last_send_time     = 0.0
        self._last_response: Optional[str] = None
        self._lock               = threading.Lock()

    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self._port, baudrate=self._baud_rate, timeout=self._timeout)
            time.sleep(2)           # wait for Arduino DTR reset
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            logger.info(f"Arduino on {self._port} @ {self._baud_rate}")
            return True
        except (serial.SerialException, OSError) as e:
            logger.warning(f"No Arduino ({e}) — vision-only mode")
            self._serial = None
            return False

    def disconnect(self) -> None:
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=1.0)
        if self._serial:
            try:   self._serial.close()
            except (serial.SerialException, OSError): pass
            self._serial = None

    def _reader_loop(self) -> None:
        while self._running and self._serial:
            try:
                if self._serial.in_waiting > 0:
                    line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        with self._lock:
                            self._last_response = line
                        logger.debug(f"Arduino: {line}")
                else:
                    time.sleep(0.005)
            except (serial.SerialException, OSError):
                if self._running:
                    logger.error("Serial read error — cable unplugged?")
                break

    def send_target(self, x: float, y: float) -> bool:
        if self._serial is None or not self._serial.is_open:
            return False
        now = time.time()
        if now - self._last_send_time < self._min_send_interval:
            return False
        try:
            self._serial.write(f"{round(x)},{round(y)}\n".encode())
            self._last_send_time = now
            return True
        except (serial.SerialException, OSError) as e:
            logger.error(f"Serial write error: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return (self._serial is not None and self._serial.is_open
                and self._reader_thread is not None
                and self._reader_thread.is_alive())

# ──────────────────────────────────────────────────────────────
#  CALIBRATION GLOBALS
# ──────────────────────────────────────────────────────────────

corner_points    = []
defense_line_x   = None          # X in table space  (along the long axis)
transform_matrix = None          # camera → table
inv_transform    = None          # table  → camera

STATE_CORNERS = "corners"
STATE_DEFENSE = "defense"
STATE_RUNNING = "running"
state         = STATE_CORNERS


# ── Projection helpers ────────────────────────────────────────

def _to_cam(pt_table):
    """Single table-space point → camera pixel."""
    if inv_transform is None:
        return (int(pt_table[0]), int(pt_table[1]))
    p = np.array([[[float(pt_table[0]), float(pt_table[1])]]], dtype=np.float32)
    r = cv2.perspectiveTransform(p, inv_transform)
    return (int(r[0][0][0]), int(r[0][0][1]))


def _to_cam_many(pts):
    """List of table-space points → list of camera pixels (batch)."""
    if inv_transform is None or not pts:
        return [(int(p[0]), int(p[1])) for p in pts]
    arr = np.array([[list(p)] for p in pts], dtype=np.float32)
    res = cv2.perspectiveTransform(arr, inv_transform)
    return [(int(r[0][0]), int(r[0][1])) for r in res]


def mouse_click(event, x, y, flags, param):
    global corner_points, defense_line_x, transform_matrix, inv_transform, state

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state == STATE_CORNERS:
        corner_points.append((x, y))
        logger.info(f"Corner {len(corner_points)}: ({x},{y})")
        if len(corner_points) == 4:
            src = np.float32(corner_points)
            # dst maps the clicked corners to the landscape table rectangle
            # TL→TR→BR→BL in camera = (0,0)→(TABLE_W,0)→(TABLE_W,TABLE_H)→(0,TABLE_H)
            dst = np.float32([
                [0,       0      ],
                [TABLE_W, 0      ],
                [TABLE_W, TABLE_H],
                [0,       TABLE_H],
            ])
            transform_matrix = cv2.getPerspectiveTransform(src, dst)
            inv_transform    = cv2.getPerspectiveTransform(dst, src)
            logger.info("Perspective calibrated. Click inside table to set defense line.")
            state = STATE_DEFENSE

    elif state in (STATE_DEFENSE, STATE_RUNNING):
        if transform_matrix is None:
            return
        # Project camera click → table space, read the X (long-axis) coordinate
        p  = np.array([[[float(x), float(y)]]], dtype=np.float32)
        tp = cv2.perspectiveTransform(p, transform_matrix)
        tx = int(tp[0][0][0])
        if 0 < tx < TABLE_W:                    # must land inside the table
            defense_line_x = tx
            logger.info(f"Defense line → table x={defense_line_x}")
            if state == STATE_DEFENSE:
                state = STATE_RUNNING


# ──────────────────────────────────────────────────────────────
#  HOCKEY AI
# ──────────────────────────────────────────────────────────────

class HockeyAI:

    def __init__(self):
        # CAP_DSHOW is Windows-only. On Pi/Linux use default backend (V4L2).
        backend = 0 if IS_PI else cv2.CAP_DSHOW
        self.cam = cv2.VideoCapture(CAMERA_INDEX, backend)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cam.set(cv2.CAP_PROP_FPS,          60)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE,    1)

        logger.info(f"Warm-up ({WARMUP_FRAMES} frames)…")
        for _ in range(WARMUP_FRAMES):
            self.cam.read()

        self.arduino = ArduinoComms()
        self.arduino.connect()      # graceful fallback if not present

        self.pos_history  = deque(maxlen=PUCK_HISTORY)
        self.time_history = deque(maxlen=PUCK_HISTORY)

        self.last_target  = None
        self.last_sent_y  = TABLE_H // 2
        self.emergency    = False

        # Robot defends the RIGHT side (high X).
        # Puck escaping through right wall  → player scores.
        # Puck escaping through left  wall  → robot scores.
        self.robot_score  = 0
        self.player_score = 0
        self._goal_cd     = 0

    # ── PUCK DETECTION ───────────────────────────────────────

    def detect_puck(self, table_frame):
        blur = cv2.GaussianBlur(table_frame, (5, 5), 0)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < MIN_PUCK_AREA:
            return None, mask
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None, mask
        return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), mask

    # ── VELOCITY (px/sec, weighted, frame-rate independent) ──

    def velocity(self):
        n = len(self.pos_history)
        if n < 2:
            return 0.0, 0.0
        pts   = list(self.pos_history)
        times = list(self.time_history)
        dx_sum = dy_sum = w_sum = 0.0
        for i in range(n - 1):
            dt = times[i+1] - times[i]
            if dt <= 0:
                continue
            w       = float(i + 1)
            dx_sum += (pts[i+1][0] - pts[i][0]) / dt * w
            dy_sum += (pts[i+1][1] - pts[i][1]) / dt * w
            w_sum  += w
        if w_sum == 0:
            return 0.0, 0.0
        return dx_sum / w_sum, dy_sum / w_sum

    # ── WALL BOUNCE (reflects off the LONG walls, i.e. top/bottom) ──

    def _bounce_y(self, y, vy):
        """
        The long walls are at y=0 (top) and y=TABLE_H (bottom).
        Puck bounces off these walls — Y component reflects.
        The short walls (x=0, x=TABLE_W) are the goals, not bounce walls.
        """
        if y < 0:
            y  = -y
            vy =  abs(vy)
        elif y > TABLE_H:
            y  = TABLE_H - (y - TABLE_H)
            vy = -abs(vy)
        return y, vy

    # ── SIMULATION GENERATOR ─────────────────────────────────

    def _simulate(self, x, y, vx, vy, stop_x=None):
        """
        Simulate puck path 1ms per step with friction + top/bottom bounce.
        Yields (cx, cy) each step.
        Stops when cx reaches stop_x (defense line) or vx decays to zero.
        """
        dt     = 0.001
        svx    = vx * dt
        svy    = vy * dt
        cx, cy = float(x), float(y)
        end_x  = stop_x if stop_x is not None else TABLE_W

        for _ in range(10000):
            cx  += svx
            cy  += svy
            svx *= FRICTION
            svy *= FRICTION
            cy, svy = self._bounce_y(cy, svy)
            yield cx, cy
            if abs(svx) < 1e-4:
                break
            if cx >= end_x:
                break

    # ── PREDICT INTERCEPT AT DEFENSE LINE ────────────────────

    def predict_intercept(self, x, y, vx, vy):
        if defense_line_x is None or vx < 0.1:
            return TABLE_H // 2
        last_cy = float(y)
        for cx, cy in self._simulate(x, y, vx, vy, stop_x=defense_line_x):
            last_cy = cy
            if cx >= defense_line_x:
                return int(np.clip(cy, 0, TABLE_H))
        return int(np.clip(last_cy, 0, TABLE_H))

    # ── DRAW TRAIL (past positions, camera space) ─────────────

    def draw_trail(self, display):
        pts = list(self.pos_history)
        if len(pts) < 2:
            return
        cam_pts = _to_cam_many(pts)
        n = max(len(cam_pts) - 1, 1)
        for i in range(1, len(cam_pts)):
            frac  = i / n
            # old = dim green, new = bright yellow-green
            color = (0, int(160 + 95 * frac), int(255 * (1 - frac * 0.7)))
            cv2.line(display, cam_pts[i-1], cam_pts[i], color, 2, cv2.LINE_AA)

    # ── DRAW PREDICTION PATH (future, camera space) ───────────

    def draw_prediction(self, display, x, y, vx, vy):
        """
        Draws the predicted bounce path from puck's current position
        to the defense line, including all wall reflections.
        Color: bright cyan fading to yellow at the intercept.
        """
        if vx < 0.1 or defense_line_x is None:
            return

        path = [(x, y)]
        for cx, cy in self._simulate(x, y, vx, vy, stop_x=defense_line_x):
            path.append((cx, cy))
            if cx >= defense_line_x:
                break

        if len(path) < 2:
            return

        cam_pts = _to_cam_many(path)
        n = max(len(cam_pts) - 1, 1)

        for i in range(1, len(cam_pts)):
            frac  = i / n
            # cyan (0,255,255) → yellow (0,255,0) as path approaches defense line
            b = int(255 * (1 - frac))
            cv2.line(display, cam_pts[i-1], cam_pts[i],
                     (b, 255, 255), 1, cv2.LINE_AA)

        # Intercept marker — yellow circle at defense line crossing
        cv2.circle(display, cam_pts[-1], 9,  (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(display, cam_pts[-1], 3,  (0, 255, 255), -1)

    # ── GOAL CHECK ────────────────────────────────────────────
    # Goals are on the SHORT sides: x ≤ 4 (left) and x ≥ TABLE_W-4 (right)
    # Goal opening: TABLE_H/2 ± GOAL_HALF  (centered on the short wall)

    def check_goals(self, x, y):
        if self._goal_cd > 0:
            self._goal_cd -= 1
            return
        gt = TABLE_H // 2 - GOAL_HALF
        gb = TABLE_H // 2 + GOAL_HALF

        if gt <= y <= gb:
            if x <= 4:                       # left short wall — robot scores
                self.robot_score += 1
                self._goal_cd     = GOAL_COOLDOWN_F
                self.pos_history.clear()
                self.time_history.clear()
                logger.info(f"GOAL — Robot  {self.robot_score}:{self.player_score}")
            elif x >= TABLE_W - 4:           # right short wall — player scores
                self.player_score += 1
                self._goal_cd      = GOAL_COOLDOWN_F
                self.pos_history.clear()
                self.time_history.clear()
                logger.info(f"GOAL — Player  {self.robot_score}:{self.player_score}")

    # ── SERIAL SEND ──────────────────────────────────────────

    def send(self, x, y):
        if abs(y - self.last_sent_y) < DEAD_ZONE and not self.emergency:
            return
        if self.arduino.send_target(x, y):
            self.last_sent_y = y

    # ── DRAW ALL OVERLAYS ONTO CAMERA FRAME ──────────────────

    def draw_overlays(self, display, puck_table, vx, vy, speed):

        # ── 1. Table border ───────────────────────────────────
        border = _to_cam_many([(0,0),(TABLE_W,0),(TABLE_W,TABLE_H),(0,TABLE_H)])
        cv2.polylines(display, [np.array(border)], isClosed=True,
                      color=(160,160,160), thickness=1, lineType=cv2.LINE_AA)

        # ── 2. Long walls (top & bottom) — thin grey lines ───
        # already covered by border, but emphasise them slightly
        cv2.line(display, border[0], border[1], (120,120,120), 1, cv2.LINE_AA)  # top
        cv2.line(display, border[3], border[2], (120,120,120), 1, cv2.LINE_AA)  # bottom

        # ── 3. Goal openings on SHORT sides ──────────────────
        gt = TABLE_H // 2 - GOAL_HALF
        gb = TABLE_H // 2 + GOAL_HALF

        # Left goal (x=0):  vertical segment from (0,gt) to (0,gb)
        cv2.line(display, _to_cam((0, gt)), _to_cam((0, gb)),
                 (0, 0, 220), 4, cv2.LINE_AA)
        # Right goal (x=TABLE_W):
        cv2.line(display, _to_cam((TABLE_W, gt)), _to_cam((TABLE_W, gb)),
                 (0, 0, 220), 4, cv2.LINE_AA)

        # Semi-transparent orange fill on goal slits
        for gx0, gx1 in [(0, 6), (TABLE_W-6, TABLE_W)]:
            pts_goal = np.array([
                _to_cam((gx0, gt)), _to_cam((gx1, gt)),
                _to_cam((gx1, gb)), _to_cam((gx0, gb))
            ])
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts_goal], (0, 130, 255))
            cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)

        # ── 4. Center line (mid-pitch acrylic marking) ────────
        # Matches the white center line printed on the acrylic sheet.
        mid_x = TABLE_W // 2
        cv2.line(display, _to_cam((mid_x, 0)), _to_cam((mid_x, TABLE_H)),
                 (200, 200, 200), 1, cv2.LINE_AA)

        # ── 5. Defense line — perpendicular to long side ──────
        # Vertical line in table space at defense_line_x.
        # Drawn in BLUE to match the blue acrylic marking on the table.
        # Turns red when in emergency mode.
        # Re-clickable at any time to reposition.
        if defense_line_x is not None:
            # Blue (BGR 200,80,0) matches the blue acrylic line on the table.
            # Switches to bright red during emergency mode.
            dl_color = (0, 40, 255) if self.emergency else (200, 80, 0)
            cv2.line(display,
                     _to_cam((defense_line_x, 0)),
                     _to_cam((defense_line_x, TABLE_H)),
                     dl_color, 2, cv2.LINE_AA)
            lbl_pos = _to_cam((defense_line_x + 4, 10))
            cv2.putText(display, "DEF", lbl_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dl_color, 1)

        # ── 6. Puck trail ─────────────────────────────────────
        self.draw_trail(display)

        # ── 7. Prediction path + intercept marker ─────────────
        if puck_table and defense_line_x is not None:
            self.draw_prediction(display, puck_table[0], puck_table[1], vx, vy)

        # ── 8. Puck dot + velocity arrow ──────────────────────
        if puck_table:
            cam_p = _to_cam(puck_table)
            cv2.circle(display, cam_p, 12, (0, 255, 0),  2, cv2.LINE_AA)
            cv2.circle(display, cam_p,  4, (0, 255, 0), -1)

            if speed > 5:
                arrow_t = (puck_table[0] + vx * 0.05, puck_table[1] + vy * 0.05)
                cv2.arrowedLine(display, cam_p, _to_cam(arrow_t),
                                (0, 255, 255), 2, tipLength=0.35,
                                line_type=cv2.LINE_AA)
            cv2.putText(display, f"{speed:.0f}px/s",
                        (cam_p[0]+14, cam_p[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

        # ── 9. Pusher target band on defense line ─────────────
        if self.last_target:
            tx, ty = self.last_target
            top_c  = _to_cam((tx, max(ty - DEFENSE_ZONE_HALF, 0)))
            bot_c  = _to_cam((tx, min(ty + DEFENSE_ZONE_HALF, TABLE_H)))
            cv2.line(display, top_c, bot_c, (255, 60, 60), 5, cv2.LINE_AA)
            cv2.circle(display, _to_cam((tx, ty)), 6, (255, 0, 0), -1)

        # ── 10. Scoreboard HUD ────────────────────────────────
        cv2.rectangle(display, (8, 8), (250, 72), (0, 0, 0), -1)
        cv2.putText(display, f"Robot:  {self.robot_score}",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2)
        cv2.putText(display, f"Player: {self.player_score}",
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 255), 2)

        # ── 11. Status bar ────────────────────────────────────
        ok  = self.arduino.is_connected
        cv2.putText(display, "SER:OK" if ok else "SER:--",
                    (10, FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 100) if ok else (60, 60, 255), 1)

        if self.emergency:
            cv2.putText(display, "!! EMERGENCY !!",
                        (FRAME_W//2 - 160, FRAME_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # ── MAIN LOOP ────────────────────────────────────────────

    def run(self):
        global state

        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask",   cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Camera", mouse_click)

        logger.info("=== Air Hockey AI ===")
        logger.info("Step 1 — Click 4 corners of the table in the Camera window")
        logger.info("         Order: TL → TR → BR → BL  (clockwise from top-left)")
        logger.info("Step 2 — Click inside the table to place the defense line")
        logger.info("         (Re-click any time during the game to move it)")

        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue

            now     = time.time()
            display = frame.copy()

            # ── Phase 1: corner selection ─────────────────────
            if state == STATE_CORNERS:
                for i, p in enumerate(corner_points):
                    cv2.circle(display, p, 7, (0, 0, 255), -1)
                    cv2.putText(display, str(i+1), (p[0]+9, p[1]-9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if len(corner_points) >= 2:
                    for i in range(len(corner_points)-1):
                        cv2.line(display, corner_points[i], corner_points[i+1],
                                 (0,180,255), 1)
                cv2.putText(display,
                            f"Click corner {len(corner_points)+1}/4  (TL→TR→BR→BL)",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,200,255), 2)
                cv2.imshow("Camera", display)
                cv2.imshow("Mask", np.zeros((TABLE_H, TABLE_W), dtype=np.uint8))
                if cv2.waitKey(1) == 27: break
                continue

            # ── Phase 2: defense line ─────────────────────────
            if state == STATE_DEFENSE:
                border = _to_cam_many([(0,0),(TABLE_W,0),(TABLE_W,TABLE_H),(0,TABLE_H)])
                cv2.polylines(display, [np.array(border)], True, (0,200,255), 2)
                cv2.putText(display,
                            "Click inside the table to place the defense line",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,200,255), 2)
                cv2.imshow("Camera", display)
                cv2.imshow("Mask", np.zeros((TABLE_H, TABLE_W), dtype=np.uint8))
                if cv2.waitKey(1) == 27: break
                continue

            # ── Phase 3: running ──────────────────────────────
            if transform_matrix is None:
                if cv2.waitKey(1) == 27: break
                continue

            table_frame = cv2.warpPerspective(
                frame, transform_matrix, (TABLE_W, TABLE_H))
            puck, mask_img = self.detect_puck(table_frame)

            puck_table = None
            vx = vy = speed = 0.0

            if puck:
                px, py = puck
                self.pos_history.append((px, py))
                self.time_history.append(now)
                vx, vy = self.velocity()
                speed  = np.hypot(vx, vy)
                self.emergency = speed > EMERGENCY_SPEED
                puck_table = (px, py)

                self.check_goals(px, py)

                if defense_line_x is not None:
                    intercept = self.predict_intercept(px, py, vx, vy)
                    target_y  = int(np.clip(intercept,
                                            DEFENSE_ZONE_HALF,
                                            TABLE_H - DEFENSE_ZONE_HALF))
                    # Send only when puck is heading toward robot's side
                    if vx > 0 and px < defense_line_x:
                        self.send(defense_line_x, target_y)
                        self.last_target = (defense_line_x, target_y)

            self.draw_overlays(display, puck_table, vx, vy, speed)

            if defense_line_x is None:
                cv2.putText(display,
                            "Click inside the table to set defense line",
                            (10, FRAME_H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Mask debug window
            mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            if puck:
                cv2.circle(mask_rgb, puck, 10, (0, 255, 0), 2)
            cv2.imshow("Mask",   mask_rgb)
            cv2.imshow("Camera", display)

            if cv2.waitKey(1) == 27:
                break

        self.cam.release()
        cv2.destroyAllWindows()
        self.arduino.disconnect()


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    HockeyAI().run()