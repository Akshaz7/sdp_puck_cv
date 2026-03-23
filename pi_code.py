"""
Air Hockey Defensive AI  —  hockey_cv.py
Camera : Logitech HD 720p  (CAMERA_INDEX = 1 for external USB)
Serial : "x,y\\n"  →  Arduino CoreXY pusher

──────────────────────────────────────────────────────────────
WINDOWS
  "Camera"  — the only gameplay window.
              All overlays (table boundary, goal lines, defense
              line, puck dot, trail, prediction path, scoreboard)
              are projected back onto the raw camera frame.
  "Mask"    — HSV binary mask, debug only.

CALIBRATION  (click in the Camera window)
  Step 1 – Click the 4 table corners  TL → TR → BR → BL
  Step 2 – Click once inside the table where you want the
            defense line.  A perpendicular (vertical in table
            space) white line will be drawn across the full
            table width at that x position.
            You can re-click at any time during running to
            move the defense line.

FEATURES
  • Single-window design — table content projected to camera view
  • Defense line set by clicking anywhere on the table (not hardcoded)
  • Re-clickable defense line at any time during gameplay
  • Puck trail (last N positions, colour-faded green→yellow)
  • Full bounce-prediction path drawn with wall reflections
    (yellow line that bounces off top/bottom and stops at defense line)
  • Intercept dot shown where prediction meets defense line
  • Friction-aware physics simulation (FRICTION per 1 ms step)
  • Emergency mode when speed > EMERGENCY_SPEED
  • Goal detection with per-side score + cooldown + history reset
  • Threaded ArduinoComms — non-blocking serial, drains RX buffer
  • Dead-zone + SEND_HZ rate-limit to prevent Arduino buffer flood
  • SER:OK / SER:-- indicator on screen
──────────────────────────────────────────────────────────────
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

CAMERA_INDEX      = 1          # 0 = built-in  |  1 = external Logitech
FRAME_W           = 1280
FRAME_H           = 720
WARMUP_FRAMES     = 10            # discard first N frames (exposure settle)

TABLE_W           = 600           # warped table width  — MUST match Arduino MAXX
TABLE_H           = 300           # warped table height — MUST match Arduino MAXY

GREEN_LOW         = np.array([30,  60,  60])
GREEN_HIGH        = np.array([95, 255, 255])

MIN_PUCK_AREA     = 80            # px² — reject noise blobs
PUCK_HISTORY      = 14            # positions kept for trail + velocity

SEND_HZ           = 40
SERIAL_PORT       = "COM5"        # Windows COMx  |  Linux /dev/ttyUSB0
BAUD_RATE         = 115200

GOAL_SIZE         = 80            # total goal-slit height (px in table space)
GOAL_COOLDOWN_F   = 45            # frames lockout after a goal
EMERGENCY_SPEED   = 600           # px/sec — triggers emergency mode
FRICTION          = 0.98          # velocity decay per 1 ms prediction step
DEFENSE_ZONE_HALF = 30            # visual band around pusher target (px)
DEAD_ZONE         = 5             # ignore Y nudges smaller than this (px)

# ──────────────────────────────────────────────────────────────
#  THREADED ARDUINO COMMS
# ──────────────────────────────────────────────────────────────

class ArduinoComms:
    """
    Non-blocking 2-D serial link to the Arduino.

    Send:    "x,y\\n"   — table-space integers
    Receive: drained by a daemon reader thread so the TX buffer
             never backs up and blocks the CV loop.
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
        self._serial: Optional[serial.Serial]  = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running            = False
        self._last_send_time     = 0.0
        self._last_response: Optional[str] = None
        self._lock               = threading.Lock()

    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self._port, baudrate=self._baud_rate, timeout=self._timeout)
            time.sleep(2)
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            logger.info(f"Arduino on {self._port} @ {self._baud_rate}")
            return True
        except (serial.SerialException, OSError) as e:
            logger.error(f"Serial connect failed: {e}")
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
                    logger.error("Serial read error")
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
#  CALIBRATION  GLOBALS
# ──────────────────────────────────────────────────────────────

corner_points      = []      # raw camera-frame corners (up to 4)
defense_line_x     = None    # x in TABLE space
transform_matrix   = None    # camera → table
inv_transform      = None    # table → camera  (for projecting overlays back)

STATE_CORNERS  = "corners"
STATE_DEFENSE  = "defense"
STATE_RUNNING  = "running"
state          = STATE_CORNERS


def table_pt_to_camera(pt_table):
    """Map a single (x, y) in table space back to camera-frame pixel."""
    if inv_transform is None:
        return pt_table
    p = np.array([[[float(pt_table[0]), float(pt_table[1])]]], dtype=np.float32)
    r = cv2.perspectiveTransform(p, inv_transform)
    return (int(r[0][0][0]), int(r[0][0][1]))


def table_pts_to_camera(pts):
    """Map a list of (x, y) table-space points → camera-frame pixels."""
    if inv_transform is None or len(pts) == 0:
        return pts
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
            dst = np.float32([[0,0],[TABLE_W,0],[TABLE_W,TABLE_H],[0,TABLE_H]])
            transform_matrix = cv2.getPerspectiveTransform(src, dst)
            inv_transform    = cv2.getPerspectiveTransform(dst, src)
            logger.info("Perspective ready. Click inside the table to set defense line.")
            state = STATE_DEFENSE

    elif state in (STATE_DEFENSE, STATE_RUNNING):
        # Click is in camera space — project to table space to get the x coord
        if transform_matrix is None:
            return
        p    = np.array([[[float(x), float(y)]]], dtype=np.float32)
        tp   = cv2.perspectiveTransform(p, transform_matrix)
        tx   = int(tp[0][0][0])
        # Only accept clicks that land inside the table
        if 0 <= tx <= TABLE_W:
            defense_line_x = tx
            logger.info(f"Defense line set at table-x={defense_line_x}")
            if state == STATE_DEFENSE:
                state = STATE_RUNNING

# ──────────────────────────────────────────────────────────────
#  HOCKEY AI
# ──────────────────────────────────────────────────────────────

class HockeyAI:

    def __init__(self):
        self.cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cam.set(cv2.CAP_PROP_FPS,          60)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE,    1)

        logger.info(f"Camera warm-up ({WARMUP_FRAMES} frames)…")
        for _ in range(WARMUP_FRAMES):
            self.cam.read()

        self.arduino = ArduinoComms()
        if not self.arduino.connect():
            logger.warning("No Arduino — vision-only mode")

        self.pos_history  = deque(maxlen=PUCK_HISTORY)
        self.time_history = deque(maxlen=PUCK_HISTORY)

        self.last_target  = None      # (table_x, table_y)
        self.last_sent_y  = TABLE_H // 2
        self.emergency    = False

        self.robot_score  = 0
        self.player_score = 0
        self._goal_cd     = 0

    # ── PUCK DETECTION ───────────────────────────────────────

    def detect_puck(self, table_frame):
        """Detect green puck in warped table frame. Returns ((cx,cy), mask)."""
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
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), mask

    # ── VELOCITY  (px/sec, weighted, frame-rate independent) ─

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

    # ── WALL BOUNCE ──────────────────────────────────────────

    def _bounce_y(self, y, vy):
        if y < 0:
            y  = -y;                vy =  abs(vy)
        elif y > TABLE_H:
            y  = TABLE_H - (y - TABLE_H);  vy = -abs(vy)
        return y, vy

    # ── SIMULATION  (shared by predict + draw) ───────────────

    def _simulate(self, x, y, vx, vy, stop_x=None, max_steps=10000):
        """
        Simulate puck path with friction + wall bounce.
        Yields (table_x, table_y) each step until stop_x is reached
        or vx decays to nearly zero.
        stop_x defaults to TABLE_W (far wall).
        """
        dt     = 0.001
        svx    = vx * dt
        svy    = vy * dt
        cx, cy = float(x), float(y)
        end_x  = stop_x if stop_x is not None else TABLE_W

        for _ in range(max_steps):
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

    # ── PREDICT INTERCEPT ────────────────────────────────────

    def predict_intercept(self, x, y, vx, vy):
        if defense_line_x is None or vx < 0.1:
            return TABLE_H // 2
        for cx, cy in self._simulate(x, y, vx, vy, stop_x=defense_line_x):
            if cx >= defense_line_x:
                return int(np.clip(cy, 0, TABLE_H))
        return TABLE_H // 2

    # ── DRAW TRAIL  (past positions projected to camera) ─────

    def draw_trail(self, display):
        pts = list(self.pos_history)
        if len(pts) < 2:
            return
        cam_pts = table_pts_to_camera(pts)
        n = max(len(cam_pts) - 1, 1)
        for i in range(1, len(cam_pts)):
            # fade from dim green (old) → bright yellow-green (new)
            frac  = i / n
            color = (0, int(180 + 75 * frac), int(255 * (1 - frac)))
            cv2.line(display, cam_pts[i-1], cam_pts[i], color, 2)

    # ── DRAW PREDICTION PATH  (future path projected to camera)

    def draw_prediction(self, display, x, y, vx, vy):
        """
        Draw the predicted bounce path on the camera frame.
        Yellow line that reflects off top/bottom walls all the way
        to the defense line (or until friction stops it).
        Also marks the intercept point with a cyan circle.
        """
        if vx < 0.1 or defense_line_x is None:
            return

        # Collect table-space path points
        path_table = [(x, y)]
        for cx, cy in self._simulate(x, y, vx, vy, stop_x=defense_line_x):
            path_table.append((cx, cy))
            if cx >= defense_line_x:
                break

        if len(path_table) < 2:
            return

        # Project all at once for efficiency
        cam_pts = table_pts_to_camera(path_table)

        # Draw path segments — colour transitions yellow → orange
        n = max(len(cam_pts) - 1, 1)
        for i in range(1, len(cam_pts)):
            frac  = i / n
            color = (0, int(220 * (1 - frac * 0.5)), 255)   # cyan→yellow-ish
            cv2.line(display, cam_pts[i-1], cam_pts[i], color, 1, cv2.LINE_AA)

        # Intercept dot at defense line
        intercept_cam = cam_pts[-1]
        cv2.circle(display, intercept_cam, 8, (255, 255, 0), 2)   # yellow ring
        cv2.circle(display, intercept_cam, 3, (255, 255, 0), -1)

    # ── PROJECT A VERTICAL TABLE LINE → CAMERA ───────────────

    def draw_table_vline(self, display, table_x, color, thickness=2):
        """Draw a vertical line at table_x across full table height on camera frame."""
        top    = table_pt_to_camera((table_x, 0))
        bottom = table_pt_to_camera((table_x, TABLE_H))
        cv2.line(display, top, bottom, color, thickness, cv2.LINE_AA)

    # ── PROJECT A HORIZONTAL TABLE LINE → CAMERA ─────────────

    def draw_table_hline(self, display, table_y, x0, x1, color, thickness=2):
        p0 = table_pt_to_camera((x0, table_y))
        p1 = table_pt_to_camera((x1, table_y))
        cv2.line(display, p0, p1, color, thickness, cv2.LINE_AA)

    # ── GOAL CHECK ───────────────────────────────────────────

    def check_goals(self, x, y):
        if self._goal_cd > 0:
            self._goal_cd -= 1
            return
        half = GOAL_SIZE // 2
        gt   = TABLE_H // 2 - half
        gb   = TABLE_H // 2 + half
        if gt <= y <= gb:
            if x <= 4:
                self.robot_score += 1
                self._goal_cd     = GOAL_COOLDOWN_F
                self.pos_history.clear()
                self.time_history.clear()
                logger.info(f"GOAL — Robot  {self.robot_score}:{self.player_score}")
            elif x >= TABLE_W - 4:
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
        """
        Project every overlay — table border, goal lines, defense line,
        puck dot, trail, prediction, target marker, scoreboard —
        onto the camera frame using the inverse perspective transform.
        """
        half = GOAL_SIZE // 2
        gt   = TABLE_H // 2 - half
        gb   = TABLE_H // 2 + half

        # ── Table border (thin white quad) ───────────────────
        border_table = [(0,0),(TABLE_W,0),(TABLE_W,TABLE_H),(0,TABLE_H)]
        border_cam   = table_pts_to_camera(border_table)
        cv2.polylines(display, [np.array(border_cam)], isClosed=True,
                      color=(180,180,180), thickness=1, lineType=cv2.LINE_AA)

        # ── Goal lines (red) on left and right edges ─────────
        self.draw_table_hline(display, gt, 0, 0,       (0,0,255), 3)   # left top post
        self.draw_table_hline(display, gb, 0, 0,       (0,0,255), 3)   # left bot post
        self.draw_table_hline(display, gt, TABLE_W, TABLE_W, (0,0,255), 3)
        self.draw_table_hline(display, gb, TABLE_W, TABLE_W, (0,0,255), 3)

        # left goal slit  (vertical segment on x=0)
        p_gt_l = table_pt_to_camera((0, gt))
        p_gb_l = table_pt_to_camera((0, gb))
        cv2.line(display, p_gt_l, p_gb_l, (0, 0, 255), 3, cv2.LINE_AA)

        # right goal slit (vertical segment on x=TABLE_W)
        p_gt_r = table_pt_to_camera((TABLE_W, gt))
        p_gb_r = table_pt_to_camera((TABLE_W, gb))
        cv2.line(display, p_gt_r, p_gb_r, (0, 0, 255), 3, cv2.LINE_AA)

        # ── Defense line (white / red in emergency) ──────────
        if defense_line_x is not None:
            lc = (0, 80, 255) if self.emergency else (255, 255, 255)
            self.draw_table_vline(display, defense_line_x, lc, 2)

        # ── Puck trail ───────────────────────────────────────
        self.draw_trail(display)

        # ── Prediction path + intercept ──────────────────────
        if puck_table and defense_line_x is not None:
            px, py = puck_table
            self.draw_prediction(display, px, py, vx, vy)

        # ── Puck dot ─────────────────────────────────────────
        if puck_table:
            cam_puck = table_pt_to_camera(puck_table)
            cv2.circle(display, cam_puck, 12, (0, 255, 0),  2, cv2.LINE_AA)
            cv2.circle(display, cam_puck,  4, (0, 255, 0), -1)

            # Velocity arrow
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                arrow_end_table = (puck_table[0] + vx * 0.05,
                                   puck_table[1] + vy * 0.05)
                ae_cam = table_pt_to_camera(arrow_end_table)
                cv2.arrowedLine(display, cam_puck, ae_cam,
                                (0, 255, 255), 2, tipLength=0.35, line_type=cv2.LINE_AA)

            cv2.putText(display, f"{speed:.0f}px/s",
                        (cam_puck[0] + 14, cam_puck[1] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

        # ── Pusher target band ───────────────────────────────
        if self.last_target:
            tx, ty = self.last_target
            top_cam = table_pt_to_camera((tx, max(ty - DEFENSE_ZONE_HALF, 0)))
            bot_cam = table_pt_to_camera((tx, min(ty + DEFENSE_ZONE_HALF, TABLE_H)))
            cv2.line(display, top_cam, bot_cam, (255, 60, 60), 6, cv2.LINE_AA)
            mid_cam = table_pt_to_camera((tx, ty))
            cv2.circle(display, mid_cam, 7, (255, 0, 0), -1)

        # ── Scoreboard (HUD, top-left corner of camera frame) ─
        cv2.rectangle(display, (8, 8), (260, 68), (0, 0, 0), -1)
        cv2.putText(display, f"Robot:  {self.robot_score}",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2)
        cv2.putText(display, f"Player: {self.player_score}",
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)

        # ── Status bar (bottom-left) ─────────────────────────
        ok  = self.arduino.is_connected
        ser_txt = "SER:OK" if ok else "SER:--"
        ser_col = (0, 255, 100) if ok else (0, 80, 255)
        cv2.putText(display, ser_txt,
                    (10, FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ser_col, 1)

        if self.emergency:
            cv2.putText(display, "!! EMERGENCY DEFENSE !!",
                        (FRAME_W // 2 - 200, FRAME_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # ── MAIN LOOP ────────────────────────────────────────────

    def run(self):
        global state, defense_line_x

        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask",   cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Camera", mouse_click)

        logger.info("=== Air Hockey AI ===")
        logger.info("1. Click 4 table corners in the Camera window  (TL→TR→BR→BL)")
        logger.info("2. Click inside the table to set defense line")
        logger.info("   (You can re-click at any time to move the defense line)")

        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue

            now     = time.time()
            display = frame.copy()

            # ── Phase: corner calibration ─────────────────────
            if state == STATE_CORNERS:
                for i, p in enumerate(corner_points):
                    cv2.circle(display, p, 7, (0, 0, 255), -1)
                    cv2.putText(display, str(i+1), (p[0]+9, p[1]-9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw partial polygon as corners are added
                if len(corner_points) >= 2:
                    for i in range(len(corner_points) - 1):
                        cv2.line(display, corner_points[i], corner_points[i+1],
                                 (0, 200, 255), 1)
                cv2.putText(display,
                            f"Click corner {len(corner_points)+1}/4  (TL→TR→BR→BL)",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                cv2.imshow("Camera", display)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # ── Phase: defense line setup ─────────────────────
            if state == STATE_DEFENSE:
                # Draw calibrated table border
                border_cam = table_pts_to_camera(
                    [(0,0),(TABLE_W,0),(TABLE_W,TABLE_H),(0,TABLE_H)])
                cv2.polylines(display, [np.array(border_cam)], True,
                              (0,200,255), 2)
                cv2.putText(display,
                            "Click inside the table to set the defense line",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.imshow("Camera", display)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # ── Phase: running ────────────────────────────────
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
                    if vx > 0 and px < defense_line_x:
                        self.send(defense_line_x, target_y)
                        self.last_target = (defense_line_x, target_y)

            # Draw everything onto the camera frame
            self.draw_overlays(display, puck_table, vx, vy, speed)

            # Defense-line hint if not yet set
            if defense_line_x is None:
                cv2.putText(display,
                            "Click on the table to set defense line",
                            (10, FRAME_H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Mask debug window
            mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            if puck:
                cv2.circle(mask_rgb, puck, 10, (0,255,0), 2)
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