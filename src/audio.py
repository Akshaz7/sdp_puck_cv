"""
Game Bridge — connects TFT Touch Panel, Gameduino Displays, and Motor Controller.

The Pi has up to 4 Arduinos plugged in via USB:
  - TFT Uno: sends MODE:xxx and START when user selects a game mode
  - Human Display: Gameduino facing the human player
  - Robot Display: Gameduino facing the robot side (optional)
  - Motor Controller: H-bot gantry for pusher movement (optional)

Usage:
    python3 game_bridge.py --tft /dev/ttyACM0 --human /dev/ttyACM1 --motor /dev/ttyACM2
"""

import argparse
import serial
import socket
import time
import logging
import sys
import select
import tty
import termios
import random
import math
import threading

try:
    from audio import AudioEngine
    HAS_AUDIO = True
except ImportError:
    try:
        from src.audio import AudioEngine
        HAS_AUDIO = True
    except ImportError:
        HAS_AUDIO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# UDP config — two-way communication with CV Pi
UDP_LISTEN_PORT = 5555
CV_PI_IP = "192.168.105.134"  # tentomon
CV_PI_PORT = 5556  # tentomon listens on this port


class CVBridgeListener:
    """Listens for UDP messages from the CV Pi (goal detection, scores, etc.)."""

    def __init__(self, port: int = UDP_LISTEN_PORT):
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.setblocking(False)
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        log.info("UDP listener started on port %d" % port)

    def _listen_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(1024)
                msg = data.decode("utf-8").strip()
                if msg:
                    with self._lock:
                        self._pending.append(msg)
                    log.info("<- CV Pi (%s): %s" % (addr[0], msg))
            except BlockingIOError:
                time.sleep(0.01)
            except OSError:
                if self._running:
                    log.error("UDP receive error")
                break

    def get_messages(self) -> list[str]:
        with self._lock:
            msgs = self._pending.copy()
            self._pending.clear()
        return msgs

    def stop(self):
        self._running = False
        self._sock.close()


class CVPiSender:
    """Sends game state messages to the CV Pi over UDP."""

    def __init__(self, ip: str = CV_PI_IP, port: int = CV_PI_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (ip, port)

    def send(self, msg: str):
        try:
            self._sock.sendto((msg + "\n").encode("utf-8"), self._addr)
            log.info("-> CV Pi: %s" % msg)
        except OSError as e:
            log.warning("UDP send to CV Pi failed: %s" % e)

    def stop(self):
        self._sock.close()


# Gantry constants (mm) — calibrated 2026-03-12
TABLE_WIDTH = 300.0
DEFENCE_Y = 95.0  # centre of Y range (190mm / 2)
CENTRE_X = TABLE_WIDTH / 2.0
MARGIN = 15.0  # don't go right to the edge


class RobotBrain:
    """Controls the robot pusher movement in different patterns.

    Moves along the defence line (y=1100) with varying strategies
    to make it unpredictable and fun to play against.
    """

    def __init__(self, motor_ser):
        self.motor = motor_ser
        self.current_x = CENTRE_X
        self.target_x = CENTRE_X
        self.last_send = 0
        self.min_interval = 0.05  # 50ms between motor commands
        self.pattern = "patrol"
        self.pattern_start = 0
        self.difficulty = 1  # 1=easy, 2=medium, 3=hard

        # Patrol state
        self.patrol_target = CENTRE_X
        self.patrol_wait_until = 0

        # Fake-out state
        self.fake_dir = 1
        self.fake_phase = 0

        # Pattern switch timer
        self.next_pattern_switch = 0

    def set_difficulty(self, mode):
        """Set difficulty based on game mode."""
        if mode in ("FREE", "FIRST3"):
            self.difficulty = 1
        elif mode in ("FIRST5", "TIMED3"):
            self.difficulty = 2
        else:
            self.difficulty = 3

    def send_position(self, x, y=DEFENCE_Y):
        """Send position to motor controller (rate-limited)."""
        now = time.time()
        if now - self.last_send < self.min_interval:
            return
        x = max(MARGIN, min(TABLE_WIDTH - MARGIN, x))
        cmd = "X%dY%d" % (int(x), int(y))
        self.motor.write((cmd + "\n").encode())
        self.motor.flush()
        self.current_x = x
        self.last_send = now

    def update(self):
        """Called every loop iteration. Moves the pusher."""
        now = time.time()

        # Switch patterns periodically for unpredictability
        if now >= self.next_pattern_switch:
            self._pick_pattern()

        if self.pattern == "patrol":
            self._do_patrol(now)
        elif self.pattern == "stalk":
            self._do_stalk(now)
        elif self.pattern == "fakeout":
            self._do_fakeout(now)
        elif self.pattern == "guard_centre":
            self._do_guard_centre(now)
        elif self.pattern == "sweep":
            self._do_sweep(now)

    def _pick_pattern(self):
        """Randomly select next movement pattern."""
        patterns = ["patrol", "stalk", "fakeout", "guard_centre", "sweep"]
        weights = {
            1: [30, 10, 10, 30, 20],  # easy: mostly patrol and guard centre
            2: [20, 20, 20, 15, 25],  # medium: balanced
            3: [10, 25, 30, 10, 25],  # hard: more fakeouts and stalking
        }
        w = weights[self.difficulty]
        self.pattern = random.choices(patterns, weights=w, k=1)[0]
        self.pattern_start = time.time()

        # Duration varies by pattern
        durations = {
            "patrol": random.uniform(3, 6),
            "stalk": random.uniform(2, 5),
            "fakeout": random.uniform(2, 4),
            "guard_centre": random.uniform(2, 4),
            "sweep": random.uniform(3, 5),
        }
        self.next_pattern_switch = time.time() + durations[self.pattern]
        log.info("Robot pattern: %s (difficulty %d)" % (self.pattern, self.difficulty))

    def _do_patrol(self, now):
        """Move to random positions along the defence line, pause, repeat."""
        if now >= self.patrol_wait_until:
            self.patrol_target = random.uniform(MARGIN + 15, TABLE_WIDTH - MARGIN - 15)
            pause = random.uniform(0.3, 1.5) if self.difficulty < 3 else random.uniform(0.1, 0.6)
            self.patrol_wait_until = now + pause + abs(self.patrol_target - self.current_x) / 400
        # Move toward target
        self.send_position(self.patrol_target)

    def _do_stalk(self, now):
        """Slowly drift toward a random side, then snap to the other — mimics reading the player."""
        elapsed = now - self.pattern_start
        cycle = elapsed % 3.0
        if cycle < 2.0:
            # Slow drift to one side
            drift_target = MARGIN + 20 if int(elapsed / 3) % 2 == 0 else TABLE_WIDTH - MARGIN - 20
            speed = 0.3 if self.difficulty == 1 else 0.5
            new_x = self.current_x + (drift_target - self.current_x) * speed * 0.02
            self.send_position(new_x)
        else:
            # Quick snap to opposite
            snap_target = TABLE_WIDTH - MARGIN - 20 if int(elapsed / 3) % 2 == 0 else MARGIN + 20
            self.send_position(snap_target)

    def _do_fakeout(self, now):
        """Quick juke movements — start going one way then snap the other."""
        elapsed = now - self.pattern_start
        cycle_len = 1.0 if self.difficulty < 3 else 0.6
        phase = (elapsed % cycle_len) / cycle_len

        if phase < 0.4:
            # Juke one direction
            juke_x = CENTRE_X + self.fake_dir * 70 * phase
            self.send_position(juke_x)
        elif phase < 0.5:
            # Pause
            pass
        else:
            # Snap opposite
            snap_x = CENTRE_X - self.fake_dir * 80 * (phase - 0.5) * 2
            self.send_position(snap_x)

        # Flip direction each cycle
        if phase > 0.95:
            self.fake_dir *= -1

    def _do_guard_centre(self, now):
        """Hold near centre with small random jitters."""
        jitter = random.uniform(-15, 15) if self.difficulty == 1 else random.uniform(-30, 30)
        target = CENTRE_X + jitter
        new_x = self.current_x + (target - self.current_x) * 0.1
        self.send_position(new_x)

    def _do_sweep(self, now):
        """Smooth sinusoidal sweep across the table."""
        elapsed = now - self.pattern_start
        speed = 1.5 if self.difficulty < 3 else 2.5
        amplitude = (TABLE_WIDTH / 2) - MARGIN - 20
        x = CENTRE_X + amplitude * math.sin(elapsed * speed)
        self.send_position(x)

    def go_home(self):
        """Return to centre of defence line."""
        self.send_position(CENTRE_X, DEFENCE_Y)

    def stop(self):
        """Stop and go home."""
        self.go_home()


class GameState:
    """Tracks the current game state."""

    def __init__(self):
        self.mode = None
        self.human_score = 0
        self.robot_score = 0
        self.playing = False
        self.start_time = 0
        self.time_limit = 0
        self.target_score = 0

    def reset(self, mode):
        self.mode = mode
        self.human_score = 0
        self.robot_score = 0
        self.playing = True
        self.start_time = time.time()
        if mode == "FREE":
            self.time_limit = 0; self.target_score = 0
        elif mode == "FIRST3":
            self.time_limit = 0; self.target_score = 3
        elif mode == "FIRST5":
            self.time_limit = 0; self.target_score = 5
        elif mode == "FIRST7":
            self.time_limit = 0; self.target_score = 7
        elif mode == "TIMED3":
            self.time_limit = 180; self.target_score = 0
        elif mode == "TIMED5":
            self.time_limit = 300; self.target_score = 0

    def check_win(self):
        if not self.playing:
            return None
        if self.target_score > 0:
            if self.human_score >= self.target_score:
                return "HUMAN"
            if self.robot_score >= self.target_score:
                return "ROBOT"
        if self.time_limit > 0:
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit:
                if self.human_score > self.robot_score:
                    return "HUMAN"
                elif self.robot_score > self.human_score:
                    return "ROBOT"
                else:
                    return None
        return None

    def score_string(self):
        return "S:%d:%d" % (self.human_score, self.robot_score)


def find_serial_ports():
    import glob
    ports = []
    for p in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
        ports.extend(glob.glob(p))
    ports.sort()
    return ports


def send(ser, msg):
    ser.write((msg + "\n").encode())
    ser.flush()
    log.info("-> %s: %s" % (ser.port, msg))


def send_both(human_disp, robot_disp, msg):
    send(human_disp, msg)
    if robot_disp:
        send(robot_disp, msg)


def send_win(human_disp, robot_disp, winner):
    send(human_disp, "WIN:%s" % winner)
    if robot_disp:
        flipped = "ROBOT" if winner == "HUMAN" else "HUMAN"
        send(robot_disp, "WIN:%s" % flipped)


def main():
    parser = argparse.ArgumentParser(description="GOAL-E Game Bridge")
    parser.add_argument("--tft", type=str, default=None,
                        help="Serial port for TFT Touch Uno")
    parser.add_argument("--human", type=str, default=None,
                        help="Serial port for human-side Gameduino display")
    parser.add_argument("--robot", type=str, default=None,
                        help="Serial port for robot-side Gameduino display")
    parser.add_argument("--motor", type=str, default=None,
                        help="Serial port for motor controller")
    parser.add_argument("--baud", type=int, default=9600,
                        help="Baud rate (default 9600)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio")
    args = parser.parse_args()

    # Auto-detect ports if not specified
    if args.tft is None or args.human is None:
        ports = find_serial_ports()
        if len(ports) < 2:
            log.error("Need at least 2 serial ports, found %d: %s" % (len(ports), ports))
            return
        if args.tft is None:
            args.tft = ports[0]
        if args.human is None:
            args.human = ports[1]
        if args.robot is None and len(ports) >= 3:
            remaining = [p for p in ports if p != args.tft and p != args.human]
            if remaining:
                args.robot = remaining[0]
        log.info("Auto-detected: TFT=%s, Human=%s, Robot=%s" % (
            args.tft, args.human, args.robot))

    log.info("Opening TFT on %s" % args.tft)
    tft = serial.Serial(args.tft, args.baud, timeout=0.05)
    log.info("Opening Human Display on %s" % args.human)
    human_disp = serial.Serial(args.human, args.baud, timeout=0.05)

    robot_disp = None
    if args.robot:
        log.info("Opening Robot Display on %s" % args.robot)
        robot_disp = serial.Serial(args.robot, args.baud, timeout=0.05)

    motor = None
    brain = None
    if args.motor:
        log.info("Opening Motor Controller on %s" % args.motor)
        motor = serial.Serial(args.motor, args.baud, timeout=0.05)
        time.sleep(2)  # wait for motor Arduino to reset
        brain = RobotBrain(motor)

    # Wait for Arduinos to reset after serial open
    time.sleep(2)
    log.info("All Arduinos connected")

    # Init audio
    audio = None
    if HAS_AUDIO and not args.no_audio:
        try:
            audio = AudioEngine(device_name="plughw:1,0")
        except Exception as e:
            log.warning("Audio init failed: %s" % e)
            audio = None

    # Put displays in idle/waiting state
    send_both(human_disp, robot_disp, "STATE:WAITING")
    if audio:
        audio.play_idle_music()
    if brain:
        brain.go_home()

    game = GameState()

    # Start UDP listener for CV Pi goal events + sender for game state
    cv_listener = CVBridgeListener()
    cv_sender = CVPiSender()

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    log.info("Keyboard: [h] human goal, [r] robot goal, [q] quit")

    def score_goal(scorer):
        if not game.playing:
            log.info("No game in progress")
            return
        if scorer == "H":
            game.human_score += 1
        else:
            game.robot_score += 1
        log.info("GOAL! %s -- %d:%d" % (
            "Human" if scorer == "H" else "Robot",
            game.human_score, game.robot_score))
        send_both(human_disp, robot_disp, "G:" + scorer)
        send(tft, "G:" + scorer)
        if audio:
            audio.play_goal()
        send_both(human_disp, robot_disp, game.score_string())
        send(tft, game.score_string())

        # Pause robot during goal celebration
        if brain:
            brain.go_home()

        winner = game.check_win()
        if winner:
            time.sleep(0.5)
            send_win(human_disp, robot_disp, winner)
            send(tft, "WIN:" + winner)
            game.playing = False
            if brain:
                brain.stop()
            if audio:
                if winner == "HUMAN":
                    audio.play_win()
                else:
                    audio.play_lose()
            log.info("Game over! Winner: %s" % winner)

    try:
        while True:
            # Keyboard input
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == "h":
                    score_goal("H")
                elif key == "r":
                    score_goal("R")
                elif key == "q":
                    break

            # Read from TFT touch panel
            if tft.in_waiting > 0:
                line = tft.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    log.info("<- TFT: %s" % line)

                    if line.startswith("MODE:"):
                        game.reset(line[5:])
                        if brain:
                            brain.set_difficulty(game.mode)
                        log.info("Game mode: %s" % game.mode)

                    elif line == "START":
                        send_both(human_disp, robot_disp, "STATE:PLAYING")
                        send(tft, "STATE:PLAYING")
                        cv_sender.send("STATE:PLAYING")
                        cv_sender.send("MODE:%s" % game.mode)
                        send_both(human_disp, robot_disp, game.score_string())
                        if game.time_limit > 0:
                            send_both(human_disp, robot_disp,
                                      "T:%d" % game.time_limit)
                        if audio:
                            audio.play_gameon()
                            time.sleep(0.8)
                            audio.play_gameplay_music()
                        log.info("Game started!")

                    elif line == "RESET":
                        game.playing = False
                        send_both(human_disp, robot_disp, "STATE:WAITING")
                        send(tft, "STATE:WAITING")
                        cv_sender.send("STATE:WAITING")
                        if brain:
                            brain.stop()
                        if audio:
                            audio.play_idle_music()
                        log.info("Game reset")

            # Read from CV Pi (UDP goal events)
            for msg in cv_listener.get_messages():
                if msg.startswith("G:"):
                    scorer = msg[2:]  # "H" or "R"
                    score_goal(scorer)
                elif msg.startswith("S:"):
                    # Score sync from CV Pi — update our state
                    parts = msg.split(":")
                    if len(parts) == 3:
                        game.human_score = int(parts[1])
                        game.robot_score = int(parts[2])
                        send_both(human_disp, robot_disp, game.score_string())
                        send(tft, game.score_string())
                elif msg.startswith("STATE:"):
                    state = msg[6:]
                    if state == "PLAYING":
                        send_both(human_disp, robot_disp, "STATE:PLAYING")
                        send(tft, "STATE:PLAYING")
                    elif state == "WAITING":
                        game.playing = False
                        send_both(human_disp, robot_disp, "STATE:WAITING")
                        send(tft, "STATE:WAITING")
                elif msg.startswith("WIN:"):
                    winner = msg[4:]
                    send_win(human_disp, robot_disp, winner)
                    send(tft, "WIN:" + winner)
                    game.playing = False
                    if brain:
                        brain.stop()
                    if audio:
                        if winner == "HUMAN":
                            audio.play_win()
                        else:
                            audio.play_lose()

            # Read from displays (drain any output)
            if human_disp.in_waiting > 0:
                line = human_disp.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    log.info("<- HumanDisp: %s" % line)

            if robot_disp and robot_disp.in_waiting > 0:
                line = robot_disp.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    log.info("<- RobotDisp: %s" % line)

            # Read from motor controller (drain OK responses)
            if motor and motor.in_waiting > 0:
                line = motor.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    log.info("<- Motor: %s" % line)

            # Update robot movement during gameplay
            if game.playing and brain:
                brain.update()

            # Check timed game expiry
            if game.playing and game.time_limit > 0:
                winner = game.check_win()
                if winner:
                    send_win(human_disp, robot_disp, winner)
                    send(tft, "WIN:" + winner)
                    game.playing = False
                    if brain:
                        brain.stop()
                    if audio:
                        if winner == "HUMAN":
                            audio.play_win()
                        else:
                            audio.play_lose()
                    log.info("Time's up! Winner: %s" % winner)

            time.sleep(0.01)

    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cv_listener.stop()
        cv_sender.stop()
        if brain:
            brain.stop()
        if audio:
            audio.cleanup()
        tft.close()
        human_disp.close()
        if robot_disp:
            robot_disp.close()
        if motor:
            motor.close()


if __name__ == "__main__":
    main()
