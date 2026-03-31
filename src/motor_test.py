"""
Motor test — control the H-bot from your Mac.

Keyboard controls:
    Arrow keys: move pusher (left/right/forward/back)
    1-5: run movement patterns (patrol, stalk, fakeout, guard, sweep)
    c: centre (home position)
    q: quit

Or just let it run a random "beat the robot" demo.
"""

import serial
import time
import sys
import tty
import termios
import select
import math
import random

PORT = "/dev/cu.usbmodem21101"
BAUD = 9600

GANTRY_WIDTH = 300.0
GANTRY_HEIGHT = 190.0
CENTRE_X = GANTRY_WIDTH / 2.0
CENTRE_Y = GANTRY_HEIGHT / 2.0
DEFENCE_Y = CENTRE_Y  # for now just use centre
MARGIN = 10.0

current_x = CENTRE_X
current_y = CENTRE_Y
STEP = 10  # mm per keypress


def send_pos(ser, x, y):
    x = max(0, min(GANTRY_WIDTH, x))
    y = max(0, min(GANTRY_HEIGHT, y))
    cmd = "X%dY%d" % (int(x), int(y))
    ser.write((cmd + "\n").encode())
    ser.flush()
    print("Sent: %s" % cmd)
    return x, y


def run_pattern_demo(ser, pattern, duration=8.0):
    """Run a movement pattern for a set duration."""
    print("Running pattern: %s for %.0fs" % (pattern, duration))
    start = time.time()
    x = CENTRE_X

    # Patrol state
    patrol_target = CENTRE_X
    patrol_wait = 0

    # Fakeout state
    fake_dir = 1

    while time.time() - start < duration:
        now = time.time()
        elapsed = now - start

        if pattern == "patrol":
            if now >= patrol_wait:
                patrol_target = random.uniform(MARGIN + 10, GANTRY_WIDTH - MARGIN - 10)
                patrol_wait = now + random.uniform(0.5, 1.5)
            x = patrol_target
            send_pos(ser, x, DEFENCE_Y)

        elif pattern == "sweep":
            amp = (GANTRY_WIDTH / 2) - MARGIN - 20
            x = CENTRE_X + amp * math.sin(elapsed * 2.0)
            send_pos(ser, x, DEFENCE_Y)

        elif pattern == "fakeout":
            cycle = elapsed % 1.2
            if cycle < 0.5:
                x = CENTRE_X + fake_dir * 70 * (cycle / 0.5)
            elif cycle < 0.6:
                pass
            else:
                x = CENTRE_X - fake_dir * 80 * ((cycle - 0.6) / 0.6)
            if cycle > 1.1:
                fake_dir *= -1
            send_pos(ser, x, DEFENCE_Y)

        elif pattern == "stalk":
            cycle = elapsed % 3.0
            if cycle < 2.0:
                drift = MARGIN + 20 if int(elapsed / 3) % 2 == 0 else GANTRY_WIDTH - MARGIN - 20
                x = x + (drift - x) * 0.03
            else:
                x = GANTRY_WIDTH - MARGIN - 20 if int(elapsed / 3) % 2 == 0 else MARGIN + 20
            send_pos(ser, x, DEFENCE_Y)

        elif pattern == "guard":
            jitter = random.uniform(-20, 20)
            target = CENTRE_X + jitter
            x = x + (target - x) * 0.1
            send_pos(ser, x, DEFENCE_Y)

        time.sleep(0.05)

    # Return home
    send_pos(ser, CENTRE_X, DEFENCE_Y)
    print("Pattern done, returned home")


def main():
    global current_x, current_y

    print("Connecting to motor controller on %s..." % PORT)
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)

    # Read startup message
    while ser.in_waiting:
        print("<- %s" % ser.readline().decode("utf-8", errors="ignore").strip())

    print("\nMotor Test Ready!")
    print("Controls:")
    print("  a/d  = move left/right")
    print("  w/s  = move forward/back")
    print("  c    = centre (home)")
    print("  1    = patrol pattern")
    print("  2    = sweep pattern")
    print("  3    = fakeout pattern")
    print("  4    = stalk pattern")
    print("  5    = guard centre pattern")
    print("  0    = random mix (beat-the-robot demo)")
    print("  q    = quit")
    print("")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            # Read motor responses
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print("<- Motor: %s" % line)

            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)

                if key == "q":
                    break
                elif key == "a":
                    current_x -= STEP
                    current_x, current_y = send_pos(ser, current_x, current_y)
                elif key == "d":
                    current_x += STEP
                    current_x, current_y = send_pos(ser, current_x, current_y)
                elif key == "w":
                    current_y -= STEP
                    current_x, current_y = send_pos(ser, current_x, current_y)
                elif key == "s":
                    current_y += STEP
                    current_x, current_y = send_pos(ser, current_x, current_y)
                elif key == "c":
                    current_x, current_y = CENTRE_X, DEFENCE_Y
                    send_pos(ser, current_x, current_y)
                elif key == "1":
                    run_pattern_demo(ser, "patrol")
                elif key == "2":
                    run_pattern_demo(ser, "sweep")
                elif key == "3":
                    run_pattern_demo(ser, "fakeout")
                elif key == "4":
                    run_pattern_demo(ser, "stalk")
                elif key == "5":
                    run_pattern_demo(ser, "guard")
                elif key == "0":
                    # Random mix — cycle through patterns
                    patterns = ["patrol", "sweep", "fakeout", "stalk", "guard"]
                    random.shuffle(patterns)
                    for p in patterns:
                        run_pattern_demo(ser, p, duration=5.0)
                    print("Demo complete!")

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        ser.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
