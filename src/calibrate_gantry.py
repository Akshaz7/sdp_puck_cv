"""
Gantry Calibration — free movement, no limits.
a/d = X small, A/D = X big
w/s = Y small, W/S = Y big
1 = mark corner, 2 = mark corner, 3 = mark corner, 4 = mark corner
p = print, q = quit
"""

import serial
import time
import sys
import tty
import termios
import select

PORT = "/dev/cu.usbmodem21101"
BAUD = 9600

pos_x = 0
pos_y = 0
corners = {}


def send(ser, x, y):
    # No clamping — full freedom
    cmd = "X%dY%d" % (int(x), int(y))
    ser.write((cmd + "\n").encode())
    ser.flush()


def main():
    global pos_x, pos_y

    # Temporarily remove clamp from firmware by sending huge range
    # Actually we just send raw — firmware clamps but we set limits huge
    # Let me just send direct

    print("Connecting to %s..." % PORT)
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    while ser.in_waiting:
        print("<- %s" % ser.readline().decode("utf-8", errors="ignore").strip())

    print("\nFREE MOVE — no limits")
    print("a/d = X -/+ 5mm    A/D = X -/+ 20mm")
    print("w/s = Y -/+ 5mm    W/S = Y -/+ 20mm")
    print("1/2/3/4 = mark corners")
    print("p = print    q = quit\n")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            if ser.in_waiting:
                ser.readline()

            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)
                moved = False

                if key == "a": pos_x -= 5; moved = True
                elif key == "d": pos_x += 5; moved = True
                elif key == "A": pos_x -= 20; moved = True
                elif key == "D": pos_x += 20; moved = True
                elif key == "w": pos_y -= 5; moved = True
                elif key == "s": pos_y += 5; moved = True
                elif key == "W": pos_y -= 20; moved = True
                elif key == "S": pos_y += 20; moved = True
                elif key in ("1", "2", "3", "4"):
                    corners[key] = (pos_x, pos_y)
                    print("\n>> Corner %s marked at X=%d Y=%d" % (key, pos_x, pos_y))
                elif key == "p":
                    print("\nPos: X=%d Y=%d" % (pos_x, pos_y))
                    for k in sorted(corners):
                        print("  Corner %s: X=%d Y=%d" % (k, corners[k][0], corners[k][1]))
                elif key == "q":
                    break

                if moved:
                    send(ser, pos_x, pos_y)
                    sys.stdout.write("\rX=%d  Y=%d    " % (pos_x, pos_y))
                    sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        ser.close()

        print("\n\n=== CORNERS ===")
        for k in sorted(corners):
            print("Corner %s: X=%d Y=%d" % (k, corners[k][0], corners[k][1]))
        if len(corners) >= 2:
            xs = [c[0] for c in corners.values()]
            ys = [c[1] for c in corners.values()]
            print("\nX range: %d to %d = %d mm" % (min(xs), max(xs), max(xs) - min(xs)))
            print("Y range: %d to %d = %d mm" % (min(ys), max(ys), max(ys) - min(ys)))


if __name__ == "__main__":
    main()
