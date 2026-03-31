"""Quick test for TF-Luna LiDAR over UART.

Wiring:
    Red   → 5V
    Black → GND
    Green (TX) → Pi GPIO 15 (RXD)
    White (RX) → Pi GPIO 14 (TXD)

Usage:
    python3 lidar_test.py
    python3 lidar_test.py --port /dev/ttyAMA0
"""

import serial
import time
import argparse


def read_tfluna(ser):
    """Read one distance measurement from TF-Luna.

    TF-Luna sends 9-byte frames:
        [0x59, 0x59, Dist_L, Dist_H, Strength_L, Strength_H, Temp_L, Temp_H, Checksum]

    Returns:
        (distance_cm, strength) or (None, None) if read fails.
    """
    # Find frame header (two 0x59 bytes)
    while True:
        b = ser.read(1)
        if not b:
            return None, None
        if b[0] == 0x59:
            b2 = ser.read(1)
            if not b2:
                return None, None
            if b2[0] == 0x59:
                break

    # Read remaining 7 bytes
    data = ser.read(7)
    if len(data) < 7:
        return None, None

    # Parse
    dist_cm = data[0] + (data[1] << 8)
    strength = data[2] + (data[3] << 8)

    # Checksum: sum of all 9 bytes (including two 0x59 headers) & 0xFF
    checksum = (0x59 + 0x59 + sum(data[:6])) & 0xFF
    if checksum != data[6]:
        return None, None

    return dist_cm, strength


def main():
    parser = argparse.ArgumentParser(description="TF-Luna LiDAR test")
    parser.add_argument("--port", default="/dev/serial0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    args = parser.parse_args()

    print("Opening %s at %d baud..." % (args.port, args.baud))
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(0.5)

    # Track baseline for goal detection demo
    readings = []
    baseline = None
    GOAL_DROP_CM = 3  # distance must drop by at least this much

    print("Reading... (Ctrl+C to stop)")
    print("First 20 readings will set the baseline (empty box distance)")
    print()

    try:
        while True:
            dist, strength = read_tfluna(ser)
            if dist is not None:
                # Build baseline from first 20 readings
                if len(readings) < 20:
                    readings.append(dist)
                    if len(readings) == 20:
                        baseline = sum(readings) / len(readings)
                        print("Baseline set: %.1f cm" % baseline)
                        print("Drop a puck in to test goal detection!")
                        print()

                # Check for goal
                goal_detected = ""
                if baseline is not None and dist < baseline - GOAL_DROP_CM:
                    goal_detected = " *** GOAL! ***"

                print("Dist: %4d cm  Strength: %5d%s" % (dist, strength, goal_detected), end="\r")
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
