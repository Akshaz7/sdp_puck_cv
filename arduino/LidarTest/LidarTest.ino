/*
 * LidarTest.ino — TF-Luna distance test
 *
 * Reads TF-Luna over SoftwareSerial and prints distance to Serial Monitor.
 *
 * Wiring:
 *   TF-Luna Red   → Arduino 5V
 *   TF-Luna Black → Arduino GND
 *   TF-Luna Green (TX) → Arduino pin 2
 *   TF-Luna White (RX) → Arduino pin 3
 */

#include <SoftwareSerial.h>

SoftwareSerial lidarSerial(2, 3); // RX=2 (Luna TX), TX=3 (Luna RX)

void setup() {
    Serial.begin(115200);
    lidarSerial.begin(115200);
    Serial.println("TF-Luna LiDAR Test");
    Serial.println("Dist(cm)  Strength");
}

void loop() {
    static uint8_t buf[9];
    static int idx = 0;

    while (lidarSerial.available()) {
        uint8_t b = lidarSerial.read();

        if (idx == 0 && b != 0x59) continue;
        if (idx == 1 && b != 0x59) { idx = 0; continue; }

        buf[idx++] = b;

        if (idx == 9) {
            idx = 0;

            // Verify checksum
            uint8_t check = 0;
            for (int i = 0; i < 8; i++) check += buf[i];
            if (check != buf[8]) continue;

            uint16_t dist = buf[2] | (buf[3] << 8);
            uint16_t strength = buf[4] | (buf[5] << 8);

            Serial.print(dist);
            Serial.print(" cm\t");
            Serial.println(strength);
        }
    }
}
