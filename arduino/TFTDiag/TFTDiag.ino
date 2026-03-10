// Try common 2.2" TFT driver IDs one by one
#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

MCUFRIEND_kbv tft;

// Common 2.2" driver IDs to try
uint16_t ids[] = { 0x9341, 0x9325, 0x9328, 0x7575, 0x0154, 0x9486, 0x1581 };
int numIds = 7;

void tryId(uint16_t id) {
    Serial.print("Trying 0x");
    Serial.println(id, HEX);

    tft.begin(id);
    tft.setRotation(0);
    tft.fillScreen(0x0000);  // black

    tft.setTextColor(0x07FF);  // cyan
    tft.setTextSize(3);
    tft.setCursor(20, 80);
    tft.print("ID: 0x");
    tft.println(id, HEX);

    tft.setTextSize(2);
    tft.setCursor(30, 150);
    tft.setTextColor(0xF81F);  // magenta
    tft.print("GOAL-E");

    tft.setTextSize(1);
    tft.setCursor(20, 250);
    tft.setTextColor(0xFFFF);
    tft.print("Send 'n' for next ID");
}

int currentIdx = 0;

void setup() {
    Serial.begin(9600);
    delay(500);
    Serial.println("Send 'n' to try next driver ID");
    tryId(ids[currentIdx]);
}

void loop() {
    if (Serial.available() > 0) {
        char c = Serial.read();
        if (c == 'n' || c == 'N') {
            currentIdx = (currentIdx + 1) % numIds;
            tryId(ids[currentIdx]);
        }
    }
}
