/*
 * TouchDiag.ino — Raw touch reading diagnostic
 * Prints raw X, Y, Z (pressure) over serial at 9600 baud.
 * Use Serial Monitor to see if touch is registering.
 */

#include <SPI.h>
#include <TFTv2.h>
#include <SeeedTouchScreen.h>

TouchScreen ts = TouchScreen(XP, YP, XM, YM);

void setup() {
    Serial.begin(9600);
    SPI.begin();
    Tft.TFTinit();
    TFT_BL_ON;

    Tft.fillScreen(0, 319, 0, 239, BLACK);
    Tft.drawString("TOUCH THE SCREEN", 10, 50, 2, 0x07FF);
    Tft.drawString("Check Serial Monitor", 10, 100, 1, 0xFFFF);
    Tft.drawString("for raw values", 10, 120, 1, 0xFFFF);

    Serial.println("Touch Diagnostic Ready");
    Serial.println("Touch the screen - raw values will print below");
    Serial.println("Format: rawX, rawY, pressure");
    Serial.println("---");
}

void loop() {
    Point p = ts.getPoint();

    // Always print if any pressure at all
    if (p.z > 0) {
        Serial.print("Raw X=");
        Serial.print(p.x);
        Serial.print("  Raw Y=");
        Serial.print(p.y);
        Serial.print("  Pressure=");
        Serial.println(p.z);

        // Also show mapped values
        int mx = map(p.x, TS_MINX, TS_MAXX, 0, 240);
        int my = map(p.y, TS_MINY, TS_MAXY, 0, 320);
        Serial.print("  Mapped X=");
        Serial.print(mx);
        Serial.print("  Mapped Y=");
        Serial.println(my);
    }

    delay(100);
}
