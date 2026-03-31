/*
 * GoalETouchPanel.ino — Seeed TFT Touch Shield v2.0
 * Mode select controller. Sends chosen mode to Pi via serial.
 * Also shows a mini live scoreboard when game is active.
 * Portrait mode (240x320).
 */

#include <SPI.h>
#include <TFTv2.h>
#include <SeeedTouchScreen.h>

TouchScreen ts = TouchScreen(XP, YP, XM, YM);

#define NEON_CYAN    0x07FF
#define NEON_MAGENTA 0xF81F
#define DARK_CYAN    0x0410
#define DARK_MAG     0x8010
#define GRID_COL     0x0208
#define ORANGE       0xFD20

int SW = 240;
int SH = 320;

enum Screen { SCR_MODE, SCR_LIVE };
Screen currentScreen = SCR_MODE;

int humanScore = 0;
int robotScore = 0;
int lastDrawnH = -1;
int lastDrawnR = -1;

char serialBuf[64];
int serialIdx = 0;

const char* modeNames[] = { "FREE PLAY", "FIRST TO 3", "FIRST TO 5", "TIMED 3:00", "TIMED 5:00" };
const char* modeCodes[] = { "FREE", "FIRST3", "FIRST5", "TIMED3", "TIMED5" };

// ===============================================
void drawCentered(const char* t, int y, int sz, unsigned int col) {
    int w = strlen(t) * sz * 8;
    int x = (SW - w) / 2;
    if (x < 0) x = 0;
    Tft.drawString((char*)t, x, y, sz, col);
}

void drawCornerBrackets(int x1, int y1, int x2, int y2, int len, unsigned int col) {
    Tft.drawHorizontalLine(x1, y1, len, col);
    Tft.drawVerticalLine(x1, y1, len, col);
    Tft.drawHorizontalLine(x2 - len, y1, len, col);
    Tft.drawVerticalLine(x2, y1, len, col);
    Tft.drawHorizontalLine(x1, y2, len, col);
    Tft.drawVerticalLine(x1, y2 - len, len, col);
    Tft.drawHorizontalLine(x2 - len, y2, len, col);
    Tft.drawVerticalLine(x2, y2 - len, len, col);
}

// ===============================================
//              MODE SELECT (portrait)
// ===============================================
void drawModeScreen() {
    currentScreen = SCR_MODE;
    Tft.fillScreen(0, 319, 0, 319, BLACK);

    // Grid
    for (int x = 0; x < SW; x += 30) Tft.drawVerticalLine(x, 0, SH, GRID_COL);
    for (int y = 0; y < SH; y += 30) Tft.drawHorizontalLine(0, y, SW, GRID_COL);
    Tft.drawRectangle(0, 0, SW, SH, DARK_CYAN);

    drawCornerBrackets(3, 3, SW - 3, SH - 3, 20, NEON_CYAN);

    drawCentered("G O A L - E", 15, 2, NEON_CYAN);
    Tft.drawHorizontalLine(40, 34, SW - 80, NEON_CYAN);
    Tft.drawHorizontalLine(50, 35, SW - 100, DARK_CYAN);

    drawCentered("SELECT GAME MODE", 42, 1, DARK_CYAN);
    Tft.drawHorizontalLine(20, 58, SW - 40, DARK_CYAN);

    for (int i = 0; i < 5; i++) {
        int by = 70 + i * 48;
        Tft.fillRectangle(20, by, SW - 40, 38, BLACK);
        Tft.drawRectangle(20, by, SW - 40, 38, DARK_CYAN);
        drawCornerBrackets(22, by + 2, SW - 22, by + 36, 6, NEON_CYAN);
        int tw = strlen(modeNames[i]) * 16;
        Tft.drawString((char*)modeNames[i], (SW - tw) / 2, by + 11, 2, DARK_CYAN);
    }
}

// ===============================================
//     MINI LIVE SCOREBOARD (portrait)
// ===============================================
void drawLiveLayout() {
    currentScreen = SCR_LIVE;
    Tft.fillScreen(0, 319, 0, 319, BLACK);

    // Grid
    for (int x = 0; x < SW; x += 30) Tft.drawVerticalLine(x, 0, SH, GRID_COL);
    for (int y = 0; y < SH; y += 30) Tft.drawHorizontalLine(0, y, SW, GRID_COL);

    Tft.drawRectangle(0, 0, SW, SH, DARK_CYAN);
    drawCornerBrackets(3, 3, SW - 3, SH - 3, 20, NEON_CYAN);

    drawCentered("G O A L - E", 10, 2, NEON_CYAN);
    Tft.drawHorizontalLine(20, 32, SW - 40, NEON_CYAN);

    // Labels
    drawCentered("HUMAN", 42, 2, NEON_MAGENTA);
    drawCentered("VS", 70, 1, DARK_CYAN);
    drawCentered("GOAL-E", 85, 2, NEON_CYAN);

    Tft.drawHorizontalLine(20, 108, SW - 40, DARK_CYAN);

    // Score area will be drawn by drawLiveScores()
    lastDrawnH = -1;
    lastDrawnR = -1;

    // Bottom: RESET button
    Tft.drawRectangle(50, 275, SW - 100, 35, DARK_CYAN);
    drawCentered("RESET", 283, 2, DARK_CYAN);
}

void drawLiveScores() {
    if (humanScore == lastDrawnH && robotScore == lastDrawnR) return;

    // Clear score area
    Tft.fillRectangle(10, 115, SW - 20, 140, BLACK);
    // Redraw grid in area
    for (int x = 10; x < SW - 10; x += 30) Tft.drawVerticalLine(x, 115, 140, GRID_COL);
    for (int y = 115; y < 255; y += 30) Tft.drawHorizontalLine(10, y, SW - 20, GRID_COL);

    // Human score
    char hb[4]; itoa(humanScore, hb, 10);
    int hw = strlen(hb) * 64;
    Tft.drawNumber(humanScore, (SW / 2 - hw) / 2 + 2, 132, 8, DARK_MAG);
    Tft.drawNumber(humanScore, (SW / 2 - hw) / 2, 130, 8, NEON_MAGENTA);

    // Dash
    Tft.drawHorizontalLine(SW / 2 - 15, 175, 30, DARK_CYAN);

    // Robot score
    char rb[4]; itoa(robotScore, rb, 10);
    int rw = strlen(rb) * 64;
    Tft.drawNumber(robotScore, (SW / 2 - rw) / 2 + SW / 2 + 2, 132, 8, DARK_CYAN);
    Tft.drawNumber(robotScore, (SW / 2 - rw) / 2 + SW / 2, 130, 8, NEON_CYAN);

    lastDrawnH = humanScore;
    lastDrawnR = robotScore;
}

// ===============================================
//              SERIAL FROM PI
// ===============================================
void processSerial(const char* cmd) {
    // Pi tells us to show mode select
    if (strncmp(cmd, "STATE:WAITING", 13) == 0) {
        drawModeScreen();
    }
    // Pi tells us game started
    else if (strncmp(cmd, "STATE:PLAYING", 13) == 0) {
        humanScore = 0; robotScore = 0;
        drawLiveLayout();
        drawLiveScores();
    }
    // Score update
    else if (cmd[0] == 'S' && cmd[1] == ':') {
        sscanf(cmd + 2, "%d:%d", &humanScore, &robotScore);
        if (currentScreen == SCR_LIVE) drawLiveScores();
    }
    // Goal scored — flash the score colour briefly
    else if (strncmp(cmd, "G:H", 3) == 0 || strncmp(cmd, "G:R", 3) == 0) {
        // Just update scores, main animation is on Gameduino 3X
        if (currentScreen == SCR_LIVE) drawLiveScores();
    }
    // Game over — back to mode select
    else if (strncmp(cmd, "WIN:", 4) == 0) {
        delay(5000);  // Let Gameduino show win screen
        drawModeScreen();
    }
}

// ===============================================
//              TOUCH
// ===============================================
void handleTouch(int rawX, int rawY) {
    // Map raw touch to screen coords — swap X/Y and invert if needed
    // Try standard mapping first, then swap axes
    int tx = map(rawX, TS_MINX, TS_MAXX, 0, 240);
    int ty = map(rawY, TS_MINY, TS_MAXY, 0, 320);

    // Debug: print mapped coords to serial
    Serial.print("TOUCH tx=");
    Serial.print(tx);
    Serial.print(" ty=");
    Serial.println(ty);

    if (currentScreen == SCR_MODE) {
        for (int i = 0; i < 5; i++) {
            int by = 70 + i * 48;
            if (ty >= by && ty < by + 38 && tx >= 20 && tx < 220) {
                // Visual feedback — briefly highlight button
                Tft.fillRectangle(20, by, SW - 40, 38, DARK_CYAN);
                int tw = strlen(modeNames[i]) * 16;
                Tft.drawString((char*)modeNames[i], (SW - tw) / 2, by + 11, 2, NEON_CYAN);
                delay(150);

                // Send mode to Pi
                Serial.print("MODE:");
                Serial.println(modeCodes[i]);
                delay(100);
                Serial.println("START");

                // Restore button
                Tft.fillRectangle(20, by, SW - 40, 38, BLACK);
                Tft.drawRectangle(20, by, SW - 40, 38, DARK_CYAN);
                drawCornerBrackets(22, by + 2, SW - 22, by + 36, 6, NEON_CYAN);
                Tft.drawString((char*)modeNames[i], (SW - tw) / 2, by + 11, 2, DARK_CYAN);
                return;
            }
        }
    }
    else if (currentScreen == SCR_LIVE) {
        // Check RESET button
        if (ty >= 275 && ty < 310 && tx >= 50 && tx < 190) {
            Serial.println("RESET");
        }
    }
}

// ===============================================
void setup() {
    Serial.begin(9600);
    SPI.begin();
    Tft.TFTinit();
    TFT_BL_ON;
    randomSeed(analogRead(A5));
    drawModeScreen();
}

void loop() {
    // Serial from Pi
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialIdx > 0) {
                serialBuf[serialIdx] = '\0';
                processSerial(serialBuf);
                serialIdx = 0;
            }
        } else if (serialIdx < 63)
            serialBuf[serialIdx++] = c;
    }

    // Touch
    Point p = ts.getPoint();
    if (p.z > __PRESSURE) {
        handleTouch(p.x, p.y);
        delay(300);
    }
}
