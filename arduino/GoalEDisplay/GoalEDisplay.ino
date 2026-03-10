/*
 * GoalEDisplay.ino — GOAL-E main scoreboard on Gameduino 3X.
 * 800x480, no touch. All input via serial from Pi.
 * Uses Vertex2f for full 800px width support.
 */

#include <EEPROM.h>
#include <SPI.h>
#include <GD2.h>

// ======= COLOURS =======
#define C_CYAN     0x00FFFF
#define C_MAGENTA  0xFF00FF
#define C_DCYAN    0x004444
#define C_DMAG     0x440044
#define C_GRID     0x0A1A2A
#define C_TRON     0x003366
#define C_DIM      0x888888
#define C_ORANGE   0xFF8800
#define C_GREEN    0x00FF00

// Vertex2f helper — VertexFormat(0) means 1 pixel units
#define VX(x, y) GD.Vertex2f((x), (y))

// ======= STATE =======
enum Screen { SCR_IDLE, SCR_PLAY, SCR_GOAL, SCR_WIN };
Screen currentScreen = SCR_IDLE;

int hScore = 0;
int rScore = 0;

unsigned long goalTimer = 0;
char goalScorer = ' ';
bool goalDone = false;

unsigned long gameStartTime = 0;
int gameTotalSec = 0;

char serialBuf[64];
int serialIdx = 0;

int W, H;  // screen dims, set in setup()

// ======= IDLE ANIMATION STATE =======
#define NUM_PUCKS 5
#define NUM_PUSHERS 2

struct Sprite {
    float x, y, vx, vy;
    int size;
};

Sprite pucks[NUM_PUCKS];
Sprite pushers[NUM_PUSHERS];
unsigned long idleFrame = 0;

void initSprites() {
    // Bouncing pucks — small 8-bit style circles
    for (int i = 0; i < NUM_PUCKS; i++) {
        pucks[i].x = 80 + random(W - 160);
        pucks[i].y = 80 + random(H - 160);
        pucks[i].vx = (random(2) ? 1 : -1) * (1.5 + random(100) / 50.0);
        pucks[i].vy = (random(2) ? 1 : -1) * (1.5 + random(100) / 50.0);
        pucks[i].size = 8 + random(6);
    }
    // Pushers — larger circles with rings
    for (int i = 0; i < NUM_PUSHERS; i++) {
        pushers[i].x = 100 + random(W - 200);
        pushers[i].y = 100 + random(H - 200);
        pushers[i].vx = (random(2) ? 1 : -1) * (0.8 + random(60) / 100.0);
        pushers[i].vy = (random(2) ? 1 : -1) * (0.8 + random(60) / 100.0);
        pushers[i].size = 22;
    }
}

void updateSprite(Sprite &s, int margin) {
    s.x += s.vx;
    s.y += s.vy;
    if (s.x < margin)     { s.x = margin;     s.vx = -s.vx; }
    if (s.x > W - margin) { s.x = W - margin; s.vx = -s.vx; }
    if (s.y < margin)     { s.y = margin;     s.vy = -s.vy; }
    if (s.y > H - margin) { s.y = H - margin; s.vy = -s.vy; }
}

void drawPuck8bit(int cx, int cy, int r, uint32_t col, uint32_t hi) {
    // Outer ring
    GD.ColorRGB(col);
    GD.PointSize(16 * r);
    GD.Begin(POINTS);
    VX(cx, cy);
    // Inner highlight for 8-bit look
    GD.ColorRGB(hi);
    GD.PointSize(16 * (r - 3));
    VX(cx - 1, cy - 1);
    // Centre dot
    GD.ColorRGB(col);
    GD.PointSize(16 * (r / 3));
    VX(cx, cy);
}

void drawPusher8bit(int cx, int cy, int r, uint32_t col, uint32_t ring) {
    // Outer ring
    GD.ColorRGB(ring);
    GD.PointSize(16 * r);
    GD.Begin(POINTS);
    VX(cx, cy);
    // Inner solid
    GD.ColorRGB(col);
    GD.PointSize(16 * (r - 4));
    VX(cx, cy);
    // Handle nub
    GD.ColorRGB(ring);
    GD.PointSize(16 * (r / 3));
    VX(cx, cy);
    // Highlight
    GD.ColorRGB(0xFFFFFF);
    GD.ColorA(120);
    GD.PointSize(16 * 3);
    VX(cx - r / 3, cy - r / 3);
    GD.ColorA(255);
}

// ======= 7-SEGMENT DIGIT DRAWING =======
// Segments: A=top, B=topR, C=botR, D=bot, E=botL, F=topL, G=mid
const byte segTable[] = {
    0x3F, // 0: ABCDEF
    0x06, // 1: BC
    0x5B, // 2: ABDEG
    0x4F, // 3: ABCDG
    0x66, // 4: BCFG
    0x6D, // 5: ACDFG
    0x7D, // 6: ACDEFG
    0x07, // 7: ABC
    0x7F, // 8: ABCDEFG
    0x6F, // 9: ABCDFG
};

void drawSeg(int x, int y, int sw, int sh, int t, byte segs, uint32_t col) {
    GD.ColorRGB(col);
    GD.Begin(RECTS);
    int hw = sh / 2;
    if (segs & 0x01) { VX(x + t, y);          VX(x + sw - t, y + t); }         // A top
    if (segs & 0x02) { VX(x + sw - t, y + t);  VX(x + sw, y + hw - t/2); }     // B topR
    if (segs & 0x04) { VX(x + sw - t, y + hw + t/2); VX(x + sw, y + sh - t); } // C botR
    if (segs & 0x08) { VX(x + t, y + sh - t);  VX(x + sw - t, y + sh); }       // D bot
    if (segs & 0x10) { VX(x, y + hw + t/2);     VX(x + t, y + sh - t); }       // E botL
    if (segs & 0x20) { VX(x, y + t);            VX(x + t, y + hw - t/2); }      // F topL
    if (segs & 0x40) { VX(x + t, y + hw - t/2); VX(x + sw - t, y + hw + t/2); }// G mid
}

// Draw a number centered at (cx, cy) with digit size (dw x dh)
void drawBigNum(int num, int cx, int cy, int dw, int dh, int t, uint32_t col, uint32_t shadow) {
    char buf[4];
    itoa(num, buf, 10);
    int len = strlen(buf);
    int gap = t;
    int totalW = len * dw + (len - 1) * gap;
    int sx = cx - totalW / 2;
    int sy = cy - dh / 2;

    for (int i = 0; i < len; i++) {
        int d = buf[i] - '0';
        int dx = sx + i * (dw + gap);
        // Shadow
        drawSeg(dx + 3, sy + 3, dw, dh, t, segTable[d], shadow);
        // Main
        drawSeg(dx, sy, dw, dh, t, segTable[d], col);
    }
}

// ======= DRAWING HELPERS =======
void frameStart() {
    GD.ClearColorRGB(0x000000);
    GD.Clear();
    GD.VertexFormat(0);  // Vertex2f in whole pixels
}

void drawGrid(int sp) {
    GD.ColorRGB(C_GRID);
    GD.LineWidth(16);
    GD.Begin(LINES);
    for (int x = 0; x < W; x += sp) { VX(x, 0); VX(x, H); }
    for (int y = 0; y < H; y += sp) { VX(0, y); VX(W, y); }
}

void drawBrackets(int x1, int y1, int x2, int y2, int len, uint32_t col) {
    GD.ColorRGB(col);
    GD.LineWidth(16 * 2);
    GD.Begin(LINES);
    VX(x1, y1); VX(x1 + len, y1);
    VX(x1, y1); VX(x1, y1 + len);
    VX(x2 - len, y1); VX(x2, y1);
    VX(x2, y1); VX(x2, y1 + len);
    VX(x1, y2 - len); VX(x1, y2);
    VX(x1, y2); VX(x1 + len, y2);
    VX(x2, y2 - len); VX(x2, y2);
    VX(x2 - len, y2); VX(x2, y2);
}

void fillR(int x, int y, int w, int h, uint32_t col) {
    GD.ColorRGB(col);
    GD.Begin(RECTS);
    VX(x, y); VX(x + w, y + h);
}

void drawBorder() {
    GD.LineWidth(16 * 2);
    GD.ColorRGB(C_TRON);
    GD.Begin(LINE_STRIP);
    VX(2, 2); VX(W - 2, 2); VX(W - 2, H - 2); VX(2, H - 2); VX(2, 2);
    GD.ColorRGB(C_DCYAN);
    GD.Begin(LINE_STRIP);
    VX(5, 5); VX(W - 5, 5); VX(W - 5, H - 5); VX(5, H - 5); VX(5, 5);
    drawBrackets(8, 8, W - 8, H - 8, 25, C_CYAN);
}

// ===============================================
//              IDLE / WAITING SCREEN
// ===============================================
void drawIdle() {
    idleFrame++;
    frameStart();
    drawGrid(50);

    // Update and draw bouncing pucks
    for (int i = 0; i < NUM_PUCKS; i++) {
        updateSprite(pucks[i], pucks[i].size + 10);
        uint32_t pc = (i % 2 == 0) ? C_GREEN : C_CYAN;
        uint32_t ph = (i % 2 == 0) ? 0x88FF88 : 0x88FFFF;
        drawPuck8bit((int)pucks[i].x, (int)pucks[i].y, pucks[i].size, pc, ph);
    }

    // Update and draw pushers
    for (int i = 0; i < NUM_PUSHERS; i++) {
        updateSprite(pushers[i], pushers[i].size + 10);
        uint32_t pc = (i == 0) ? C_MAGENTA : C_CYAN;
        uint32_t pr = (i == 0) ? 0xFF66FF : 0x66FFFF;
        drawPusher8bit((int)pushers[i].x, (int)pushers[i].y, pushers[i].size, pc, pr);
    }

    drawBorder();

    // Animated centre stripe — pulsing glow
    int pulse = abs((int)(idleFrame % 120) - 60);  // 0-60-0
    uint32_t lineCol = (pulse * 2) << 8 | (pulse * 3);  // teal pulse
    GD.ColorRGB(lineCol);
    GD.LineWidth(16 * 2);
    GD.Begin(LINES);
    VX(W / 6, H / 2 + 20); VX(5 * W / 6, H / 2 + 20);
    GD.ColorRGB(C_TRON);
    GD.LineWidth(16);
    VX(W / 5, H / 2 + 24); VX(4 * W / 5, H / 2 + 24);

    // Dark panel behind title
    GD.ColorA(180);
    fillR(W / 6, H / 4 - 20, 4 * W / 6, 100, 0x000000);
    GD.ColorA(255);

    // GOAL-E title — ROM font 34, with bobbing
    // Simple bob: triangle wave ±3px
    int bobPhase = (idleFrame / 2) % 24;
    int bob = (bobPhase < 12) ? (bobPhase - 6) / 2 : (18 - bobPhase) / 2;
    GD.cmd_romfont(1, 34);
    GD.ColorRGB(C_DCYAN);
    GD.cmd_text(W / 2 + 2, H / 3 + bob + 2, 1, OPT_CENTER, "GOAL-E");
    GD.ColorRGB(C_CYAN);
    GD.cmd_text(W / 2, H / 3 + bob, 1, OPT_CENTER, "GOAL-E");

    // "WAITING FOR GAME..." — blink
    if ((idleFrame / 40) % 2 == 0) {
        GD.ColorRGB(C_DCYAN);
        GD.cmd_text(W / 2, H / 2 + 50, 29, OPT_CENTER, "WAITING FOR GAME...");
    }

    // Animated corner sparkles
    GD.Begin(POINTS);
    for (int i = 0; i < 8; i++) {
        int t = (idleFrame + i * 15) % 120;
        int brightness = abs(t - 60) * 4;
        if (brightness > 255) brightness = 255;
        GD.ColorRGB(0x000000 | (brightness << 16) | (brightness << 8) | brightness);
        GD.PointSize(16 * (2 + (t % 4)));
        // Scatter near corners
        int cx = (i < 4) ? 30 + (i % 2) * 20 : W - 50 + (i % 2) * 20;
        int cy = (i < 2 || (i >= 4 && i < 6)) ? 30 + (i % 3) * 15 : H - 60 + (i % 3) * 15;
        VX(cx, cy);
    }

    // Scanning line effect — horizontal line sweeping down
    int scanY = (idleFrame * 2) % (H + 40) - 20;
    GD.ColorA(60);
    GD.ColorRGB(C_CYAN);
    GD.LineWidth(16 * 2);
    GD.Begin(LINES);
    VX(0, scanY); VX(W, scanY);
    GD.ColorA(30);
    VX(0, scanY + 4); VX(W, scanY + 4);
    VX(0, scanY - 4); VX(W, scanY - 4);
    GD.ColorA(255);

    GD.swap();
}

// ===============================================
//              SCOREBOARD
// ===============================================
void drawScoreboard() {
    frameStart();
    drawGrid(50);
    drawBorder();

    // Header bar
    fillR(6, 6, W - 12, H / 8, 0x001118);

    int hdrY = H / 16 + 3;
    GD.ColorRGB(C_MAGENTA);
    GD.cmd_text(W / 4, hdrY, 31, OPT_CENTER, "HUMAN");
    GD.ColorRGB(C_DCYAN);
    GD.cmd_text(W / 2, hdrY, 31, OPT_CENTER, "VS");
    GD.ColorRGB(C_CYAN);
    GD.cmd_text(3 * W / 4, hdrY, 31, OPT_CENTER, "GOAL-E");

    int hdrBot = H / 8 + 6;
    GD.ColorRGB(C_CYAN);
    GD.LineWidth(16 * 2);
    GD.Begin(LINES);
    VX(6, hdrBot); VX(W - 6, hdrBot);
    GD.ColorRGB(C_TRON);
    GD.LineWidth(16);
    VX(6, hdrBot + 2); VX(W - 6, hdrBot + 2);

    int timerTop = H - H / 8;

    // Dashed centre divider
    GD.ColorRGB(C_DCYAN);
    GD.LineWidth(16);
    GD.Begin(LINES);
    for (int y = hdrBot + 8; y < timerTop - 8; y += 24) {
        VX(W / 2, y); VX(W / 2, min(y + 12, timerTop - 8));
    }

    // SCORES — custom 7-segment, massive
    int scoreY = (hdrBot + timerTop) / 2;
    int dh = timerTop - hdrBot - 40;  // fill available height
    int dw = dh * 5 / 9;             // aspect ratio
    int thick = dh / 10;

    drawBigNum(hScore, W / 4, scoreY, dw, dh, thick, C_MAGENTA, C_DMAG);
    drawBigNum(rScore, 3 * W / 4, scoreY, dw, dh, thick, C_CYAN, C_DCYAN);

    // Timer separator
    GD.ColorRGB(C_TRON);
    GD.LineWidth(16 * 2);
    GD.Begin(LINES);
    VX(6, timerTop); VX(W - 6, timerTop);
    GD.ColorRGB(C_DCYAN);
    GD.LineWidth(16);
    VX(6, timerTop + 2); VX(W - 6, timerTop + 2);

    // Timer bar — always visible
    int barX = 15, barW = W - 140;
    int barY = timerTop + (H - timerTop) / 2 - 12;
    int barH = 24;

    int elapsed = (millis() - gameStartTime) / 1000;

    // Bar outline
    GD.ColorRGB(C_DCYAN);
    GD.LineWidth(16 * 2);
    GD.Begin(LINE_STRIP);
    VX(barX, barY); VX(barX + barW, barY);
    VX(barX + barW, barY + barH); VX(barX, barY + barH);
    VX(barX, barY);

    if (gameTotalSec > 0) {
        // Timed mode — countdown bar
        int remaining = gameTotalSec - elapsed;
        if (remaining < 0) remaining = 0;

        int fillW = (long)barW * remaining / gameTotalSec;
        if (fillW > 0) {
            uint32_t bc = C_GREEN;
            if (remaining < gameTotalSec / 4) bc = 0xFF0000;
            else if (remaining < gameTotalSec / 2) bc = C_ORANGE;
            fillR(barX + 1, barY + 1, fillW - 1, barH - 2, bc);
            GD.ColorRGB(0xFFFFFF);
            GD.Begin(LINES);
            VX(barX + fillW, barY + 1);
            VX(barX + fillW, barY + barH - 1);
        }

        char tb[8];
        sprintf(tb, "%d:%02d", remaining / 60, remaining % 60);
        GD.ColorRGB(0xFFFFFF);
        GD.cmd_text(W - 60, barY + barH / 2, 31, OPT_CENTER, tb);
    } else {
        // Free play — elapsed time bar (fills up over 5 min)
        int maxSec = 300;
        int fillW = min((long)barW * elapsed / maxSec, (long)barW);
        if (fillW > 0) {
            fillR(barX + 1, barY + 1, fillW, barH - 2, C_GREEN);
        }

        char tb[8];
        sprintf(tb, "%d:%02d", elapsed / 60, elapsed % 60);
        GD.ColorRGB(0xFFFFFF);
        GD.cmd_text(W - 60, barY + barH / 2, 31, OPT_CENTER, tb);
    }

    GD.swap();
}

// ===============================================
//              GOAL! ANIMATION
// ===============================================
void startGoal(char scorer) {
    goalScorer = scorer;
    goalTimer = millis();
    goalDone = false;
    currentScreen = SCR_GOAL;
}

void drawGoal() {
    if (goalDone) return;
    unsigned long el = millis() - goalTimer;
    uint32_t mc = (goalScorer == 'H') ? C_MAGENTA : C_CYAN;
    uint32_t dc = (goalScorer == 'H') ? C_DMAG : C_DCYAN;

    // Rapid pulsing scanline flash for 1 second
    if (el < 1000) {
        frameStart();
        // Alternate between bright scanlines and dim every ~80ms
        bool bright = ((el / 80) % 2 == 0);
        uint32_t col = bright ? mc : dc;
        GD.ColorRGB(col);
        GD.Begin(RECTS);
        int offset = (el / 40) % 8;  // scroll the strips
        for (int y = -8 + offset; y < H; y += 8) {
            VX(0, y); VX(W, y + 4);
        }
        GD.swap();
        return;
    }

    frameStart();

    // Grid + scanlines
    drawGrid(30);
    GD.ColorRGB(dc);
    GD.LineWidth(16);
    GD.Begin(LINES);
    for (int y = 0; y < H; y += 5) { VX(0, y); VX(W, y); }

    // Double border
    GD.LineWidth(16 * 3);
    GD.ColorRGB(mc);
    GD.Begin(LINE_STRIP);
    VX(6, 6); VX(W - 6, 6); VX(W - 6, H - 6); VX(6, H - 6); VX(6, 6);
    GD.LineWidth(16 * 2);
    GD.Begin(LINE_STRIP);
    VX(12, 12); VX(W - 12, 12); VX(W - 12, H - 12); VX(12, H - 12); VX(12, 12);

    drawBrackets(16, 16, W - 16, H - 16, 30, 0xFFFFFF);

    // "GOAL!" — ROM font 34
    GD.cmd_romfont(1, 34);
    GD.ColorRGB(dc);
    GD.cmd_text(W / 2 + 2, H / 5 + 2, 1, OPT_CENTER, "GOAL!");
    GD.ColorRGB(mc);
    GD.cmd_text(W / 2, H / 5, 1, OPT_CENTER, "GOAL!");

    // Scorer — font 31
    GD.ColorRGB(0xFFFFFF);
    if (goalScorer == 'H')
        GD.cmd_text(W / 2, H / 2, 31, OPT_CENTER, "HUMAN SCORES!");
    else
        GD.cmd_text(W / 2, H / 2, 31, OPT_CENTER, "GOAL-E SCORES!");

    // Score — ROM font 34
    char sc[12];
    sprintf(sc, "%d - %d", hScore, rScore);
    GD.ColorRGB(dc);
    GD.cmd_text(W / 2 + 2, H * 2 / 3 + 2, 1, OPT_CENTER, sc);
    GD.ColorRGB(0xFFFFFF);
    GD.cmd_text(W / 2, H * 2 / 3, 1, OPT_CENTER, sc);

    // Accent
    GD.ColorRGB(dc);
    GD.LineWidth(16);
    GD.Begin(LINES);
    VX(W / 6, H - 40); VX(5 * W / 6, H - 40);

    // Corner sparkles
    GD.ColorRGB(mc);
    GD.PointSize(16 * 3);
    GD.Begin(POINTS);
    for (int i = 0; i < 6; i++) {
        VX(22 + i * 6, 22 + i * 6);
        VX(W - 22 - i * 6, 22 + i * 6);
        VX(22 + i * 6, H - 22 - i * 6);
        VX(W - 22 - i * 6, H - 22 - i * 6);
    }

    GD.swap();
    if (el >= 3800) goalDone = true;
}

// ===============================================
//              WIN / LOSE
// ===============================================
void drawWin(bool humanWins) {
    currentScreen = SCR_WIN;
    uint32_t mc = humanWins ? C_MAGENTA : C_CYAN;
    uint32_t dc = humanWins ? C_DMAG : C_DCYAN;

    frameStart();
    drawGrid(30);

    // Scanlines
    GD.ColorRGB(0x050510);
    GD.Begin(LINES);
    for (int y = 0; y < H; y += 5) { VX(0, y); VX(W, y); }

    // Triple border
    GD.LineWidth(16 * 2);
    GD.ColorRGB(dc);
    GD.Begin(LINE_STRIP);
    VX(3, 3); VX(W - 3, 3); VX(W - 3, H - 3); VX(3, H - 3); VX(3, 3);
    GD.ColorRGB(mc);
    GD.Begin(LINE_STRIP);
    VX(8, 8); VX(W - 8, 8); VX(W - 8, H - 8); VX(8, H - 8); VX(8, 8);
    GD.ColorRGB(dc);
    GD.Begin(LINE_STRIP);
    VX(13, 13); VX(W - 13, 13); VX(W - 13, H - 13); VX(13, H - 13); VX(13, 13);

    drawBrackets(16, 16, W - 16, H - 16, 35, mc);

    // Title — ROM font 34
    GD.cmd_romfont(1, 34);
    const char* title = humanWins ? "YOU WIN!" : "YOU LOSE!";
    GD.ColorRGB(dc);
    GD.cmd_text(W / 2 + 2, H / 6 + 2, 1, OPT_CENTER, title);
    GD.ColorRGB(mc);
    GD.cmd_text(W / 2, H / 6, 1, OPT_CENTER, title);

    // Line
    GD.ColorRGB(mc);
    GD.LineWidth(16 * 2);
    GD.Begin(LINES);
    VX(W / 5, H / 3 + 10); VX(4 * W / 5, H / 3 + 10);

    // Score — ROM font 34
    char buf[12];
    sprintf(buf, "%d - %d", hScore, rScore);
    GD.ColorRGB(dc);
    GD.cmd_text(W / 2 + 2, H / 2 + 2, 1, OPT_CENTER, buf);
    GD.ColorRGB(0xFFFFFF);
    GD.cmd_text(W / 2, H / 2, 1, OPT_CENTER, buf);

    // Line
    GD.ColorRGB(mc);
    GD.Begin(LINES);
    VX(W / 5, H * 2 / 3 - 10); VX(4 * W / 5, H * 2 / 3 - 10);

    // Trophy
    int tx = W / 2, ty = H * 2 / 3;
    fillR(tx - 30, ty, 60, 30, mc);
    fillR(tx - 22, ty + 30, 44, 6, mc);
    fillR(tx - 6, ty + 36, 12, 15, dc);
    fillR(tx - 22, ty + 51, 44, 6, mc);
    GD.ColorRGB(mc);
    GD.LineWidth(16 * 2);
    GD.Begin(LINE_STRIP);
    VX(tx - 30, ty + 5); VX(tx - 42, ty + 5);
    VX(tx - 42, ty + 22); VX(tx - 30, ty + 22);
    GD.Begin(LINE_STRIP);
    VX(tx + 30, ty + 5); VX(tx + 42, ty + 5);
    VX(tx + 42, ty + 22); VX(tx + 30, ty + 22);
    GD.ColorRGB(0xFFFFFF);
    GD.PointSize(16 * 5);
    GD.Begin(POINTS); VX(tx, ty + 13);

    // Sparkles
    GD.PointSize(16 * 2);
    for (int i = 0; i < 10; i++) {
        GD.ColorRGB((i & 1) ? 0xFFFFFF : mc);
        VX(tx - 50 + (i * 11) % 100, ty - 10 + (i * 7) % 70);
    }

    GD.swap();
}

// ===============================================
//              SERIAL FROM PI
// ===============================================
void processSerial(const char* cmd) {
    if (strncmp(cmd, "G:H", 3) == 0) startGoal('H');
    else if (strncmp(cmd, "G:R", 3) == 0) startGoal('R');
    else if (cmd[0] == 'S' && cmd[1] == ':') {
        sscanf(cmd + 2, "%d:%d", &hScore, &rScore);
    }
    else if (strncmp(cmd, "STATE:PLAYING", 13) == 0) {
        hScore = 0; rScore = 0;
        gameStartTime = millis();
        currentScreen = SCR_PLAY;
    }
    else if (strncmp(cmd, "STATE:WAITING", 13) == 0) {
        currentScreen = SCR_IDLE;
        initSprites();
        idleFrame = 0;
    }
    else if (strncmp(cmd, "WIN:", 4) == 0) {
        drawWin(strncmp(cmd + 4, "HUMAN", 5) == 0);
    }
    else if (cmd[0] == 'T' && cmd[1] == ':') {
        gameTotalSec = atoi(cmd + 2);
        gameStartTime = millis();
    }
}

// ===============================================
void setup() {
    Serial.begin(9600);
    GD.begin(0);
    W = GD.w;
    H = GD.h;
    Serial.print("Screen: ");
    Serial.print(W);
    Serial.print("x");
    Serial.println(H);
    randomSeed(analogRead(A5));
    initSprites();
}

void loop() {
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

    if (currentScreen == SCR_IDLE) {
        drawIdle();
    }
    else if (currentScreen == SCR_GOAL) {
        drawGoal();
        if (goalDone) currentScreen = SCR_PLAY;
    }
    else if (currentScreen == SCR_PLAY) {
        drawScoreboard();
    }
    // SCR_WIN stays static after drawWin()
}
