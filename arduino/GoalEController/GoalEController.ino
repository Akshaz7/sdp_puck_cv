/*
 * GoalEController.ino
 *
 * Arduino Uno firmware for the GOAL-E air hockey robot.
 * Controls an H-bot gantry with CNC Shield v3 + A4988/DRV8825 drivers.
 *
 * H-Bot kinematics:
 *   stepper1 position = (X + Y) * STEPS_PER_MM
 *   stepper2 position = (-X + Y) * STEPS_PER_MM
 *
 *   Pure X motion  -> motors spin in OPPOSITE directions
 *   Pure Y motion  -> motors spin in SAME direction
 *   Diagonal       -> only one motor spins
 *
 * Gantry dimensions: 300mm wide (X) x 190mm deep (Y), centre at (140, 85)
 *
 * Serial protocol (9600 baud):
 *   Receive: "X<int>Y<int>\n"  — target position in mm (e.g. "X215Y135\n")
 *            "X<int>\n"        — X only, Y stays at current target
 *            "T1\n" / "T2\n"  — test: spin motor 1 or 2 individually
 *   Send:    "OK\n"            — acknowledgement after parsing
 *
 * Requires the AccelStepper library.
 */

#include <AccelStepper.h>

// --------------- Motor Setup ---------------
// CNC Shield v3 pin mapping:
//   X axis: Step=2, Dir=5
//   Y axis: Step=3, Dir=6
//   Enable (active LOW, shared): pin 8
#define ENABLE_PIN 8

AccelStepper stepper1(AccelStepper::DRIVER, 2, 5);  // CNC X slot
AccelStepper stepper2(AccelStepper::DRIVER, 3, 6);  // CNC Y slot

// --------------- Tuning Constants ---------------
static const float STEPS_PER_MM = 10.0;

static const float MAX_SPEED = 1200.0;
static const float ACCELERATION = 2400.0;

// Gantry dimensions in mm
static const float GANTRY_WIDTH_MM  = 300.0;  // X range (a bit of extra room)
static const float GANTRY_HEIGHT_MM = 190.0;  // Y range (a bit of extra room)

// Direction inversion — set to -1.0 if a motor moves the wrong way.
static const float INVERT_X = 1.0;
static const float INVERT_Y = 1.0;

// --------------- Serial Buffer ---------------
static const int SERIAL_BUF_SIZE = 64;
char serialBuffer[SERIAL_BUF_SIZE];
int bufferIndex = 0;

// --------------- Position State ---------------
static const float HOME_X_MM = GANTRY_WIDTH_MM / 2.0;
static const float HOME_Y_MM = GANTRY_HEIGHT_MM / 2.0;

float targetX_MM = HOME_X_MM;
float targetY_MM = HOME_Y_MM;

void setup() {
    Serial.begin(9600);

    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, LOW);

    stepper1.setMaxSpeed(MAX_SPEED);
    stepper1.setAcceleration(ACCELERATION);

    stepper2.setMaxSpeed(MAX_SPEED);
    stepper2.setAcceleration(ACCELERATION);

    // Start at 0,0 — assume gantry is at home on power-up
    stepper1.setCurrentPosition(0);
    stepper2.setCurrentPosition(0);

    Serial.println("GOAL-E ready");
}

void loop() {
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (bufferIndex > 0) {
                serialBuffer[bufferIndex] = '\0';
                processCommand(serialBuffer);
                bufferIndex = 0;
            }
        } else if (bufferIndex < SERIAL_BUF_SIZE - 1) {
            serialBuffer[bufferIndex++] = c;
        }
    }

    stepper1.run();
    stepper2.run();
}

void setMotorPositions(float xMM, float yMM) {
    // H-bot kinematics
    float x = xMM * INVERT_X;
    float y = yMM * INVERT_Y;

    long s1 = long((x + y) * STEPS_PER_MM);
    long s2 = long((-x + y) * STEPS_PER_MM);

    stepper1.moveTo(s1);
    stepper2.moveTo(s2);
}

void processCommand(const char* cmd) {
    // Test commands
    if (cmd[0] == 'T' && cmd[1] == '1') {
        stepper1.moveTo(stepper1.currentPosition() + 2000);
        Serial.println("TEST: motor1 +2000 steps");
        return;
    }
    if (cmd[0] == 'T' && cmd[1] == '2') {
        stepper2.moveTo(stepper2.currentPosition() + 2000);
        Serial.println("TEST: motor2 +2000 steps");
        return;
    }

    // Parse "X<num>Y<num>" or "X<num>"
    if (cmd[0] != 'X' && cmd[0] != 'x') return;

    float newX = atof(cmd + 1);

    const char* yPtr = strchr(cmd, 'Y');
    if (yPtr == NULL) yPtr = strchr(cmd, 'y');

    float newY = targetY_MM;
    if (yPtr != NULL) {
        newY = atof(yPtr + 1);
    }

    // Clamp to gantry bounds
    if (newX < 0.0) newX = 0.0;
    if (newX > GANTRY_WIDTH_MM) newX = GANTRY_WIDTH_MM;
    if (newY < 0.0) newY = 0.0;
    if (newY > GANTRY_HEIGHT_MM) newY = GANTRY_HEIGHT_MM;

    targetX_MM = newX;
    targetY_MM = newY;

    setMotorPositions(targetX_MM, targetY_MM);

    Serial.println("OK");
}
