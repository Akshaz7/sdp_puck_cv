/*
 * GoalEController.ino
 *
 * Arduino Uno firmware for the GOAL-E air hockey robot.
 * Controls an H-bot gantry with two 4-wire stepper motors to position
 * a magnetic pusher anywhere on the table surface (X and Y).
 *
 * H-Bot kinematics:
 *   stepper1 position = (X + Y) * STEPS_PER_MM
 *   stepper2 position = (-X + Y) * STEPS_PER_MM
 *
 *   Pure X motion  → motors spin in OPPOSITE directions (matches your go_right/go_left)
 *   Pure Y motion  → motors spin in SAME direction (matches your go_forward/go_back)
 *   Diagonal       → only one motor spins
 *
 * Serial protocol (9600 baud):
 *   Receive: "X<int>Y<int>\n"  — target position in mm (e.g. "X305Y1100\n")
 *            "X<int>\n"        — X only, Y stays at current target (backwards compat)
 *   Send:    "OK\n"            — acknowledgement after parsing
 *
 * If motors move the wrong direction, swap the sign on INVERT_X / INVERT_Y below.
 *
 * Requires the AccelStepper library (install via Arduino Library Manager).
 */

#include <AccelStepper.h>

// --------------- Motor Setup ---------------
// FULL4WIRE: direct 4-wire stepper (28BYJ-48 / NEMA without driver board)
// Pin order per AccelStepper docs: IN1, IN3, IN2, IN4
#define MOTOR_INTERFACE_TYPE AccelStepper::FULL4WIRE

AccelStepper stepper1(MOTOR_INTERFACE_TYPE, 3, 4, 5, 6);
AccelStepper stepper2(MOTOR_INTERFACE_TYPE, 8, 9, 10, 11);

// --------------- Tuning Constants ---------------
// Steps per millimetre — calibrate by measuring actual travel distance.
static const float STEPS_PER_MM = 10.0;

// Maximum speed in steps/second (your current code uses 1200).
static const float MAX_SPEED = 1200.0;

// Acceleration in steps/second^2.
static const float ACCELERATION = 2400.0;

// Table dimensions in mm — used to clamp incoming targets.
static const float TABLE_WIDTH_MM  = 610.0;
static const float TABLE_HEIGHT_MM = 1220.0;

// Direction inversion — set to -1.0 if a motor moves the wrong way.
static const float INVERT_X = 1.0;
static const float INVERT_Y = 1.0;

// --------------- Serial Buffer ---------------
static const int SERIAL_BUF_SIZE = 64;
char serialBuffer[SERIAL_BUF_SIZE];
int bufferIndex = 0;

// --------------- Position State ---------------
// Assumed start position (centre of defence line). Change if you add limit switches.
static const float HOME_X_MM = 305.0;
static const float HOME_Y_MM = 1100.0;

float targetX_MM = HOME_X_MM;
float targetY_MM = HOME_Y_MM;

void setup() {
    Serial.begin(9600);

    stepper1.setMaxSpeed(MAX_SPEED);
    stepper1.setAcceleration(ACCELERATION);

    stepper2.setMaxSpeed(MAX_SPEED);
    stepper2.setAcceleration(ACCELERATION);

    // Set initial motor positions to match assumed home
    setMotorPositions(HOME_X_MM, HOME_Y_MM);
    stepper1.setCurrentPosition(stepper1.targetPosition());
    stepper2.setCurrentPosition(stepper2.targetPosition());

    Serial.println("GOAL-E ready");
}

void loop() {
    // --- Read serial commands (non-blocking) ---
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

    // --- Step motors toward their targets ---
    stepper1.run();
    stepper2.run();
}

void setMotorPositions(float xMM, float yMM) {
    // H-bot kinematics: convert carriage (X, Y) to motor positions
    float x = xMM * INVERT_X;
    float y = yMM * INVERT_Y;

    long s1 = long((x + y) * STEPS_PER_MM);
    long s2 = long((-x + y) * STEPS_PER_MM);

    stepper1.moveTo(s1);
    stepper2.moveTo(s2);
}

void processCommand(const char* cmd) {
    // Parse "X<num>Y<num>" or "X<num>"
    if (cmd[0] != 'X' && cmd[0] != 'x') return;

    // Find X value
    float newX = atof(cmd + 1);

    // Look for Y portion
    const char* yPtr = strchr(cmd, 'Y');
    if (yPtr == NULL) yPtr = strchr(cmd, 'y');

    float newY = targetY_MM;  // default: keep current Y target
    if (yPtr != NULL) {
        newY = atof(yPtr + 1);
    }

    // Clamp to table bounds
    if (newX < 0.0) newX = 0.0;
    if (newX > TABLE_WIDTH_MM) newX = TABLE_WIDTH_MM;
    if (newY < 0.0) newY = 0.0;
    if (newY > TABLE_HEIGHT_MM) newY = TABLE_HEIGHT_MM;

    targetX_MM = newX;
    targetY_MM = newY;

    setMotorPositions(targetX_MM, targetY_MM);

    Serial.println("OK");
}
