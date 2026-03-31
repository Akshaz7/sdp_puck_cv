/*
 * PIRGoalTest.ino — Grove Adjustable PIR goal detection test
 *
 * Plug Grove PIR into digital port D2 on the Grove shield.
 * Outputs "GOAL" over serial when motion is detected.
 *
 * The Grove Adjustable PIR has a potentiometer for sensitivity
 * and a jumper for hold time. Set hold time to LOW (short)
 * for faster re-triggering.
 */

const int PIR_PIN = 2;

unsigned long lastGoal = 0;
const unsigned long COOLDOWN_MS = 2000; // ignore re-triggers for 2 seconds

void setup() {
    Serial.begin(9600);
    pinMode(PIR_PIN, INPUT);
    Serial.println("PIR Goal Detector Ready");
    Serial.println("Waiting for motion...");
    delay(3000); // let PIR stabilise
    Serial.println("GO");
}

void loop() {
    int val = digitalRead(PIR_PIN);

    if (val == HIGH && millis() - lastGoal > COOLDOWN_MS) {
        Serial.println("GOAL");
        lastGoal = millis();
    }

    delay(50);
}
