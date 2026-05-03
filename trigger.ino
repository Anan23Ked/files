#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

#define OLED_W      128
#define OLED_H       64
#define OLED_RESET   -1

#define LEFT_EYE_ADDR  0x3C
#define RIGHT_EYE_ADDR 0x3D

Adafruit_SSD1306 leftEye(OLED_W, OLED_H, &Wire, OLED_RESET);
Adafruit_SSD1306 rightEye(OLED_W, OLED_H, &Wire, OLED_RESET);

#define RX2_PIN        16
#define TX2_PIN        17
#define BAUD           9600

#define LEFT_BROW_PIN  13
#define RIGHT_BROW_PIN 12

Servo leftBrow;
Servo rightBrow;

// ── Brow neutral positions — calibrate these to your physical setup ──
// 90 = horizontal neutral. Increase = raises brow, decrease = lowers.
#define BROW_NEUTRAL     90
#define BROW_HAPPY       75    // both slightly raised, soft
#define BROW_SAD_INNER   70    // inner corners raised (drooping look)
#define BROW_ANGRY_LOW   110   // both pushed down
#define BROW_ANGRY_SKEW   8   // extra offset on one brow for asymmetry

int cx = 64, cy = 32;
int pupilSize     = 10;
int targetPupil   = 10;
int irisDensity   = 8;
int highlightSize = 4;

int lastExpressionCode    = -1;
int pendingExpressionCode = -1;

// ── Smoothly move both brows to target angles ─────────────────────
void moveBrows(int leftTarget, int rightTarget, int stepDelay = 12) {
  int leftCurrent  = leftBrow.read();
  int rightCurrent = rightBrow.read();

  int steps = max(abs(leftTarget - leftCurrent), abs(rightTarget - rightCurrent));

  for (int i = 1; i <= steps; i++) {
    // Interpolate each brow independently
    int l = leftCurrent  + (leftTarget  - leftCurrent)  * i / steps;
    int r = rightCurrent + (rightTarget - rightCurrent) * i / steps;
    leftBrow.write(l);
    rightBrow.write(r);
    delay(stepDelay);
  }

  // Ensure we land exactly on target
  leftBrow.write(leftTarget);
  rightBrow.write(rightTarget);
}

// ── Draw one eye ──────────────────────────────────────────────────
void drawEye(Adafruit_SSD1306 &d, int pupil, int offsetX) {
  d.clearDisplay();

  d.fillCircle(cx, cy, 32, SSD1306_WHITE);

  int irisRadius = 30;
  for (int a = 0; a < 360; a += irisDensity) {
    float r = a * 3.14159 / 180.0;
    int x1 = cx + offsetX + cos(r) * pupil;
    int y1 = cy            + sin(r) * pupil;
    int x2 = cx + offsetX + cos(r) * irisRadius;
    int y2 = cy            + sin(r) * irisRadius;
    d.drawLine(x1, y1, x2, y2, SSD1306_BLACK);
  }

  d.drawCircle(cx + offsetX, cy, irisRadius, SSD1306_BLACK);
  d.fillCircle(cx + offsetX, cy, pupil, SSD1306_BLACK);
  d.fillCircle(cx + offsetX - 10, cy - 10, highlightSize,     SSD1306_WHITE);
  d.fillCircle(cx + offsetX -  4, cy -  6, highlightSize / 2, SSD1306_WHITE);

  d.display();
}

// ── Animate pupil toward target ───────────────────────────────────
void animateEyes() {
  while (pupilSize != targetPupil) {
    if (pupilSize < targetPupil) pupilSize++;
    else                         pupilSize--;
    drawEye(leftEye,  pupilSize, +3);
    drawEye(rightEye, pupilSize, -3);
    delay(30);
  }
  drawEye(leftEye,  pupilSize, +3);
  drawEye(rightEye, pupilSize, -3);
}

// ── Expressions ───────────────────────────────────────────────────
void showHappy() {
  Serial.println("[Face] HAPPY");
  targetPupil   = 6;
  irisDensity   = 12;
  highlightSize = 5;

  // Brows and eyes animate in parallel feel — brows first, snappy
  moveBrows(BROW_HAPPY, BROW_HAPPY, 10);
  animateEyes();
}

void showSad() {
  Serial.println("[Face] SAD");
  targetPupil   = 14;
  irisDensity   = 14;
  highlightSize = 2;

  // Inner brows raised gives drooping sad look
  // Left brow up, right brow up — mirrored sad angle
  moveBrows(BROW_SAD_INNER, BROW_SAD_INNER, 15);
  animateEyes();
}

void showAngry() {
  Serial.println("[Face] ANGRY");
  targetPupil   = 5;
  irisDensity   = 4;
  highlightSize = 3;

  // Both brows pushed down, left slightly more than right — asymmetric scowl
  moveBrows(BROW_ANGRY_LOW + BROW_ANGRY_SKEW,   // left brow lower
            BROW_ANGRY_LOW,                       // right brow less low
            8);                                   // fast, snappy movement
  animateEyes();
}

void showNeutral() {
  Serial.println("[Face] NEUTRAL");
  targetPupil   = 10;
  irisDensity   = 8;
  highlightSize = 4;

  moveBrows(BROW_NEUTRAL, BROW_NEUTRAL, 12);
  animateEyes();
}

// ── Setup ─────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("[ESP32] Booting...");

  Serial2.begin(BAUD, SERIAL_8N1, RX2_PIN, TX2_PIN);

  Wire.begin();

  if (!leftEye.begin(SSD1306_SWITCHCAPVCC, LEFT_EYE_ADDR)) {
    Serial.println("[ERROR] Left OLED not found at 0x3C");
    while (true);
  }
  Serial.println("[OLED] Left eye OK");

  if (!rightEye.begin(SSD1306_SWITCHCAPVCC, RIGHT_EYE_ADDR)) {
    Serial.println("[ERROR] Right OLED not found at 0x3D");
    while (true);
  }
  Serial.println("[OLED] Right eye OK");

  leftEye.clearDisplay();  leftEye.display();
  rightEye.clearDisplay(); rightEye.display();

  // Servos
  leftBrow.attach(LEFT_BROW_PIN);
  rightBrow.attach(RIGHT_BROW_PIN);
  leftBrow.write(BROW_NEUTRAL);
  rightBrow.write(BROW_NEUTRAL);
  Serial.println("[Servo] Brows initialised at neutral");

  // Startup blink
  showNeutral();
  delay(300);
  targetPupil = 2;  animateEyes();
  delay(150);
  targetPupil = 10; animateEyes();

  Serial.println("[ESP32] Ready");
}

// ── Loop ──────────────────────────────────────────────────────────
void loop() {
  static String incomingLine = "";

  while (Serial2.available()) {
    char c = Serial2.read();
    if (c == '\n') {
      if (incomingLine.length() > 0) {
        int code = incomingLine.toInt();
        incomingLine = "";
        if (code >= 0 && code <= 3)
          pendingExpressionCode = code;
      }
    } else if (c >= '0' && c <= '3') {
      incomingLine += c;
      if (incomingLine.length() > 4) incomingLine = "";
    }
  }

  if (pendingExpressionCode != -1 && pendingExpressionCode != lastExpressionCode) {
    lastExpressionCode = pendingExpressionCode;
    Serial.print("[ESP32] Class code: ");
    Serial.println(lastExpressionCode);

    switch (lastExpressionCode) {
      case 0: showAngry();  break;
      case 1: showHappy();  break;
      case 2: showSad();    break;
      case 3: showNeutral(); break;
    }
  }

  pendingExpressionCode = -1;
}