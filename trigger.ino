#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define OLED_W      128
#define OLED_H       64
#define OLED_RESET   -1

#define LEFT_EYE_ADDR  0x3C
#define RIGHT_EYE_ADDR 0x3D

Adafruit_SSD1306 leftEye(OLED_W, OLED_H, &Wire, OLED_RESET);
Adafruit_SSD1306 rightEye(OLED_W, OLED_H, &Wire, OLED_RESET);

#define RX2_PIN 16
#define TX2_PIN 17
#define BAUD    9600

int cx = 64, cy = 32;
int pupilSize    = 10;
int targetPupil  = 10;
int irisDensity  = 8;
int highlightSize = 4;

int lastExpressionCode    = -1;
int pendingExpressionCode = -1;

// ── Draw one eye ──────────────────────────────────────────────────
void drawEye(Adafruit_SSD1306 &d, int pupil, int offsetX) {
  d.clearDisplay();

  // Eyeball
  d.fillCircle(cx, cy, 32, SSD1306_WHITE);

  // Iris texture
  int irisRadius = 30;
  for (int a = 0; a < 360; a += irisDensity) {
    float r = a * 3.14159 / 180.0;
    int x1 = cx + offsetX + cos(r) * pupil;
    int y1 = cy          + sin(r) * pupil;
    int x2 = cx + offsetX + cos(r) * irisRadius;
    int y2 = cy          + sin(r) * irisRadius;
    d.drawLine(x1, y1, x2, y2, SSD1306_BLACK);
  }

  // Iris ring
  d.drawCircle(cx + offsetX, cy, irisRadius, SSD1306_BLACK);

  // Pupil
  d.fillCircle(cx + offsetX, cy, pupil, SSD1306_BLACK);

  // Highlights
  d.fillCircle(cx + offsetX - 10, cy - 10, highlightSize,     SSD1306_WHITE);
  d.fillCircle(cx + offsetX -  4, cy -  6, highlightSize / 2, SSD1306_WHITE);

  d.display();
}

// ── Animate pupil size toward target ─────────────────────────────
void animate() {
  // Step toward target, render each step so it looks smooth
  while (pupilSize != targetPupil) {
    if (pupilSize < targetPupil) pupilSize++;
    else                         pupilSize--;

    drawEye(leftEye,  pupilSize, +3);
    drawEye(rightEye, pupilSize, -3);
    delay(30);
  }
  // Final render at target
  drawEye(leftEye,  pupilSize, +3);
  drawEye(rightEye, pupilSize, -3);
}

// ── Expressions ───────────────────────────────────────────────────
void showHappy() {
  Serial.println("[Face] HAPPY");
  targetPupil   = 6;
  irisDensity   = 12;
  highlightSize = 5;
  animate();
}

void showSad() {
  Serial.println("[Face] SAD");
  targetPupil   = 14;
  irisDensity   = 14;
  highlightSize = 2;
  animate();
}

void showAngry() {
  Serial.println("[Face] ANGRY");
  targetPupil   = 5;
  irisDensity   = 4;
  highlightSize = 3;
  animate();
}

void showNeutral() {
  Serial.println("[Face] NEUTRAL");
  targetPupil   = 10;
  irisDensity   = 8;
  highlightSize = 4;
  animate();
}

// ── Setup ─────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("[ESP32] Booting...");

  Serial2.begin(BAUD, SERIAL_8N1, RX2_PIN, TX2_PIN);

  Wire.begin();

  // ── Init displays with error checking ────────────────────────
  if (!leftEye.begin(SSD1306_SWITCHCAPVCC, LEFT_EYE_ADDR)) {
    Serial.println("[ERROR] Left OLED not found at 0x3C — check wiring/address");
    while (true);   // halt so you can see the error
  }
  Serial.println("[OLED] Left eye OK");

  if (!rightEye.begin(SSD1306_SWITCHCAPVCC, RIGHT_EYE_ADDR)) {
    Serial.println("[ERROR] Right OLED not found at 0x3D — check wiring/address");
    while (true);
  }
  Serial.println("[OLED] Right eye OK");

  leftEye.clearDisplay();  leftEye.display();
  rightEye.clearDisplay(); rightEye.display();

  // Startup blink
  showNeutral();
  delay(300);
  targetPupil = 2;  animate();
  delay(150);
  targetPupil = 10; animate();
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