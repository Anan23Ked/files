// ─────────────────────────────────────────────────────────────────
//  robot_face.ino  —  ESP32 robot face controller
//
//  Receives expression labels from Raspberry Pi via UART2
//  Controls:
//    - 2x SSD1306 OLED (128x64) for eyes  via I2C
//    - 2x SG90 servo for eyebrows          via PWM
//    - 1x SG90 servo for mouth flip tile   via PWM
//
//  Wiring:
//    Pi TX (GPIO14)  -> ESP32 GPIO16 (RX2)
//    Pi RX (GPIO15)  <- ESP32 GPIO17 (TX2)
//    Pi GND          -> ESP32 GND
//
//    Left OLED  SDA -> GPIO21  SCL -> GPIO22  (I2C bus 0, addr 0x3C)
//    Right OLED SDA -> GPIO21  SCL -> GPIO22  (same bus, addr 0x3D)
//      * Set right OLED I2C address to 0x3D by bridging
//        the address solder pad on the back of the module
//
//    Left brow servo  -> GPIO13
//    Right brow servo -> GPIO12
//    Mouth servo      -> GPIO14
//
//  Libraries (install via Arduino Library Manager):
//    Adafruit SSD1306
//    Adafruit GFX
//    ESP32Servo
// ─────────────────────────────────────────────────────────────────

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

// ── Display config ────────────────────────────────────────────────
#define OLED_W       128
#define OLED_H        64
#define OLED_RESET    -1

// Both OLEDs share the same I2C bus — different addresses
Adafruit_SSD1306 leftEye(OLED_W, OLED_H, &Wire, OLED_RESET);
Adafruit_SSD1306 rightEye(OLED_W, OLED_H, &Wire, OLED_RESET);

#define LEFT_EYE_ADDR  0x3C
#define RIGHT_EYE_ADDR 0x3D

// ── Servo config ──────────────────────────────────────────────────
#define LEFT_BROW_PIN   13
#define RIGHT_BROW_PIN  12
#define MOUTH_PIN       14

Servo leftBrow;
Servo rightBrow;
Servo mouth;

// Servo angle reference (degrees)
//  Eyebrows: 90 = neutral flat
//            110 = raised (happy/surprised)
//             60 = furrowed inward (angry)
//             70 = drooped outer edge (sad)
//  Mouth:     90 = neutral flat tile
//             60 = smile position
//            120 = frown position
//             45 = open O (surprised)

// ── Serial ────────────────────────────────────────────────────────
#define RX2_PIN  16
#define TX2_PIN  17
#define BAUD     9600

int lastExpressionCode = -1;   // -1 = unset, 0 = angry, 1 = happy, 2 = sad, 3 = neutral
int pendingExpressionCode = -1; // For debouncing rapid serial input
// ─────────────────────────────────────────────────────────────────
//  Eye drawing helpers
// ─────────────────────────────────────────────────────────────────

// Draw a simple circular eye
void drawCircleEye(Adafruit_SSD1306 &display, int pupilX, int pupilY) {
  display.clearDisplay();
  // Outer eye white (filled rounded rect)
  display.fillRoundRect(10, 8, 108, 50, 20, WHITE);
  // Pupil
  display.fillCircle(pupilX, pupilY, 18, BLACK);
  // Highlight glint
  display.fillCircle(pupilX + 6, pupilY - 6, 5, WHITE);
  display.display();
}

// Draw a crescent (happy squinting) eye
void drawCrescentEye(Adafruit_SSD1306 &display) {
  display.clearDisplay();
  // Draw arc shape — filled ellipse bottom half only
  for (int y = 32; y <= 56; y++) {
    float ratio  = (float)(y - 32) / 24.0;
    int   halfW  = (int)(50 * sin(ratio * 3.14159));
    display.drawFastHLine(64 - halfW, y, halfW * 2, WHITE);
  }
  display.display();
}

// Draw wide open (surprised) eye
void drawWideEye(Adafruit_SSD1306 &display) {
  display.clearDisplay();
  display.fillCircle(64, 32, 30, WHITE);   // large white
  display.fillCircle(64, 32, 18, BLACK);   // pupil
  display.fillCircle(72, 24,  6, WHITE);   // glint
  display.display();
}

// Draw squinting (angry) eye — narrowed rectangle
void drawSquintEye(Adafruit_SSD1306 &display) {
  display.clearDisplay();
  display.fillRoundRect(10, 22, 108, 22, 8, WHITE);
  display.fillCircle(64, 33, 10, BLACK);
  display.display();
}

// Draw half-closed (sad) eye
void drawSadEye(Adafruit_SSD1306 &display) {
  display.clearDisplay();
  display.fillRoundRect(10, 8, 108, 50, 20, WHITE);
  // Shadow over top half to look droopy
  display.fillRect(10, 8, 108, 25, BLACK);
  display.fillCircle(64, 38, 14, BLACK);
  display.fillCircle(70, 32,  5, WHITE);
  display.display();
}

// Draw neutral eye — standard circle with centred pupil
void drawNeutralEye(Adafruit_SSD1306 &display) {
  drawCircleEye(display, 64, 32);
}

// Blink animation — both eyes
void blinkEyes(void (*drawFn)(Adafruit_SSD1306&)) {
  // Close
  for (int h = 32; h >= 0; h -= 8) {
    leftEye.clearDisplay();
    leftEye.fillRoundRect(10, 32 - h, 108, h * 2, 8, WHITE);
    leftEye.display();
    rightEye.clearDisplay();
    rightEye.fillRoundRect(10, 32 - h, 108, h * 2, 8, WHITE);
    rightEye.display();
    delay(30);
  }
  delay(80);
  // Open
  drawFn(leftEye);
  drawFn(rightEye);
}


// ─────────────────────────────────────────────────────────────────
//  Servo helpers
// ─────────────────────────────────────────────────────────────────

void moveBrows(int leftAngle, int rightAngle, int stepDelay = 15) {
  int lCurrent = leftBrow.read();
  int rCurrent = rightBrow.read();
  int steps    = max(abs(leftAngle - lCurrent),
                     abs(rightAngle - rCurrent));

  Serial.print("[Servo] moveBrows: left ");
  Serial.print(lCurrent); Serial.print("→"); Serial.print(leftAngle);
  Serial.print(" | right ");
  Serial.print(rCurrent); Serial.print("→"); Serial.print(rightAngle);
  Serial.print(" | steps: "); Serial.println(steps);

  for (int i = 0; i <= steps; i++) {
    int l = lCurrent + (leftAngle  - lCurrent) * i / steps;
    int r = rCurrent + (rightAngle - rCurrent) * i / steps;
    leftBrow.write(l);
    rightBrow.write(r);
    delay(stepDelay);
  }
}

void moveMouth(int angle, int stepDelay = 15) {
  int current = mouth.read();
  int steps   = abs(angle - current);
  
  Serial.print("[Servo] moveMouth: ");
  Serial.print(current); Serial.print("→"); Serial.print(angle);
  Serial.print(" | steps: "); Serial.println(steps);
  
  for (int i = 0; i <= steps; i++) {
    mouth.write(current + (angle - current) * i / steps);
    delay(stepDelay);
  }
}


// ─────────────────────────────────────────────────────────────────
//  Expression animations
// ─────────────────────────────────────────────────────────────────

void showHappy() {
  Serial.println("[Face] HAPPY — calling expression");
  Serial.println("[Face] → blinkEyes(drawCrescentEye)");
  blinkEyes(drawCrescentEye);
  Serial.println("[Face] → drawCrescentEye both displays");
  drawCrescentEye(leftEye);
  drawCrescentEye(rightEye);
  Serial.println("[Face] → moveBrows(110, 110)");
  moveBrows(110, 110);   // both brows up
  Serial.println("[Face] → moveMouth(60)");
  moveMouth(60);         // smile tile position
  Serial.println("[Face] HAPPY complete");
}

void showSad() {
  Serial.println("[Face] SAD — calling expression");
  Serial.println("[Face] → drawSadEye both displays");
  drawSadEye(leftEye);
  drawSadEye(rightEye);
  Serial.println("[Face] → moveBrows(75, 75)");
  moveBrows(75, 75);     // inner edges down — droopy
  Serial.println("[Face] → moveMouth(120)");
  moveMouth(120);        // frown tile position
  Serial.println("[Face] SAD complete");
}

void showAngry() {
  Serial.println("[Face] ANGRY — calling expression");
  Serial.println("[Face] → drawSquintEye both displays");
  drawSquintEye(leftEye);
  drawSquintEye(rightEye);
  Serial.println("[Face] → moveBrows(65, 65)");
  moveBrows(65, 65);     // both furrowed down
  Serial.println("[Face] → moveMouth(95)");
  moveMouth(95);         // thin line — nearly neutral, slight tension
  Serial.println("[Face] ANGRY complete");
}

void showNeutral() {
  Serial.println("[Face] NEUTRAL — calling expression");
  Serial.println("[Face] → drawNeutralEye both displays");
  drawNeutralEye(leftEye);
  drawNeutralEye(rightEye);
  Serial.println("[Face] → moveBrows(90, 90)");
  moveBrows(90, 90);     // flat centre
  Serial.println("[Face] → moveMouth(90)");
  moveMouth(90);         // flat tile
  Serial.println("[Face] NEUTRAL complete");
}

// Startup animation — test all components
void startupAnimation() {
  Serial.println("[Face] Startup animation...");

  // Sweep servos full range
  moveBrows(60, 60,  8);
  delay(200);
  moveBrows(120, 120, 8);
  delay(200);
  moveBrows(90, 90,  8);

  moveMouth(45,  8); delay(200);
  moveMouth(120, 8); delay(200);
  moveMouth(90,  8);

  // Flash all eye states
  void (*eyeFns[])(Adafruit_SSD1306&) = {
    drawNeutralEye, drawCrescentEye, drawWideEye,
    drawSquintEye,  drawSadEye
  };
  // Note: drawHappyEye = drawCrescentEye — alias below
  drawCrescentEye(leftEye); drawCrescentEye(rightEye); delay(300);
  drawWideEye(leftEye);     drawWideEye(rightEye);     delay(300);
  drawSquintEye(leftEye);   drawSquintEye(rightEye);   delay(300);
  drawSadEye(leftEye);      drawSadEye(rightEye);      delay(300);

  showNeutral();
  Serial.println("[Face] Ready.");
}


// ─────────────────────────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────────────────────────

void setup() {
  // Debug serial (USB to laptop if connected)
  Serial.begin(115200);
  Serial.println("[ESP32] Booting...");

  // UART2 — receives from Raspberry Pi
  Serial2.begin(BAUD, SERIAL_8N1, RX2_PIN, TX2_PIN);
  Serial.println("[ESP32] UART2 listening on GPIO16");

  // I2C
  Wire.begin();

  // Init OLEDs
  if (!leftEye.begin(SSD1306_SWITCHCAPVCC, LEFT_EYE_ADDR)) {
    Serial.println("[ERROR] Left OLED not found at 0x3C");
  }
  if (!rightEye.begin(SSD1306_SWITCHCAPVCC, RIGHT_EYE_ADDR)) {
    Serial.println("[ERROR] Right OLED not found at 0x3D");
  }

  leftEye.clearDisplay();  leftEye.display();
  rightEye.clearDisplay(); rightEye.display();
  Serial.println("[ESP32] OLEDs initialised");

  // Init servos
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);

  leftBrow.setPeriodHertz(50);
  rightBrow.setPeriodHertz(50);
  mouth.setPeriodHertz(50);

  leftBrow.attach(LEFT_BROW_PIN,  500, 2400);
  rightBrow.attach(RIGHT_BROW_PIN, 500, 2400);
  mouth.attach(MOUTH_PIN,          500, 2400);

  Serial.print("[Servo] Attached left brow to GPIO"); Serial.println(LEFT_BROW_PIN);
  Serial.print("[Servo] Attached right brow to GPIO"); Serial.println(RIGHT_BROW_PIN);
  Serial.print("[Servo] Attached mouth to GPIO"); Serial.println(MOUTH_PIN);

  // Start at neutral
  leftBrow.write(90);
  rightBrow.write(90);
  mouth.write(90);
  Serial.println("[ESP32] Servos initialised");

  // Run startup test
  startupAnimation();
}




// ─────────────────────────────────────────────────────────────────
//  Loop — read serial, dispatch expression
// ─────────────────────────────────────────────────────────────────

void loop() {
  static String incomingLine = "";

  // Read all available bytes from UART2
  while (Serial2.available()) {
    char c = Serial2.read();
    if(c == '\n'){
      if (incomingLine.length() > 0) {
        int code = incomingLine.toInt();
        incomingLine = "";

        if(code >=0 && code <= 3){
          pendingExpressionCode = code;
        }
      }
    }
    else if (c >= '0' && c <= '3'){
      incomingLine += c;

      //Safety, prevent runaway strings
      if(incomingLine.length() > 4){
        incomingLine = "";
      }
    }
    else{
      incomingLine = "";
      return;
    }
  }
      
  // Reocess new expression
  if(pendingExpressionCode != -1 && pendingExpressionCode != lastExpressionCode){
    lastExpressionCode = pendingExpressionCode;
    Serial.print("[ESP32] Received class code: ");
    Serial.println(lastExpressionCode);

    switch(lastExpressionCode){
      case 0: showAngry(); break;
      case 1: showHappy(); break;
      case 2: showSad(); break;
      case 3: showNeutral(); break;
    }
  }

    pendingExpressionCode = -1; // reset pending code

    // Idle behaviour
    switch(lastExpressionCode){
      default: showNeutral(); break; 
    }
}
