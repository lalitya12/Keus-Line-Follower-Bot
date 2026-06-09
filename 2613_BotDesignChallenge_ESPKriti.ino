#include <ESP32Servo.h>
#include <Wire.h>
#include <MPU6050.h>

// --- ORIGINAL PINS ---
#define TRIG 5
#define ECHO 18
#define SERVO_PIN 19
#define TRIGGER_PIN 4

// --- NEW PINS ---
#define TRIGL 25
#define ECHOL 26
#define TRIGR 27
#define ECHOR 32
#define SERVOC_PIN 13
#define TRIGGERC_PIN 33
#define IR_SENSOR_PIN 14
#define toPI 23
#define fromPI 34
// ----------------

bool lastTriggerState = LOW;
bool lastTriggerStateC = LOW;
unsigned long lastCameraMoveTime = 0; // Tracks the 100ms cooldown

MPU6050 mpu;
Servo radarServo;
Servo cameraServo;

HardwareSerial ArduinoSerial(2);   // UART2

bool tracking = false;

float rawAngle = 0;
float smoothAngle = 0;

float gyroBiasZ = 0;

unsigned long prevTime;
unsigned long lastPrint = 0;

TaskHandle_t Task1;
TaskHandle_t Task2;

void setup() {

  Serial.begin(115200);         // PC Serial Monitor
  ArduinoSerial.begin(9600, SERIAL_8N1, 16, 17);  
  // RX = GPIO16, TX = GPIO17

  radarServo.attach(SERVO_PIN);
  cameraServo.attach(SERVOC_PIN);

  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(TRIGR, OUTPUT);
  pinMode(ECHOR, INPUT);
  pinMode(TRIGL, OUTPUT);
  pinMode(ECHOL, INPUT);
  pinMode(TRIGGER_PIN, INPUT);
  pinMode(TRIGGERC_PIN, INPUT);
  pinMode(IR_SENSOR_PIN, INPUT);
  pinMode(toPI, OUTPUT);
  pinMode(fromPI, INPUT);

  Wire.begin();
  mpu.initialize();

  delay(1000);

  // -------- GYRO CALIBRATION --------
  Serial.println("Calibrating gyro... Keep robot still");

  long sum = 0;

  for(int i = 0; i < 1000; i++)
  {
    int16_t ax,ay,az,gx,gy,gz;
    mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);

    sum += gz;
    delay(2);
  }

  gyroBiasZ = (sum / 1000.0) / 131.0;

  Serial.print("Gyro bias Z = ");
  Serial.println(gyroBiasZ);

  // -------- TASKS --------
  xTaskCreatePinnedToCore(
    imu,
    "IMU",
    10000,
    NULL,
    1,
    &Task1,
    0);

  xTaskCreatePinnedToCore(
    radar,
    "RADAR",
    10000,
    NULL,
    1,
    &Task2,
    1);
}

void loop() {}

long getDistance() {

  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG, LOW);

  long duration = pulseIn(ECHO, HIGH);

  return duration * 0.034 / 2;
}

long getDistanceR() {

  digitalWrite(TRIGR, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIGR, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIGR, LOW);

  long duration = pulseIn(ECHOR, HIGH);

  return duration * 0.034 / 2;
}

long getDistanceL() {

  digitalWrite(TRIGL, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIGL, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIGL, LOW);

  long duration = pulseIn(ECHOL, HIGH);

  return duration * 0.034 / 2;
}


void imu(void * parameter) {

  while(true) {

    int triggerState = digitalRead(TRIGGER_PIN);

    // Trigger pressed
    if (triggerState == HIGH && lastTriggerState == LOW) {

      resetAll();
      startTracking();
    }

    // Trigger released
    if (triggerState == LOW && lastTriggerState == HIGH) {

      stopTracking();
    }

    lastTriggerState = triggerState;

    if (tracking) {
      readIMU();
    }

    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

void startTracking() {

  tracking = true;

  rawAngle = 0;
  smoothAngle = 0;

  prevTime = millis();
}

void stopTracking() {

  tracking = false;
}

void resetAll() {

  rawAngle = 0;
  smoothAngle = 0;

  prevTime = millis();
}

void readIMU() {

  int16_t ax, ay, az, gx, gy, gz;

  mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);

  unsigned long now = millis();

  float dt = (now - prevTime)/1000.0;

  prevTime = now;

  // Remove gyro bias
  float gyroZrate = (gz / 131.0) - gyroBiasZ;

  rawAngle += gyroZrate * dt;

  // smoothing
  smoothAngle = 0.9 * smoothAngle + 0.1 * rawAngle;

  if (millis() - lastPrint > 20) {

    Serial.print("Deviation angle: ");
    Serial.println(smoothAngle);

    // Send IMU data to Arduino
    ArduinoSerial.print("IMU:");
    ArduinoSerial.println(smoothAngle);

    lastPrint = millis();
  }
}

void sweep() {
  for (int angle = 0; angle <= 180; angle += 5) {
      radarServo.write(angle);
      long dist = getDistance();
      if (dist > 0 && dist < 40) {
        Serial.print("Object at angle: ");
        Serial.println(angle);
      }
    }
    for (int angle = 180; angle >= 0; angle -= 5) {
      radarServo.write(angle);
      long dist = getDistance();
      if (dist > 0 && dist < 40) {
        Serial.print("Object at angle: ");
        Serial.println(angle);
      }
    }
  ArduinoSerial.println("DONE");
}

void PI_Signal()
{
  digitalWrite(toPI, HIGH);
  unsigned long timeout = millis();
  while (millis() - timeout < 1000) {
    if (digitalRead(fromPI) == HIGH) {
      break;
    }
  }
  digitalWrite(toPI, LOW);
  return;
}

void radar(void * parameter) {
  int angle = 30;
  int step = 5;

  while(true) {
    digitalWrite(toPI, LOW);
    radarServo.write(angle);
    int triggerStateC = digitalRead(TRIGGERC_PIN);

    if(triggerStateC == 1 && lastTriggerStateC == 0)
      sweep();

    lastTriggerStateC = triggerStateC;
   
    long distR = getDistanceR();
    long distL = getDistanceL();
    long dist = getDistance();

    // Check if 100ms has passed before allowing a camera move
    if (millis() - lastCameraMoveTime > 200) {
      bool moved = false;

      if (distR > 0 && distR < 40) {
        Serial.print("Object at right");
        cameraServo.write(180);
        PI_Signal();
        moved = true;
      }
      else if (distL > 0 && distL < 40) {
        Serial.print("Object at left");
        cameraServo.write(0);
        PI_Signal();
        moved = true;
      }
      else if (dist > 0 && dist < 40) {
        Serial.print("Object at angle: ");
        Serial.println(angle);
        cameraServo.write(angle);
        PI_Signal();
        moved = true;
      }

      // If we actually moved the servo, reset the timer
      if (moved) {
        lastCameraMoveTime = millis();
      }
    }

    // Radar logic continues regardless of whether camera moved
    angle += step;
    int irValue = digitalRead(IR_SENSOR_PIN);
    if (irValue == LOW) {
        ArduinoSerial.println("Obstacle Detected");
    }

    if (angle >= 150) {
      angle = 150;
      step = -5;  
    } else if (angle <= 30) {
      angle = 30;  
      step = 5;    
    }
   
    vTaskDelay(pdMS_TO_TICKS(20)); // Small delay to prevent task hogging
  }
}