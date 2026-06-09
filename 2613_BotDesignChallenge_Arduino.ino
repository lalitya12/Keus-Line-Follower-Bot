#include <SoftwareSerial.h>
SoftwareSerial espSerial(2, 3);



#define IMU_Reset A2
#define RadarScan A4
#define LL_SENSOR 9
#define L_SENSOR 13
#define M_SENSOR 8
#define R_SENSOR 4
#define RR_SENSOR A3

#define MOTOR1_EN 5
#define MOTOR1_IN1 6
#define MOTOR1_IN2 7

#define MOTOR2_EN 10
#define MOTOR2_IN1 12
#define MOTOR2_IN2 11

#define obsR A1
#define obsL A0

#define NormalSpeed 255

unsigned long lastAvoidTime = 0; // Stores the time when avoidance finished
bool lastRight = true;
int f = 0;

struct Motor {
  int EN, IN1, IN2, pwmValue;
};

Motor motor1, motor2;

void initMotor(Motor &m) {
  pinMode(m.EN, OUTPUT);
  pinMode(m.IN1, OUTPUT);
  pinMode(m.IN2, OUTPUT);
}

void setMotor(Motor &m, int pwm, bool forward) {
  digitalWrite(m.IN1, forward ? HIGH : LOW);
  digitalWrite(m.IN2, forward ? LOW : HIGH);
  analogWrite(m.EN, constrain(pwm, 0, 255));
}

// Directional Helpers
void moveForward(Motor &m1, Motor &m2) { setMotor(m1, NormalSpeed, false); setMotor(m2, NormalSpeed, false); }
void moveBackward(Motor &m1, Motor &m2) { setMotor(m1, NormalSpeed, true); setMotor(m2, NormalSpeed, true); }
void turnLeft(Motor &m1, Motor &m2, double frac) { setMotor(m1, NormalSpeed*frac, false); setMotor(m2, NormalSpeed*frac, true); }
void turnRight(Motor &m1, Motor &m2, double frac) { setMotor(m1, NormalSpeed*frac, true); setMotor(m2, NormalSpeed*frac, false); }
void stopMotors(Motor &m1, Motor &m2) { setMotor(m1, 0, true); setMotor(m2, 0, true); }

// ==========================================
// UPDATED READ YAW (Matches your working code)
// ==========================================
double ReadYaw() {
  unsigned long timeout = millis();
 
  while (true) {
    // Wait for data
    while (!espSerial.available()) {
      if (millis() - timeout > 500) return -999; // Timeout
    }

    String msg = espSerial.readStringUntil('\n');
    msg.trim(); // Clean up stray \r or spaces
   
    // ONLY accept the data if it actually contains "IMU:"
    if (msg.startsWith("IMU:")) {
      msg.remove(0, 4);
      double yaw = msg.toDouble();
      Serial.print("YAW:");
      Serial.println(yaw);
      return yaw;
    }
    // If it was text like "Obstacle", the loop just ignores it and tries again!
  }
}
void RotateTill90(Motor &m1, Motor &m2, bool lastR)  {
  // 1. Send a proper LOW-to-HIGH pulse to trigger the ESP32 reset
  digitalWrite(IMU_Reset, LOW);
  delay(50);
  digitalWrite(IMU_Reset, HIGH); // Leave it HIGH so the ESP32 tracks the angle
  delay(100);

  // 2. Clear the buffer of old data AFTER the reset
  while(espSerial.available()) espSerial.read();

  double yStart = ReadYaw();
  double yCur = yStart;

  if(lastR) {
    while(abs(yCur - yStart) < 70) {
      turnRight(m1, m2, 0.8);
      double temp = ReadYaw();
      if(temp != -999) yCur = temp;
    }
  } else {
    while(abs(yCur - yStart) < 70) {
      turnLeft(m1, m2, 0.8);
      double temp = ReadYaw();
      if(temp != -999) yCur = temp;
    }
  }
  stopMotors(m1, m2);
}

void FullScan()
{
  digitalWrite(RadarScan, HIGH);
  unsigned long timeout = millis();
  while (millis() - timeout < 5000) {
    if (espSerial.available()) {
      String feedback = espSerial.readStringUntil('\n');
      // If the message contains "DONE", the ESP has finished its task
      if (feedback.indexOf("DONE") >= 0) {
        break;
      }
    }
  }
  digitalWrite(RadarScan, LOW);
  return;
}
// ... (Rest of MoveForwardTillLine and MoveForwardTillAway remains same)

void MoveForwardTillLine(Motor &m1, Motor &m2, bool lastR)  {
    bool check = true;
    if(lastR) {
      while(digitalRead(RR_SENSOR) != 1 || (digitalRead(obsL) == 0)) {
        moveForward(m1, m2);
        delay(10);
        if(digitalRead(obsL) == 1){
          check  = false;
          delay(500);
          }
      }
      stopMotors(m1, m2);
    } else {
      while(digitalRead(LL_SENSOR) != 1 || (digitalRead(obsR) == 0)) {
        moveForward(m1, m2);
        delay(10);
        if(digitalRead(obsR) == 1){
          check  = false;
          delay(500);
          }
      }
      stopMotors(m1, m2);
    }
    moveBackward(m1,m2);
    delay(150);
    stopMotors(m1,m2);
    FullScan();
    if(lastR) {
      while(digitalRead(RR_SENSOR) != 1) {
        moveForward(m1, m2);
        delay(10);
      }
      turnRight(m1, m2, 1);
      delay(150);
    } else {
      while(digitalRead(LL_SENSOR) != 1) {
        moveForward(m1, m2);
        delay(10);
      }
      turnLeft(m1, m2, 1);
      delay(150);
    }
    moveForward(m1,m2);
}

void MoveForwardTillLineSpecial(Motor &m1, Motor &m2, bool lastR)  {
    if(lastR) {
      while(digitalRead(RR_SENSOR) != 1) {
        moveForward(m1, m2);
        delay(10);
      }
      stopMotors(m1, m2);
    } else {
      while(digitalRead(LL_SENSOR) != 1) {
        moveForward(m1, m2);
        delay(10);
      }
      stopMotors(m1, m2);
    }
    moveBackward(m1,m2);
    delay(150);
    stopMotors(m1,m2);
    FullScan();
    if(lastR) {
      while(digitalRead(RR_SENSOR) != 1) {
        moveForward(m1, m2);
        delay(10);
      }
      turnRight(m1, m2, 1);
      delay(150);
    } else {
      while(digitalRead(LL_SENSOR) != 1) {
        moveForward(m1, m2);
        delay(10);
      }
      turnLeft(m1, m2, 1);
      delay(150);
    }
    moveForward(m1,m2);
}

void obsAvoid(Motor &m1, Motor &m2, bool rNow)  {
  stopMotors(m1, m2);
  delay(200);
 
  // Go back to give turning space
  unsigned long backTime = millis();
  while(millis() - backTime < 400) {
    moveBackward(m1, m2);
  }
  stopMotors(m1, m2);
  delay(200);

  RotateTill90(m1, m2, rNow);
 
  MoveForwardTillLine(m1, m2, rNow);
  moveForward(m1,m2);
}

void obsAvoidFront(Motor &m1, Motor &m2)  {
  stopMotors(m1, m2);
  delay(200);

  RotateTill90(m1, m2, lastRight);
  moveForward(m1,m2);
  delay(1000);
  stopMotors(m1,m2);
  RotateTill90(m1, m2, !lastRight);
  moveForward(m1,m2);
  delay(2300);
  stopMotors(m1,m2);
  RotateTill90(m1, m2, !lastRight);
  moveForward(m1,m2);
  delay(500);
  stopMotors(m1,m2);
  MoveForwardTillLineSpecial(m1, m2, lastRight);
}

void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);
 
  pinMode(IMU_Reset, OUTPUT);
  digitalWrite(IMU_Reset, LOW); // Start low
  pinMode(RadarScan, OUTPUT);
  digitalWrite(RadarScan, LOW); // Start low

  pinMode(LL_SENSOR, INPUT);
  pinMode(L_SENSOR, INPUT);
  pinMode(M_SENSOR, INPUT);
  pinMode(R_SENSOR, INPUT);
  pinMode(RR_SENSOR, INPUT);
  pinMode(obsL, INPUT);
  pinMode(obsR, INPUT);

  motor1.EN = MOTOR1_EN; motor1.IN1 = MOTOR1_IN1; motor1.IN2 = MOTOR1_IN2;
  initMotor(motor1);
  motor2.EN = MOTOR2_EN; motor2.IN1 = MOTOR2_IN1; motor2.IN2 = MOTOR2_IN2;
  initMotor(motor2);
  stopMotors(motor1, motor2);
}

void loop() {
  // Keep the Reset pin HIGH as in your working code if needed
  digitalWrite(IMU_Reset, LOW);

  int ll = digitalRead(LL_SENSOR);
  int l = digitalRead(L_SENSOR);
  int m = digitalRead(M_SENSOR);
  int r = digitalRead(R_SENSOR);
  int rr = digitalRead(RR_SENSOR);
  Serial.print(ll);
  Serial.print(" ");
  Serial.print(l);
  Serial.print(" ");
  Serial.print(m);
  Serial.print(" ");
  Serial.print(r);
  Serial.print(" ");
  Serial.print(rr);
  Serial.println(" ");
  if(m==1 && l==1 && r==1 && rr==1 && ll==1)
    stopMotors(motor1,motor2);
  if (m == 1) {
    moveForward(motor1, motor2);
    f = 0;
  } else if (l == 1) {
    setMotor(motor1, NormalSpeed, false);
    setMotor(motor2, NormalSpeed/2, false);
    f = 0;
  } else if (r == 1) {
    setMotor(motor1, NormalSpeed/2, false);
    setMotor(motor2, NormalSpeed, false);
    f = 0;
  } else if (ll == 1) {
    lastRight = false;
    turnLeft(motor1, motor2, 1);
    f = 0;
  } else if (rr == 1) {
    lastRight = true;
    turnRight(motor1, motor2, 1);
    f = 0;
  } else {
    if(f < 160) { f++; delay(20); }
    else stopMotors(motor1, motor2);
  } // Obstacle Check
  if (millis() - lastAvoidTime > 3000) {
    // Front Obstacle (Serial)
    if (espSerial.available()) {
      String msg = espSerial.readStringUntil('\n');
      if (msg.startsWith("Obstacle")) {
        obsAvoidFront(motor1, motor2);
        lastAvoidTime = millis(); // Reset cooldown timer
        Serial.println("FRONT AVOID DONE - COOLDOWN START");
      }
    }

    // Side Obstacles (Infrared/Digital)
    if (digitalRead(obsL) == 0) {
      Serial.println("LEFT");
      obsAvoid(motor1, motor2, false);
      lastAvoidTime = millis(); // Reset cooldown timer
    }
    else if (digitalRead(obsR) == 0) {
      Serial.println("RIGHT");
      obsAvoid(motor1, motor2, true);
      lastAvoidTime = millis(); // Reset cooldown timer
    }
  }
}
