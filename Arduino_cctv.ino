#include <Servo.h>

Servo myServo;  // create servo object to control a servo
int servoPin = 9;  // PWM pin connected to the servo
int angle = 90;  // initial angle for the servo

void setup() {
  myServo.attach(servoPin);  // attaches the servo on pin 9 to the servo object
  myServo.write(angle);  // set servo to initial position
  Serial.begin(9600);  // start serial communication at 9600 baud
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // read the input from serial port
    angle = input.toInt();  // convert the input to an integer

    // constrain the angle to be between 0 and 180 degrees
    angle = constrain(angle, 0, 180);

    myServo.write(angle);  // move the servo to the specified angle
  }
}
