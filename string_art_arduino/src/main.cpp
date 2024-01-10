#include <Arduino.h>
#include <SPI.h>
#include <AccelStepper.h>
#include <SD.h>

// defines pins
#define stepPin 2
#define dirPin 3
#define actuatorPin1 4
#define actuatorPin2 5
#define SDPin 8

File myFile;
bool isFileOpen = false;
AccelStepper stepper(1, stepPin, dirPin);
const float stepsPerRevolution = 3200;

const unsigned long actuatorRunDuration = 2000;
unsigned long actuatorStartTime = 0;

enum State{
  IDLE,
  READING,
  MOVING,
  ACTUATING
};

State currentState = IDLE;

long radiansToSteps(float radians){
  return (long)(radians * (stepsPerRevolution / (2 * PI)));
}

void raiseActuator(){
  Serial.println("Raising actuator");
  digitalWrite(actuatorPin1, LOW);
  digitalWrite(actuatorPin2, HIGH);
  actuatorStartTime = millis();
  currentState = ACTUATING;

}

void lowerActuator(){
  Serial.println("Lowering actuator");
  digitalWrite(actuatorPin1, HIGH);
  digitalWrite(actuatorPin2, LOW);
  actuatorStartTime = millis();
  currentState = ACTUATING;
}

void stopActuator(){
  digitalWrite(actuatorPin1, LOW);
  digitalWrite(actuatorPin2, LOW);
}

void rotateToAnchor(float anchor){
  long anchorPos = radiansToSteps(anchor);
  Serial.println("Moving to: " + String(anchorPos));
  stepper.moveTo(anchorPos);
  currentState = MOVING;
}

String parseNextInstruction(){
  Serial.print("Reading next line... ");
  String line = myFile.readStringUntil(',');
  Serial.println(line);
  return line;
}

void executeInstruction(String instruction){
  char command = instruction.charAt(0);

  switch(command){
      case 'R':
        raiseActuator();
        break;
      case 'L':
        lowerActuator();
        break;
      case 'M':
        float anchor = instruction.substring(1).toFloat();
        rotateToAnchor(anchor);
        break;
      default:
        Serial.print("Unknown command: ");
        Serial.println(command);
        break;
    }
}

void setup() {
  // Sets the two pins as Outputs
  pinMode(stepPin,OUTPUT); 
  pinMode(dirPin,OUTPUT);
  pinMode(actuatorPin1,OUTPUT);
  pinMode(actuatorPin2,OUTPUT);

  stepper.setMaxSpeed(4000);
  stepper.setAcceleration(1000);

  Serial.begin(9600);
  while(!Serial){
    ; //wait for serial port to connect
  }
  Serial.println("Initializing SD card...");

  if(!SD.begin(SDPin)){
    Serial.println("Initialization failed!");
  }
  Serial.println("Initialization done");
  
  //open file, only one file can be open at a time
  Serial.println("Opening file...");
  myFile = SD.open("file.txt", FILE_READ);
  if(myFile){
    isFileOpen = true;
  } else {
    Serial.println("Error opening file!");
  }
}

void loop() {
  stepper.run();
  switch(currentState){
    case IDLE:
      if(isFileOpen && myFile.available()){
        currentState = READING;
      }
      break;
    case READING:
      if(myFile.available()){
        String instruction = parseNextInstruction();
        executeInstruction(instruction);
      }else{
        myFile.close();
        isFileOpen = false;
        currentState = IDLE;
        Serial.println("All instructions complete.");
      }
      break;
    case MOVING:
      if(stepper.distanceToGo() == 0){
        currentState = IDLE;
      }
      break;
    case ACTUATING:
      if (millis() - actuatorStartTime >= actuatorRunDuration) {
        stopActuator();
        currentState = IDLE;
      }
  }
}