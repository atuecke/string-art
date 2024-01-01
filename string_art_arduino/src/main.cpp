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

struct Anchors{
  float startAnchor;
  float endAnchor;
};

Anchors currentAnchors;

Anchors parseInstruction(String instruction){
  Serial.println("Next line: " + instruction);

  int commaIndex = instruction.indexOf(',');

  // Extract instruction values
  Anchors anchors;
  anchors.startAnchor = instruction.substring(1, commaIndex).toFloat();
  anchors.endAnchor = instruction.substring(commaIndex + 2).toFloat();

  return anchors;
}

long radiansToSteps(float radians){
  return (long)(radians * (stepsPerRevolution / (2 * PI)));
}

void raiseActuator(){
  Serial.println("Raising actuator");
  digitalWrite(actuatorPin1, HIGH);
  digitalWrite(actuatorPin2, LOW);
  delay(actuatorRunDuration);
  Serial.println("Done");
}

void lowerActuator(){
  Serial.println("Lowering actuator");
  digitalWrite(actuatorPin1, LOW);
  digitalWrite(actuatorPin2, HIGH);
  delay(actuatorRunDuration);
}

void rotateToAnchor(float anchor){
  long anchorPos = radiansToSteps(anchor);
  Serial.println("Moving to: " + String(anchorPos));
  stepper.moveTo(anchorPos);
  stepper.runToPosition();
}

void setup() {
  // Sets the two pins as Outputs
  pinMode(stepPin,OUTPUT); 
  pinMode(dirPin,OUTPUT);
  pinMode(actuatorPin1,OUTPUT);
  pinMode(actuatorPin2,OUTPUT);

  stepper.setMaxSpeed(1000);
  stepper.setAcceleration(250);

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

  rotateToAnchor(0);
  raiseActuator();

}

void loop() {
  if(isFileOpen && myFile.available()){
    Serial.print("Reading next line... ");
    String line = myFile.readStringUntil(',');
    Serial.println(line);
    if(line == "R" || line == "L"){
      Serial.println("test");
    }
    if(line == "R"){
      raiseActuator();
    }else if(line == "L"){
      lowerActuator();
    }else{
      float anchor = line.toFloat();
      rotateToAnchor(anchor);
    }
  }else if(isFileOpen){
    myFile.close();
    isFileOpen = false;
    Serial.println("All instructions complete.");
  }
}