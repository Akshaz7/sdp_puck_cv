#include <Arduino.h>
#define STEP1 9
#define DIR1 10

#define STEP2 12
#define DIR2 13

#define MAXX 600
#define MAXY 300

int pos1=0;
int pos2=0;

void stepMotor(int stepPin){
 digitalWrite(stepPin,HIGH);
 delayMicroseconds(60);
 digitalWrite(stepPin,LOW);
}

void stepMotors(int a,int b){

 digitalWrite(DIR1,a>0);
 digitalWrite(DIR2,b>0);

 a=abs(a);
 b=abs(b);

 while(a>0 || b>0){

  if(a>0){
   stepMotor(STEP1);
   a--;
  }

  if(b>0){
   stepMotor(STEP2);
   b--;
  }

  delayMicroseconds(80);
 }
}

void setup(){

 pinMode(STEP1,OUTPUT);
 pinMode(DIR1,OUTPUT);

 pinMode(STEP2,OUTPUT);
 pinMode(DIR2,OUTPUT);

 Serial.begin(115200);
}

void loop(){

 if(Serial.available()){

  int x=Serial.parseInt();
  int y=Serial.parseInt();

  int targetX = map(x,0,MAXX,0,8000);
  int targetY = map(y,0,MAXY,0,8000);

  int t1 = targetY-targetX;
  int t2 = targetX+targetY;

  stepMotors(t1-pos1,t2-pos2);

  pos1=t1;
  pos2=t2;
 }
}