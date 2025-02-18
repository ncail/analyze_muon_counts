/* Additions:
Wrapped writes to sd card in if(WRITE), WRITE is #defined at the top of the program.
Writes to sd card are not needed since muon detector will be operating with python script serialLogger.py that writes the serial data to CSV.

Implemented logging (#defined LOGGING at top of program), to debug muon detector crashing.
Logging messages are prefixed with LOG to distinguish them for the python program from other serial data.
*/


/*
  CosmicWatch Desktop Muon Detector Arduino Code

  This code is used to record data to the built in microSD card reader/writer.
  
  Questions?
  Spencer N. Axani
  saxani@mit.edu

  Requirements: Sketch->Include->Manage Libraries:
  SPI, EEPROM, SD, and Wire are probably already installed.
  1. Adafruit SSD1306     -- by Adafruit Version 1.0.1
  2. Adafruit GFX Library -- by Adafruit Version 1.0.2
  3. TimerOne             -- by Jesse Tane et al. Version 1.1.0
*/

#include <SPI.h>
#include <SD.h>
#include <EEPROM.h>


// Define WRITE to toggle on/off if the program writes to the sd card.
// Suspected that writing to sd card is causing the program to crash.
// WRITE = 0 will still receive all serial data.
#define WRITE 0
#define LOGGING 1


#define SDPIN 10
SdFile root;
Sd2Card card;
SdVolume volume;

File myFile;

const int SIGNAL_THRESHOLD    = 50;        // Min threshold to trigger on
const int RESET_THRESHOLD     = 25; 

const int LED_BRIGHTNESS      = 255;         // Brightness of the LED [0,255]

//Calibration fit data for 10k,10k,249,10pf; 20nF,100k,100k, 0,0,57.6k,  1 point
const long double cal[] = {-9.085681659276021e-27, 4.6790804314609205e-23, -1.0317125207013292e-19,
  1.2741066484319192e-16, -9.684460759517656e-14, 4.6937937442284284e-11, -1.4553498837275352e-08,
   2.8216624998078298e-06, -0.000323032620672037, 0.019538631135788468, -0.3774384056850066, 12.324891083404246};
   
const int cal_max = 1023;

//initialize variables
char detector_name[40];

unsigned long time_stamp                    = 0L;
unsigned long measurement_deadtime          = 0L;
unsigned long time_measurement              = 0L;      // Time stamp
unsigned long interrupt_timer               = 0L;      // Time stamp
int           start_time                    = 0L;      // Start time reference variable
long int      total_deadtime                = 0L;      // total time between signals

unsigned long measurement_t1;
unsigned long measurement_t2;

float temperatureC;


long int      count                         = 0L;         // A tally of the number of muon counts observed
float         last_adc_value                = 0;
char          filename[]                    = "File_000.txt";
int           Mode                          = 1;

byte SLAVE;
byte MASTER;
byte keep_pulse;


void setup() {
  analogReference (EXTERNAL);
  ADCSRA &= ~(bit (ADPS0) | bit (ADPS1) | bit (ADPS2));    // clear prescaler bits
  //ADCSRA |= bit (ADPS1);                                   // Set prescaler to 4  
  ADCSRA |= bit (ADPS0) | bit (ADPS1); // Set prescaler to 8
  
  get_detector_name(detector_name);
  pinMode(3, OUTPUT); 
  pinMode(6, INPUT);
  
  Serial.begin(9600);
  Serial.setTimeout(3000);

  if (digitalRead(6) == HIGH){
     filename[4] = 'S';
     SLAVE = 1;
     MASTER = 0;
  }
  
  else{
     //delay(10);
     filename[4] = 'M';
     MASTER = 1;
     SLAVE = 0;
     pinMode(6, OUTPUT);
     digitalWrite(6,HIGH);
     //delay(2000);
    }
    
  if (!SD.begin(SDPIN)) {
    Serial.println(F("SD initialization failed!"));
    Serial.println(F("Is there an SD card inserted?"));
    return;
  }
  
  get_Mode();
  if (Mode == 2) read_from_SD();
  else if (Mode == 3) remove_all_SD();
  // else{setup_files();}
  else if(WRITE){setup_files();}
  
  if (MASTER == 1){digitalWrite(6,LOW);}
  analogRead(A0);
  
  
  start_time = millis();
}

void loop() {
  if(Mode == 1){
    Serial.println(F("##########################################################################################"));
    Serial.println(F("### CosmicWatch: The Desktop Muon Detector"));
    Serial.println(F("### Questions? saxani@mit.edu"));
    Serial.println(F("### Comp_date Comp_time Event Ardn_time[ms] ADC[0-1023] SiPM[mV] Deadtime[ms] Temp[C] Name"));
    Serial.println(F("##########################################################################################"));
    Serial.println("Device ID: " + (String)detector_name);

    if(WRITE){
      myFile.println(F("##########################################################################################"));
      myFile.println(F("### CosmicWatch: The Desktop Muon Detector"));
      myFile.println(F("### Questions? saxani@mit.edu"));
      myFile.println(F("### Comp_date Comp_time Event Ardn_time[ms] ADC[0-1023] SiPM[mV] Deadtime[ms] Temp[C] Name"));
      myFile.println(F("##########################################################################################"));
      myFile.println("Device ID: " + (String)detector_name);
    }

    Serial.println("DATA START");

    write_to_SD();
  }
}

void setup_files(){   
  for (uint8_t i = 1; i < 201; i++) {
      int hundreds = (i-i/1000*1000)/100;
      int tens = (i-i/100*100)/10;
      int ones = i%10;
      filename[5] = hundreds + '0';
      filename[6] = tens + '0';
      filename[7] = ones + '0';
      if (! SD.exists(filename)) {
          Serial.println("Creating file: " + (String)filename);
          if (SLAVE ==1){
           digitalWrite(3,HIGH);
           delay(1000);
           digitalWrite(3,LOW);
          }
          delay(500);
          myFile = SD.open(filename, FILE_WRITE); 
          break;  
      }
   }
}

void write_to_SD(){ 
  while (1){

    int adc = analogRead(A0);
    
    // Log analogRead(A0).
    // if(LOGGING)
    // {
    //   // char logMessage[] = "LOG Operation 1 in write_to_SD -- analogRead(A0): "; 
    //   // int analogValA0 = analogRead(A0);
    //   Serial.print("LOG Operation 1 in write_to_SD -- analogRead(A0): ");
    //   Serial.println(adc);
    // }

    // if (analogRead(A0) > SIGNAL_THRESHOLD){  // Possibly buggy, define adc at top of function instead instead of analogRead(A0) twice.
    if (adc > SIGNAL_THRESHOLD){
      // int adc = analogRead(A0);

      // Log digitalWrite(6, HIGH).
      if(LOGGING)
      {
        Serial.println("LOG Operation 2 in write_to_SD -- digitalWrite(6, HIGH)");
      }
      
      if (MASTER == 1) 
      {
          digitalWrite(6, HIGH);
          count++;
          keep_pulse = 1;
      }

      // Log analogRead(A3).
      if(LOGGING)
      {
        Serial.print("LOG Operation 3 in write_to_SD -- analogRead(A3): ");
        Serial.println(analogRead(A3));
      }
      
      analogRead(A3);  // Why are we reading temperature here?
      
      // Only when operating in coincidence mode.
      if (SLAVE == 1)
      {
          if (digitalRead(6) == HIGH)
          {
              keep_pulse = 1;
              count++;
          }
      } 

      // Log analogRead(A3).
      if(LOGGING)
      {
        Serial.print("LOG Operation 4 in write_to_SD -- analogRead(A3): ");
        Serial.println(analogRead(A3));
      }

      analogRead(A3);  // Why are we reading temperature here?
      
      if (MASTER == 1)
      {
          // Log digitalWrite(6, LOW).
          if(LOGGING)
          {
            Serial.println("LOG Operation 5 in write_to_SD -- digitalWrite(6, LOW)");
          }

          digitalWrite(6, LOW);
      }

      measurement_deadtime = total_deadtime;
      time_stamp = millis() - start_time;
      measurement_t1 = micros();  

      // Log analogRead(A3).
      if(LOGGING)
      {
        Serial.print("LOG Operation 6 in write_to_SD -- analogRead(A3): ");
        Serial.println(analogRead(A3));
      }

      temperatureC = (((analogRead(A3)+analogRead(A3)+analogRead(A3))/3. * (3300./1024)) - 500)/10. ;  // Reads temp three times, returns 3 potentially different values, gets average.

      // Log temperatureC.
      if(LOGGING)
      {
        Serial.print("LOG Operation 7 in write_to_SD -- temperatureC: ");
        Serial.println(temperatureC);
      }

      if (MASTER == 1) 
      {
          // Log digitalWrite(6, LOW).
          if(LOGGING)
          {
            Serial.println("LOG Operation 8 in write_to_SD -- digitalWrite(6, LOW)");
          }

          digitalWrite(6, LOW); 
          analogWrite(3, LED_BRIGHTNESS);

          Serial.println((String)count + " " + time_stamp+ " " + adc+ " " + get_sipm_voltage(adc)+ " " + measurement_deadtime+ " " + temperatureC);

          if(WRITE)
          {
            myFile.println((String)count + " " + time_stamp+ " " + adc+ " " + get_sipm_voltage(adc)+ " " + measurement_deadtime+ " " + temperatureC);
            myFile.flush();
          }
          
          last_adc_value = adc;
      }
  
      // Only when operating in coincidence mode.
      if (SLAVE == 1) {
          if (keep_pulse == 1)
          {   
              analogWrite(3, LED_BRIGHTNESS);
              Serial.println((String)count + " " + time_stamp+ " " + adc+ " " + get_sipm_voltage(adc)+ " " + measurement_deadtime+ " " + temperatureC);

              if(WRITE)
              {
                myFile.println((String)count + " " + time_stamp+ " " + adc+ " " + get_sipm_voltage(adc)+ " " + measurement_deadtime+ " " + temperatureC);
                myFile.flush();
              }
              
              last_adc_value = adc;
          }
      }
              
      keep_pulse = 0;

      // Log digitalWrite(3, LOW).
      if(LOGGING)
      {
        Serial.println("LOG Operation 9 in write_to_SD -- digitalWrite(3, LOW)");
      }

      digitalWrite(3, LOW);

      // Log analogRead(A0).
      if(LOGGING)
      {
        Serial.print("LOG Operation 10 in write_to_SD -- analogRead(A0): ");
        Serial.println(analogRead(A0));
      }

      while(analogRead(A0) > RESET_THRESHOLD)
      {
        continue;
      }
      
      total_deadtime += (micros() - measurement_t1) / 1000.;}
    }
}

void read_from_SD(){
    while(true){
    if(SD.exists("File_210.txt")){
      SD.remove("File_209.txt");
      SD.remove("File_208.txt");
      SD.remove("File_207.txt");
      SD.remove("File_206.txt");
      SD.remove("File_205.txt");
      SD.remove("File_204.txt");
      SD.remove("File_203.txt");
      SD.remove("File_202.txt");
      SD.remove("File_201.txt");
      SD.remove("File_200.txt");
      }
    
    for (uint8_t i = 1; i < 211; i++) {
      
      int hundreds = (i-i/1000*1000)/100;
      int tens = (i-i/100*100)/10;
      int ones = i%10;
      filename[5] = hundreds + '0';
      filename[6] = tens + '0';
      filename[7] = ones + '0';
      filename[4] = 'M';

      if (SD.exists(filename)) {
          delay(10);  
          File dataFile = SD.open(filename);
          Serial.println("opening: " + (String)filename);
          while (dataFile.available()) {
              Serial.write(dataFile.read());
              }
          dataFile.close();
          Serial.println("EOF");
        }
      filename[4] = 'S';
      if (SD.exists(filename)) {
          delay(10);  
          File dataFile = SD.open(filename);
          Serial.println("opening: " + (String)filename);
          while (dataFile.available()) {
              Serial.write(dataFile.read());
              }
          dataFile.close();
          Serial.println("EOF");
        }
      }  
    
    Serial.println("Done...");
    break;
  }
  
}

void remove_all_SD() {
  while(true){
    for (uint8_t i = 1; i < 211; i++) {
      
      int hundreds = (i-i/1000*1000)/100;
      int tens = (i-i/100*100)/10;
      int ones = i%10;
      filename[5] = hundreds + '0';
      filename[6] = tens + '0';
      filename[7] = ones + '0';
      filename[4] = 'M';
      
      if (SD.exists(filename)) {
          delay(10);  
          Serial.println("Deleting file: " + (String)filename);
          SD.remove(filename);   
        }
      filename[4] = 'S';
      if (SD.exists(filename)) {
          delay(10);  
          Serial.println("Deleting file: " + (String)filename);
          SD.remove(filename);   
        }   
    }
    Serial.println("Done...");
    break;
  }
  write_to_SD();
}

void get_Mode(){ //fuction for automatic port finding on PC
    Serial.println("CosmicWatchDetector");
    Serial.println(detector_name);
    String message = "";
    message = Serial.readString();
    if(message == "write"){
      delay(1000);
      Mode = 1;
    }
    else if(message == "read"){
      delay(1000);
      Mode =  2;
    }
    else if(message == "remove"){
      delay(1000);
      Mode = 3;
    }
}

float get_sipm_voltage(float adc_value){
  float voltage = 0;
  for (int i = 0; i < (sizeof(cal)/sizeof(float)); i++) 
  {
    voltage += cal[i] * pow(adc_value,(sizeof(cal)/sizeof(float)-i-1));
  }

    // Log voltage.
    if(LOGGING)
    {
      Serial.print("LOG Operation 1 in get_sipm_voltage -- voltage: ");
      Serial.println(voltage);
    }

    return voltage;
}

boolean get_detector_name(char* det_name) 
{
    byte ch;                              // byte read from eeprom
    int bytesRead = 0;                    // number of bytes read so far
    ch = EEPROM.read(bytesRead);          // read next byte from eeprom
    det_name[bytesRead] = ch;               // store it into the user buffer
    bytesRead++;                          // increment byte counter

    while ( (ch != 0x00) && (bytesRead < 40) && ((bytesRead) <= 511) ) 
    {
        ch = EEPROM.read(bytesRead);
        det_name[bytesRead] = ch;           // store it into the user buffer
        bytesRead++;                      // increment byte counter
    }
    if ((ch != 0x00) && (bytesRead >= 1)) {det_name[bytesRead - 1] = 0;}
    return true;
}
