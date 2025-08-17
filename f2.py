#include <TinyGPS++.h>
#include <SoftwareSerial.h>
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <SD.h>
#include "LoRa_E32.h"
#include <EEPROM.h>

#define BME_SCK 13
#define BME_MISO 12
#define BME_MOSI 11
#define BME_CS 10
#define SEALEVELPRESSURE_HPA (1013.25)

String durum = "Roket Bekliyor";
double baseline = 0;
double h1;
int dataCounter = 0;
int fallCounter = 0;
int firstCounter = 0;
int bmpFailCounter = 0;
int RXPin = 9;
int TXPin = 10;
int GPSBaud = 9600;
int paketno = 0;
int STEP = -1;
int Patlatma1 = 21;
int Patlatma2 = 22;
int a;
float accY = 10;
float x = 0;
float y = 10;
float z = 0;
float xd = 0;
float yd = 0;
float zd = 0;
float kalman_old8 = 0;
float cov_old8 = 1;
float h;
float h_before = 0;
float baselat = 0;
float baselng = 0;
float basealt = 0;
float e;
float f;
float b;
float k;
float orx = 0;
float ory = 0;
float orz = 0;
float el = 0;
float avgArray = 0;
float avgArray2 = 0;
float bmpArray = 0;
float avgbmpArray = 0;
float avgbmpArray2 = 0;
long timer = 0;
long timer2 = 0;
long timer3 = 0;
long timer4 = 0;
long timer5 = 0;
const int chipSelect = BUILTIN_SDCARD;

Adafruit_BME280 bme;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
TinyGPSPlus gps;
SoftwareSerial gpsSerial(RXPin, TXPin);
SoftwareSerial mySerial(0, 1);
LoRa_E32 e32ttl(&mySerial);
File myFile;

struct Signal
{
    byte a[4];
    byte b[4];
    byte c[4];
    byte d[4];
    byte e[4];
    byte f[4];
    byte g[4];
    byte h[4];
    byte j[4];
} data;

void setup()
{
    Serial.begin(9600);
    gpsSerial.begin(GPSBaud);
    bme.begin();
    bno.begin();
    SD.begin(chipSelect);
    e32ttl.begin();

    baseline = bme.readAltitude(SEALEVELPRESSURE_HPA);
    baselat = gps.location.lat();
    baselng = gps.location.lng();
    basealt = gps.altitude.meters();

    pinMode(Patlatma1, OUTPUT);
    pinMode(Patlatma2, OUTPUT);

    digitalWrite(Patlatma1, LOW);
    digitalWrite(Patlatma2, LOW);

    paketno = EEPROM.read(0);
}

void loop()
{
    calculateData();
    fileWrite();
    algorithm();
    isImuOkay();
    isBmpOkay();
    sendLora();
}

bool isBmpOkay()
{
    double bmpArray[10];
    for (int k = 0; k < 10; k++)
    {
        bmpArray[k] = h;
    }
    for (int l = 0; l < 10; l++)
    {
        avgbmpArray += bmpArray[l];
        avgbmpArray2 = avgbmpArray / 10;
    }
    if (avgbmpArray2 == 0 || avgbmpArray2 > 4000)
    {
        avgbmpArray2 = 0;
        avgbmpArray = 0;
        return false;
    }
    else
    {
        avgbmpArray2 = 0;
        avgbmpArray = 0;
        return true;
    }
}

bool isImuOkay()
{
    double imuArray[10];
    for (int i = 0; i < 10; i++)
    {
        imuArray[i] = accY;
    }
    for (int j = 0; j < 10; j++)
    {
        avgArray += imuArray[j];
        avgArray2 = avgArray / 10;
    }
    if (avgArray2 == 0)
    {
        avgArray2 = 0;
        return false;
    }
    else
    {
        avgArray2 = 0;
        return true;
    }
}

void calculateData()
{
    e = bme.readTemperature();
    f = bme.readPressure() / 1000;
    h1 = bme.readAltitude(SEALEVELPRESSURE_HPA) - baseline;
    h = kalman_filter_bme(h1);
    k = gps.altitude.meters();
    sensors_event_t accelerometerData, orientationData;
    bno.getEvent(&accelerometerData, Adafruit_BNO055::VECTOR_ACCELEROMETER);
    bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
    printEvent(&accelerometerData);
    printEvent(&orientationData);
    uint8_t system, gyro, accel, mag;
    bno.getCalibration(&system, &gyro, &accel, &mag);

    Serial.print(a);
    Serial.print(" * ");
    Serial.print("401511");
    Serial.print(" * ");
    Serial.print(e);
    Serial.print(" * ");
    Serial.print(f);
    Serial.print(" * ");
    Serial.print(h);
    Serial.print(" * ");
    Serial.print(el);
    Serial.print(" * ");

    while (gpsSerial.available() > 0)
    {
        if (gps.encode(gpsSerial.read()))
        {
        }
    }
    if (gps.location.isValid())
    {
        Serial.print(gps.location.lat(), 6);
        Serial.print(" * ");
        Serial.print(gps.location.lng(), 6);
        Serial.print(" * ");
        Serial.print(gps.altitude.meters());
        Serial.print(" * ");
    }
    else
    {
        Serial.print(baselat);
        Serial.print(" * ");
        Serial.print(baselng);
        Serial.print(" * ");
        Serial.print(basealt);
        Serial.print(" * ");
    }

    Serial.print(" * ");
    Serial.print(orx);
    Serial.print(" * ");
    Serial.print(ory);
    Serial.print(" * ");
    Serial.print(orz);
    Serial.print(" * ");
    Serial.print(x);
    Serial.print(" * ");
    Serial.print(accY);
    Serial.print(" * ");
    Serial.print(z);
    Serial.print(" * ");
    Serial.print(STEP);
    Serial.print(" * ");
    Serial.print(isImuOkay());
    Serial.print(" * ");
    Serial.print(isBmpOkay());
    Serial.print(" * ");
    Serial.println(durum);
}

void fileWrite()
{
    myFile = SD.open("AresAna.txt", FILE_WRITE);
    if (myFile)
    {
        myFile.print("401511 ");
        myFile.print(" * ");
        myFile.print(e);
        myFile.print(" * ");
        myFile.print(f);
        myFile.print(" * ");
        myFile.print(h);
        myFile.print(" * ");
        myFile.print(el);
        myFile.print(" * ");
        myFile.print(gps.location.lat(), 6);
        myFile.print(" * ");
        myFile.print(gps.location.lng(), 6);
        myFile.print(" * ");
        myFile.print(gps.altitude.meters());
        myFile.print(" * ");
        myFile.print(orx);
        myFile.print(" * ");
        myFile.print(ory);
        myFile.print(" * ");
        myFile.println(orz);
        myFile.close();
    }
    EEPROM.write(0, paketno);
}

void algorithm()
{
    if (STEP == -1)
    {
        if (millis() - timer3 > 1000)
        {
            firstCounter = firstCounter + 1;
            timer3 = millis();
        }
        if (firstCounter >= 6)
        {
            STEP = 0;
        }
        a = 0;
    }
    else if (STEP == 0)
    {
        if (isImuOkay() == 1 && isBmpOkay() == 0)
        {

            durum = "Roket Yukseliyor";
            STEP = 1;
            a = 1;
        }
        if (isBmpOkay() == 1 && isImuOkay() == 0)
        {
            if (h > 10)
            {
                durum = "Roket Yukseliyor";
                STEP = 1;
                a = 1;
            }
        }
        if (isBmpOkay() == 1 && isImuOkay() == 1)
        {
            if (h > 10)
            {
                durum = "Roket Yukseliyor";
                STEP = 1;
                a = 1;
            }
        }
    }
    else if (STEP == 1)
    {
        if (millis() - timer > 1000)
        {
            dataCounter = dataCounter + 1;
            timer = millis();
        }
        if (dataCounter >= 6)
        {
            STEP = 2;
        }
        a = 1;
    }
    else if (STEP == 2)
    {
        if (isImuOkay() == 1 && isBmpOkay() == 0)
        {
            if (el < 30)
            {
                STEP = 3;
            }
        }
        if (isBmpOkay() == 1 && isImuOkay() == 0)
        {
            if (h > 1000)
            {
                STEP = 3;
            }
        }
        if (isBmpOkay() == 1 && isImuOkay() == 1)
        {
            if (h > 1000 && abs(el) < 30)
            {
                STEP = 3;
            }
        }
        a = 1;
    }
    else if (STEP == 3)
    {
        if (isImuOkay() == 1 && isBmpOkay() == 1)
        {
            if (h - h_before < 5 && abs(el) < 40)
            {
                digitalWrite(Patlatma1, HIGH);
                delay(1000);
                digitalWrite(Patlatma1, LOW);
                durum = "1. Patlatma";
                STEP = 4;
                a = 2;
            }
        }
        if (isImuOkay() == 0 && isBmpOkay() == 1)
        {
            if (h - h_before < 5)
            {
                digitalWrite(Patlatma1, HIGH);
                delay(1000);
                digitalWrite(Patlatma1, LOW);
                durum = "1. Patlatma";
                STEP = 4;
                a = 2;
            }
        }
        if (isBmpOkay() == 0 && isImuOkay() == 1)
        {
            if (abs(el) < 40)
            {
                digitalWrite(Patlatma1, HIGH);
                delay(1000);
                digitalWrite(Patlatma1, LOW);
                durum = "1. Patlatma";
                STEP = 4;
                a = 2;
            }
        }
        h_before = h;
    }
    else if (STEP == 4)
    {
        if (millis() > timer2)
        {
            fallCounter = fallCounter + 1;
            timer2 = millis();
        }
        if (fallCounter >= 1)
        {
            STEP = 5;
        }
        a = 2;
    }
    else if (STEP == 5)
    {
        if (isBmpOkay() == 1)
        {
            if (h < 600)
            {
                digitalWrite(Patlatma2, HIGH);
                delay(1000);
                digitalWrite(Patlatma2, LOW);
                durum = "2. Patlatma";
                STEP = 6;
                a = 4;
            }
        }
        else if (isBmpOkay() == 0)
        {
            if (millis() - timer5 > 1000)
            {
                bmpFailCounter = bmpFailCounter + 1;
                timer5 = millis();
            }
            if (bmpFailCounter == 5)
            {
                digitalWrite(Patlatma2, HIGH);
                delay(1000);
                digitalWrite(Patlatma2, LOW);
                durum = "2. Patlatma";
                STEP = 6;
                a = 4;
            }
        }
    }
    else if (STEP == 6)
    {
        durum = "Roket Yere Iniyor";
        a = 4;
    }

    delay(100);
}

void printEvent(sensors_event_t *event)
{

    if (event->type == SENSOR_TYPE_ACCELEROMETER)
    {
        x = event->acceleration.x;
        y = event->acceleration.y;
        z = event->acceleration.z;
        accY = -y;
        float Elevation = atan2(-y, sqrt(z * z + x * x));
        float b = Elevation * 180 / PI;
        el = b;
    }
    if (event->type == SENSOR_TYPE_ORIENTATION)
    {
        xd = event->orientation.x;
        yd = event->orientation.y;
        zd = event->orientation.z;
        orx = xd;
        ory = yd;
        orz = zd;
    }
}

float kalman_filter_bme(float input)
{
    float kalman_new8 = kalman_old8;
    float cov_new8 = cov_old8 + 0.4;
    float kalman_gain8 = cov_new8 / (cov_new8 + 1.2);
    float kalman_calculated8 = kalman_new8 + (kalman_gain8 * (input - kalman_new8));
    cov_new8 = (1 - kalman_gain8) * cov_old8;
    cov_old8 = cov_new8;
    kalman_old8 = kalman_calculated8;
    return kalman_calculated8;
}
void sendLora()
{
    *(float *)(data.a) = h;
    *(float *)(data.b) = el;
    *(float *)(data.c) = gps.location.lat();
    *(float *)(data.d) = gps.location.lng();
    *(float *)(data.e) = gps.altitude.meters();
    *(float *)(data.f) = orx;
    *(float *)(data.g) = ory;
    *(float *)(data.h) = orz;
    *(float *)(data.j) = a;

    e32ttl.sendFixedMessage(0, 44, 30, &data, sizeof(Signal));
}
