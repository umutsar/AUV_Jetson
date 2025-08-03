import cv2
import numpy as np
import configparser
import serial
import threading
import time


baud_rate = 57600
serial_port = "/dev/ttyACM0"
serial_enabled = 1

# Seri iletişim ve port ayarları


alacakart = None
if serial_enabled:
    try:
        alacakart = serial.Serial(serial_port, baud_rate)
        print("Connected!")
        time.sleep(6)
        alacakart.write(b'AB\n')
        # alacakart.write(b'DK\n')
        # alacakart.write(b'D 50\n')
    except serial.SerialException as error:
        print(f"Baud_rate: {baud_rate}, Serial_port: {serial_port}")
        print(f"Error initializing serial communication: {error}")
else:
    print("Serial communication disabled (serial_enabled=False)")

