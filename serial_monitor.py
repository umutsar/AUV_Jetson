import serial
import time

# Seri portu ve baud rate ayarla
ser = serial.Serial('/dev/ttyACM0', 57600, timeout=1)  # Portu kendine göre değiştir
ser.flush()

try:
    while True:
        # Seri porttan veri okuma
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').rstrip()
            print(f"Gelen: {line}")

        # Kullanıcıdan text gönderme
        user_input = input("Gönderilecek metin (ENTER boş bırakır): ")
        if user_input:
            ser.write((user_input + '\n').encode('utf-8'))
            print(f"Gönderildi: {user_input}")

except KeyboardInterrupt:
    print("Program durduruldu.")
finally:
    ser.close()
