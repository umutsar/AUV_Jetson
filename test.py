import cv2
import time

# Kamera aygıtı (Linux için /dev/video0 genellikle ilk kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    exit()

# Video ayarları
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Kare/saniye

# VideoWriter nesnesi oluştur
out = cv2.VideoWriter('output_30s.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

print("30 saniyelik kayit basliyor...")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alinamadi.")
        break

    out.write(frame)
    cv2.imshow('Kayit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("kullanici tarafindan durduruldu.")
        break

    # 30 saniye geçtiyse çık
    if time.time() - start_time > 30:
        print("30 saniye tamamlandi.")
        break

# Temizlik
cap.release()
out.release()
cv2.destroyAllWindows()

