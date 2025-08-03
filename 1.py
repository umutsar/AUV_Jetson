import cv2
import numpy as np
import configparser
import serial
import threading
import time

# Config dosyasını oku
config = configparser.ConfigParser()
config.read('/home/auv/Desktop/tg-hku-auv-ms/config.txt', encoding='utf-8')

def get_config_value(section, key, default, cast_func):
    # Config dosyasından tip dönüşümü ile değer oku
    try:
        return cast_func(config[section][key])
    except Exception:
        return default

def get_config_str(section, key, default):
    # Config dosyasından string değer oku
    try:
        return config[section][key]
    except Exception:
        return default

def str2bool(v):
    # Stringi booleana çevir
    return str(v).lower() in ("yes", "true", "1")

# Görüntü penceresi boyutları
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 720

# Video dosya adı
video_file = get_config_str('PARAMS', 'video_file', 'line-yatay.mp4')

# Seri iletişim ve port ayarları
serial_enabled = get_config_value('COMM', 'serial_enabled', 1, str2bool)
baud_rate = get_config_value('COMM', 'baud_rate', 57600, int)
serial_port = get_config_str('COMM', 'serial_port', '/dev/ttyACM0')

print(f"Serial enabled: {serial_enabled}")

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

# Video dosyasını aç
cap = cv2.VideoCapture(video_file)

# Görüntü işleme parametreleri
blur_val = get_config_value('PARAMS', 'blur_val', 1, int)
contrast = get_config_value('PARAMS', 'contrast', 2.0, float)
saturation = get_config_value('PARAMS', 'saturation', 0.0, float)
brightness = get_config_value('PARAMS', 'brightness', 50, int)

# HSV renk aralığı
lower = np.array([
    get_config_value('HSV', 'lower_h', 0, int),
    get_config_value('HSV', 'lower_s', 0, int),
    get_config_value('HSV', 'lower_v', 0, int)
])
upper = np.array([
    get_config_value('HSV', 'upper_h', 30, int),
    get_config_value('HSV', 'upper_s', 30, int),
    get_config_value('HSV', 'upper_v', 220, int)
])

# Hareket komutları
commands = {
    'forward': get_config_str('COMMANDS', 'forward', ''),
    'turn_right': get_config_str('COMMANDS', 'turn_right', ''),
    'turn_left': get_config_str('COMMANDS', 'turn_left', ''),
    'right': get_config_str('COMMANDS', 'right', ''),
    'left': get_config_str('COMMANDS', 'left', ''),
    'stop': get_config_str('COMMANDS', 'stop', '')
}

direction = ""
command_text = ""
command_lock = threading.Lock()
last_sent_command = ""
last_command_candidate = ""
last_command_candidate_time = 0

# Komutları seri porta ve terminale gönderen thread
def serial_sender():
    global last_sent_command, last_command_candidate, last_command_candidate_time
    turn_waiting_hf = False
    while True:
        time.sleep(0.05)
        with command_lock:
            cmd = command_text
        if not cmd:
            continue
        cmd_stripped = cmd.strip()
        # Komut değiştiyse 1 saniye boyunca sabit gelmesini bekle
        if last_command_candidate != cmd:
            last_command_candidate = cmd
            last_command_candidate_time = time.time()
            continue
        if (time.time() - last_command_candidate_time) < 1.0:
            continue
        if last_sent_command == cmd:
            continue
        print(f"Command sent to vehicle: {cmd_stripped}")
        # Dönüş komutları için HF bekleme mantığı
        if turn_waiting_hf:
            if cmd_stripped == commands['forward']:
                if last_sent_command != cmd:
                    print(f"[SERIAL] STOP command sent: {commands['stop']}")
                    if serial_enabled and alacakart is not None and hasattr(alacakart, 'is_open') and alacakart.is_open:
                        try:
                            alacakart.write((commands['stop'] + '\n').encode())
                            time.sleep(0.5)
                        except Exception as e:
                            print(f"Serial send error: {e}")
                print(f"[SERIAL] Command sent: {cmd}")
                last_sent_command = cmd
                turn_waiting_hf = False
                if serial_enabled and alacakart is not None and hasattr(alacakart, 'is_open') and alacakart.is_open:
                    try:
                        alacakart.write((cmd + '\n').encode())
                    except Exception as e:
                        print(f"Serial send error: {e}")
            else:
                continue
        else:
            if cmd_stripped in [commands['turn_right'], commands['turn_left']]:
                turn_waiting_hf = True
            if last_sent_command != cmd:
                print(f"[SERIAL] STOP command sent: {commands['stop']}")
                if serial_enabled and alacakart is not None and hasattr(alacakart, 'is_open') and alacakart.is_open:
                    try:
                        alacakart.write((commands['stop'] + '\n').encode())
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"Serial send error: {e}")
            print(f"[SERIAL] Command sent: {cmd}")
            last_sent_command = cmd
            if serial_enabled and alacakart is not None and hasattr(alacakart, 'is_open') and alacakart.is_open:
                try:
                    alacakart.write((cmd + '\n').encode())
                except Exception as e:
                    print(f"Serial send error: {e}")

# Komut gönderim threadini başlat
serial_thread = threading.Thread(target=serial_sender, daemon=True)
serial_thread.start()

# Çıktı video dosya adı
output_video_file = get_config_str('PARAMS', 'output_video', 'output_segmented.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Kırmızı, mavi ve yeşil kutuların koordinatları
    red_box = (int(frame_w*0.35), int(frame_h*0.1), int(frame_w*0.35), int(frame_h*0.2))
    blue_box = (int(frame_w*0.1), int(frame_h*0.4), int(frame_w*0.2), int(frame_h*0.2))
    green_box = (int(frame_w*0.7), int(frame_h*0.4), int(frame_w*0.2), int(frame_h*0.2))

    # Tolerans kutusu (kırmızı kutunun ortasında küçük bir kutu)
    tol_w, tol_h = int(red_box[2]*0.3), int(red_box[3]*0.5)
    tol_x = red_box[0] + (red_box[2] - tol_w)//2
    tol_y = red_box[1] + (red_box[3] - tol_h)//2
    tolerance_box = (tol_x, tol_y, tol_w, tol_h)

    # Görüntü işleme
    frame_adj = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    hsv_img = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[...,1] = np.clip(hsv_img[...,1] * saturation, 0, 255)
    hsv_img = hsv_img.astype(np.uint8)
    frame_adj = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    if blur_val > 1:
        frame_adj = cv2.GaussianBlur(frame_adj, (blur_val, blur_val), 0)
    hsv_img = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower, upper)

    # Kontur bulma ve birleştirme
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    combined_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 5000:
            cv2.drawContours(combined_mask, [cnt], -1, 255, -1)
    kernel = np.ones((15, 15), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    cv2.drawContours(output, combined_contours, -1, (0, 255, 0), 2)

    # Kutuları çiz
    cv2.rectangle(output, (red_box[0], red_box[1]), (red_box[0]+red_box[2], red_box[1]+red_box[3]), (0,0,255), 2)
    cv2.rectangle(output, (blue_box[0], blue_box[1]), (blue_box[0]+blue_box[2], blue_box[1]+blue_box[3]), (255,0,0), 2)
    cv2.rectangle(output, (green_box[0], green_box[1]), (green_box[0]+green_box[2], green_box[1]+green_box[3]), (0,255,0), 2)
    cv2.rectangle(output, (tolerance_box[0], tolerance_box[1]), (tolerance_box[0]+tolerance_box[2], tolerance_box[1]+tolerance_box[3]), (0,0,255), 1)

    # Alan kontrolleri
    red_area = combined_mask[red_box[1]:red_box[1]+red_box[3], red_box[0]:red_box[0]+red_box[2]]
    blue_area = combined_mask[blue_box[1]:blue_box[1]+blue_box[3], blue_box[0]:blue_box[0]+blue_box[2]]
    green_area = combined_mask[green_box[1]:green_box[1]+green_box[3], green_box[0]:green_box[0]+green_box[2]]
    tol_area = combined_mask[tolerance_box[1]:tolerance_box[1]+tolerance_box[3], tolerance_box[0]:tolerance_box[0]+tolerance_box[2]]

    new_direction = ""
    new_command_text = ""

    # Kırmızı kutuda alan var mı?
    if cv2.countNonZero(red_area) > 0:
        red_mask = np.zeros_like(combined_mask)
        red_mask[red_box[1]:red_box[1]+red_box[3], red_box[0]:red_box[0]+red_box[2]] = red_area
        moments = cv2.moments(red_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            tol_left = tolerance_box[0]
            tol_right = tolerance_box[0] + tolerance_box[2]
            if tol_left <= cx <= tol_right and cv2.countNonZero(tol_area) > 0:
                new_direction = "forward"
                new_command_text = commands['forward']
            elif cx > tol_right:
                new_direction = "move right"
                new_command_text = commands['right']
            elif cx < tol_left:
                new_direction = "move left"
                new_command_text = commands['left']
        else:
            new_direction = "forward"
            new_command_text = commands['forward']
    else:
        # Kırmızı yoksa yeşil ve maviye bak
        if cv2.countNonZero(green_area) > 0:
            new_direction = "right"
            new_command_text = commands['turn_right']
        elif cv2.countNonZero(blue_area) > 0:
            new_direction = "left"
            new_command_text = commands['turn_left']

    # Komutları thread ile paylaş
    with command_lock:
        direction = new_direction
        command_text = new_command_text

    if direction:
        cv2.putText(output, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        if last_sent_command:
            cv2.putText(output, f"Sent: {last_sent_command}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    # Kolaj oluştur ve göster
    mask_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    image_height = SCREEN_HEIGHT
    image_width = SCREEN_WIDTH // 2
    output_resized = cv2.resize(output, (image_width, image_height))
    mask_resized = cv2.resize(mask_color, (image_width, image_height))
    collage = np.hstack([output_resized, mask_resized])
    if last_sent_command:
        cv2.putText(collage, f"Sent: {last_sent_command}", (50, image_height-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    cv2.namedWindow('Line Follower Collage', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Line Follower Collage', SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.moveWindow('Line Follower Collage', 0, 0)
    cv2.imshow('Line Follower Collage', collage)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        alacakart.write(b'AK\n')
        break

cap.release()
cv2.destroyAllWindows()
