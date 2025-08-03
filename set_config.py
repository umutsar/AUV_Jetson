import cv2
import numpy as np
import configparser

# Config dosyası yolu
CONFIG_FILE = '/home/auv/Desktop/tg-hku-auv-ms/config.txt'

# Görüntü penceresi boyutları
screen_width = 960
screen_height = 720

# Config dosyasını oku
config = configparser.ConfigParser()
config.read(CONFIG_FILE, encoding='utf-8')

def get_config_str(section, key, default):
    # Config dosyasından string değer oku
    try:
        return config[section][key]
    except Exception:
        return default

def get_config_value(section, key, default, cast_func):
    # Config dosyasından tip dönüşümü ile değer oku
    try:
        return cast_func(config[section][key])
    except Exception:
        return default

# Video dosya adını al
video_file = get_config_str('PARAMS', 'video_file', 'output_30s.mp4')

# Parametreleri ve HSV değerlerini configten al
params = {
    'blur_val': get_config_value('PARAMS', 'blur_val', 1, int),
    'contrast': get_config_value('PARAMS', 'contrast', 2.0, float),
    'saturation': get_config_value('PARAMS', 'saturation', 0.0, float),
    'brightness': get_config_value('PARAMS', 'brightness', 50, int)
}
hsv = {
    'lower_h': get_config_value('HSV', 'lower_h', 0, int),
    'lower_s': get_config_value('HSV', 'lower_s', 0, int),
    'lower_v': get_config_value('HSV', 'lower_v', 0, int),
    'upper_h': get_config_value('HSV', 'upper_h', 30, int),
    'upper_s': get_config_value('HSV', 'upper_s', 30, int),
    'upper_v': get_config_value('HSV', 'upper_v', 220, int)
}

def nothing(x):
    # Trackbar callback fonksiyonu (gereksiz, placeholder)
    pass

def save_config(params, hsv):
    # Parametreleri ve HSV değerlerini config dosyasına kaydet
    config = configparser.ConfigParser()
    config['PARAMS'] = {k: str(params[k]) for k in params}
    config['PARAMS']['video_file'] = video_file
    config['HSV'] = {k: str(hsv[k]) for k in hsv}
    with open(CONFIG_FILE, 'w', encoding='utf-8') as configfile:
        config.write(configfile)

# Videodan örnek kareleri al
cap = cv2.VideoCapture(video_file)
ret, first_frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError(f"Could not read first frame from {video_file}")

cap = cv2.VideoCapture(video_file)
frame_indices = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for idx in range(0, frame_count, 100):
    frame_indices.append(idx)
frames = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
cap.release()
if not frames:
    raise RuntimeError(f"Could not read any frames from {video_file}")

# Ayar paneli penceresini oluştur
cv2.namedWindow('Settings Panel')

# Trackbar'ları oluştur
cv2.createTrackbar('Blur', 'Settings Panel', params['blur_val'], 31, nothing)
cv2.createTrackbar('Contrast x10', 'Settings Panel', int(params['contrast']*10), 50, nothing)
cv2.createTrackbar('Saturation x10', 'Settings Panel', int(params['saturation']*10), 50, nothing)
cv2.createTrackbar('Brightness', 'Settings Panel', params['brightness'], 255, nothing)

cv2.createTrackbar('Lower H', 'Settings Panel', hsv['lower_h'], 179, nothing)
cv2.createTrackbar('Lower S', 'Settings Panel', hsv['lower_s'], 255, nothing)
cv2.createTrackbar('Lower V', 'Settings Panel', hsv['lower_v'], 255, nothing)
cv2.createTrackbar('Upper H', 'Settings Panel', hsv['upper_h'], 179, nothing)
cv2.createTrackbar('Upper S', 'Settings Panel', hsv['upper_s'], 255, nothing)
cv2.createTrackbar('Upper V', 'Settings Panel', hsv['upper_v'], 255, nothing)

while True:
    # Ekran boyutunu ayarla
    image_height = screen_height
    image_width = screen_width // 2

    # Trackbar'lardan güncel değerleri al
    blur_val = cv2.getTrackbarPos('Blur', 'Settings Panel')
    if blur_val % 2 == 0:
        blur_val += 1
    contrast = cv2.getTrackbarPos('Contrast x10', 'Settings Panel') / 10.0
    saturation = cv2.getTrackbarPos('Saturation x10', 'Settings Panel') / 10.0
    brightness = cv2.getTrackbarPos('Brightness', 'Settings Panel')

    lower_h = cv2.getTrackbarPos('Lower H', 'Settings Panel')
    lower_s = cv2.getTrackbarPos('Lower S', 'Settings Panel')
    lower_v = cv2.getTrackbarPos('Lower V', 'Settings Panel')
    upper_h = cv2.getTrackbarPos('Upper H', 'Settings Panel')
    upper_s = cv2.getTrackbarPos('Upper S', 'Settings Panel')
    upper_v = cv2.getTrackbarPos('Upper V', 'Settings Panel')

    # Parametreleri güncelle
    params['blur_val'] = blur_val
    params['contrast'] = contrast
    params['saturation'] = saturation
    params['brightness'] = brightness
    hsv['lower_h'] = lower_h
    hsv['lower_s'] = lower_s
    hsv['lower_v'] = lower_v
    hsv['upper_h'] = upper_h
    hsv['upper_s'] = upper_s
    hsv['upper_v'] = upper_v

    # Her örnek kare için ayarları uygula ve kolaj oluştur
    collage_rows = []
    for frame in frames:
        frame_adj = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        hsv_img = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_img[...,1] = np.clip(hsv_img[...,1] * saturation, 0, 255)
        hsv_img = hsv_img.astype(np.uint8)
        frame_adj2 = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        if blur_val > 1:
            frame_adj2 = cv2.GaussianBlur(frame_adj2, (blur_val, blur_val), 0)
        hsv_img = cv2.cvtColor(frame_adj2, cv2.COLOR_BGR2HSV)
        lower_arr = np.array([lower_h, lower_s, lower_v])
        upper_arr = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv_img, lower_arr, upper_arr)
        preview_resized = cv2.resize(frame_adj2, (image_width, image_height // len(frames)))
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_resized = cv2.resize(mask_color, (image_width, image_height // len(frames)))
        row = np.hstack([preview_resized, mask_resized])
        collage_rows.append(row)
    collage = np.vstack(collage_rows)
    cv2.namedWindow('Settings Collage', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Settings Collage', screen_width, screen_height)
    cv2.moveWindow('Settings Collage', 0, 0)
    cv2.imshow('Settings Collage', collage)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('s'):
        save_config(params, hsv)
        print('Settings saved to config.txt.')
    elif key == ord('q'):
        break

cv2.destroyAllWindows() 




