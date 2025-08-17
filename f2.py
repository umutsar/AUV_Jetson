#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSV tabanlı siyah şerit takip – CLAHE + İkili Aralık (v4)
- V kanalına CLAHE uygular (kontrastı sabitler).
- Koyu siyaha ek olarak "açık siyah/gri" için ikinci bir aralık ekler.
- Bölgeler trackbar + 1/2/3/r kısayollarıyla ayarlanır.
Kullanım:
    python line_follower_hsv_clahe.py --video path/to/video.mp4 --show-mask --resize 640
Kısayollar: q/ESC=çıkış, p=duraklat, s=ekranı kaydet, 1/2/3=zıplat, r=reset
"""
import cv2, numpy as np, argparse
from collections import namedtuple
import serial
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=str, required=True)
ap.add_argument("--show-mask", action="store_true")
ap.add_argument(
    "--resize",
    type=int,
    default=960,
    help="Görüntü genişliğini bu değere yeniden boyutlandır",
)
ap.add_argument("--area-threshold", type=float, default=0.02)
args = ap.parse_args()


# Seri iletişim ve port ayarları

alacakart = None
if 1:
    try:
        alacakart = serial.Serial("/dev/ttyACM0", 57600)
        print("Connected!")
        time.sleep(3)
        alacakart.write(b"AB\n")
        # alacakart.write(b'DK\n')
        # alacakart.write(b'D 50\n')
    except serial.SerialException as error:
        print(f"Baud_rate: xxx, Serial_port: xxx")
        print(f"Error initializing serial communication: {error}")
else:
    print("Serial communication disabled (serial_enabled=False)")


cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"Video açılamadı: {args.video}")

# --- Pencereler
cv2.namedWindow("Takip", cv2.WINDOW_NORMAL)
cv2.namedWindow("Ayarlar", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ayarlar", 520, 320)
cv2.namedWindow("Bolgeler", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bolgeler", 520, 260)


def nothing(_):
    pass


# HSV eşikleri (koyu siyah için)
cv2.createTrackbar("H_min", "Ayarlar", 0, 180, nothing)
cv2.createTrackbar("S_min", "Ayarlar", 0, 255, nothing)
cv2.createTrackbar("V_min", "Ayarlar", 0, 255, nothing)
cv2.createTrackbar("H_max", "Ayarlar", 0, 180, nothing)
cv2.createTrackbar("S_max", "Ayarlar", 0, 255, nothing)  # biraz geniş
cv2.createTrackbar("V_max", "Ayarlar", 0, 255, nothing)  # açık siyaha yaklaş
cv2.createTrackbar("Kernel", "Ayarlar", 0, 15, nothing)
cv2.createTrackbar("Open", "Ayarlar", 0, 5, nothing)
cv2.createTrackbar("Close", "Ayarlar", 0, 5, nothing)
cv2.createTrackbar("Blur", "Ayarlar", 10, 15, nothing)

# CLAHE ayarları
cv2.createTrackbar("CLAHE_on", "Ayarlar", 0, 1, nothing)  # 0/1
cv2.createTrackbar("CLAHE_clipx10", "Ayarlar", 4, 50, nothing)  # 2.0 varsayılan (x10)
cv2.createTrackbar("CLAHE_tile", "Ayarlar", 32, 32, nothing)  # 8x8 varsayılan

# İkili aralık (Light-Black = açık siyah)
cv2.createTrackbar("LB_on", "Ayarlar", 1, 1, nothing)  # 0/1
cv2.createTrackbar("LB_Smax", "Ayarlar", 214, 255, nothing)  # açık gri için Sat üst
cv2.createTrackbar(
    "LB_Vboost", "Ayarlar", 38, 120, nothing
)  # V_max üzerine eklenecek tolerans


# Bölge konumları: yüzde (0..100)
def create_zone_trackbars():
    # SOLA DÖN
    cv2.createTrackbar("L_x1%", "Bolgeler", 5, 100, nothing)
    cv2.createTrackbar("L_y1%", "Bolgeler", 30, 100, nothing)
    cv2.createTrackbar("L_x2%", "Bolgeler", 45, 100, nothing)
    cv2.createTrackbar("L_y2%", "Bolgeler", 70, 100, nothing)
    # DÜZ GİT
    cv2.createTrackbar("C_x1%", "Bolgeler", 40, 100, nothing)
    cv2.createTrackbar("C_y1%", "Bolgeler", 10, 100, nothing)
    cv2.createTrackbar("C_x2%", "Bolgeler", 60, 100, nothing)
    cv2.createTrackbar("C_y2%", "Bolgeler", 40, 100, nothing)
    # SAĞA DÖN
    cv2.createTrackbar("R_x1%", "Bolgeler", 55, 100, nothing)
    cv2.createTrackbar("R_y1%", "Bolgeler", 30, 100, nothing)
    cv2.createTrackbar("R_x2%", "Bolgeler", 95, 100, nothing)
    cv2.createTrackbar("R_y2%", "Bolgeler", 70, 100, nothing)

    # Yeni
    cv2.createTrackbar("Sag_x1%", "Bolgeler", 40, 100, nothing)
    cv2.createTrackbar("Sag_y1%", "Bolgeler", 10, 100, nothing)
    cv2.createTrackbar("Sag_x2%", "Bolgeler", 60, 100, nothing)
    cv2.createTrackbar("Sag_y2%", "Bolgeler", 40, 100, nothing)

    cv2.createTrackbar("Sol_x1%", "Bolgeler", 55, 100, nothing)
    cv2.createTrackbar("Sol_y1%", "Bolgeler", 30, 100, nothing)
    cv2.createTrackbar("Sol_x2%", "Bolgeler", 95, 100, nothing)
    cv2.createTrackbar("Sol_y2%", "Bolgeler", 70, 100, nothing)


create_zone_trackbars()


def set_zone(label_vals):
    for k, v in label_vals.items():
        cv2.setTrackbarPos(k, "Bolgeler", int(max(0, min(100, v))))


def reset_defaults():
    set_zone(
        {
            "L_x1%": 8,
            "L_y1%": 36,
            "L_x2%": 20,
            "L_y2%": 63,
            "C_x1%": 43,
            "C_y1%": 6,
            "C_x2%": 58,
            "C_y2%": 29,
            "R_x1%": 80,
            "R_y1%": 36,
            "R_x2%": 92,
            "R_y2%": 64,
            "Sag_x1%": 66,
            "Sag_y1%": 36,
            "Sag_x2%": 79,
            "Sag_y2%": 64,
            "Sol_x1%": 21,
            "Sol_y1%": 36,
            "Sol_x2%": 34,
            "Sol_y2%": 63,
        }
    )


reset_defaults()

Zone = namedtuple("Zone", "name key x1 y1 x2 y2")
CLR_IDLE = (90, 90, 90)
CLR_FOCUS = (40, 180, 40)
CLR_TEXT = (230, 230, 230)


def read_percent(name, label):
    return cv2.getTrackbarPos(label, name) / 100.0


def read_zones(w, h):
    def to_xy(xp, yp):
        return int(xp * w), int(yp * h)

    Lx1, Ly1, Lx2, Ly2 = [
        read_percent("Bolgeler", k) for k in ("L_x1%", "L_y1%", "L_x2%", "L_y2%")
    ]
    Cx1, Cy1, Cx2, Cy2 = [
        read_percent("Bolgeler", k) for k in ("C_x1%", "C_y1%", "C_x2%", "C_y2%")
    ]
    Rx1, Ry1, Rx2, Ry2 = [
        read_percent("Bolgeler", k) 
        for k in ("R_x1%", "R_y1%", "R_x2%", "R_y2%")
    ]

    Sagx1, Sagy1, Sagx2, Sagy2 = [
        read_percent("Bolgeler", k)
        for k in ("Sag_x1%", "Sag_y1%", "Sag_x2%", "Sag_y2%")
    ]
    Solx1, Soly1, Solx2, Soly2 = [
        read_percent("Bolgeler", k)
        for k in ("Sol_x1%", "Sol_y1%", "Sol_x2%", "Sol_y2%")
    ]

    def norm(x1, y1, x2, y2):
        x1, x2 = sorted([np.clip(x1, 0, 1), np.clip(x2, 0, 1)])
        y1, y2 = sorted([np.clip(y1, 0, 1), np.clip(y2, 0, 1)])
        (x1, y1) = to_xy(x1, y1)
        (x2, y2) = to_xy(x2, y2)
        if x2 - x1 < 10:
            x2 = min(x1 + 10, w - 1)
        if y2 - y1 < 10:
            y2 = min(y1 + 10, h - 1)
        return x1, y1, x2, y2

    L = Zone("SOLA DÖN", "LEFT", *norm(Lx1, Ly1, Lx2, Ly2))
    C = Zone("DÜZ GİT", "FWD", *norm(Cx1, Cy1, Cx2, Cy2))
    R = Zone("SAĞA DÖN", "RIGHT", *norm(Rx1, Ry1, Rx2, Ry2))

    Sol = Zone("SOLA YANAŞ", "H-", *norm(Solx1, Soly1, Solx2, Soly2))
    Sag = Zone("SAĞA YANAŞ", "H+", *norm(Sagx1, Sagy1, Sagx2, Sagy2))

    
    
    return [L, C, R, Sol, Sag]


def make_mask(bgr):
    # --- HSV + opsiyonel CLAHE (V kanalına)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if cv2.getTrackbarPos("CLAHE_on", "Ayarlar") == 1:
        clip = max(1, cv2.getTrackbarPos("CLAHE_clipx10", "Ayarlar")) / 10.0  # 0.1 adım
        tile = max(2, cv2.getTrackbarPos("CLAHE_tile", "Ayarlar"))
        v = hsv[:, :, 2]
        clahe = cv2.createCLAHE(
            clipLimit=float(clip), tileGridSize=(int(tile), int(tile))
        )
        hsv[:, :, 2] = clahe.apply(v)

    # --- Koyu siyah aralığı (asıl eşik)
    hmin = cv2.getTrackbarPos("H_min", "Ayarlar")
    smin = cv2.getTrackbarPos("S_min", "Ayarlar")
    vmin = cv2.getTrackbarPos("V_min", "Ayarlar")
    hmax = cv2.getTrackbarPos("H_max", "Ayarlar")
    smax = cv2.getTrackbarPos("S_max", "Ayarlar")
    vmax = cv2.getTrackbarPos("V_max", "Ayarlar")
    lower1 = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper1 = np.array([hmax, smax, vmax], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # --- Açık siyah (light-black) aralığı (opsiyonel ikinci maske)
    mask = mask1
    if cv2.getTrackbarPos("LB_on", "Ayarlar") == 1:
        LB_Smax = cv2.getTrackbarPos("LB_Smax", "Ayarlar")
        LB_Vboost = cv2.getTrackbarPos("LB_Vboost", "Ayarlar")
        LB_Vmax = int(min(255, vmax + LB_Vboost))
        lower2 = np.array([0, 0, 0], dtype=np.uint8)
        upper2 = np.array([180, LB_Smax, LB_Vmax], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    # --- Filtreleme (bulanık + morfoloji)
    ksz = max(3, cv2.getTrackbarPos("Kernel", "Ayarlar") | 1)  # tek sayı
    opn = cv2.getTrackbarPos("Open", "Ayarlar")
    cls = cv2.getTrackbarPos("Close", "Ayarlar")
    blr = max(1, cv2.getTrackbarPos("Blur", "Ayarlar") | 1)
    if blr > 1:
        mask = cv2.medianBlur(mask, blr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
    if opn > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=opn)
    if cls > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cls)
    return mask


paused = False
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        if args.resize and args.resize > 0:
            h, w = frame.shape[:2]
            scale = args.resize / float(w)
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

        draw = frame.copy()
        h, w = frame.shape[:2]
        zones = read_zones(w, h)
        mask = make_mask(frame)

        ratios = []
        for z in zones:
            roi = mask[z.y1 : z.y2, z.x1 : z.x2]
            nz = cv2.countNonZero(roi)
            area = roi.size + 1e-6
            ratios.append(nz / area)

        # Yeni karar verme mantığı
        decision = "SERIT YOK"
        
        # Merkez (C) ve yan bölgelerin oranlarını al
        c_ratio = ratios[1]  # C bölgesi (index 1)
        sag_ratio = ratios[3]  # SAĞA YANAŞ bölgesi (index 3)
        sol_ratio = ratios[4]  # SOLA YANAŞ bölgesi (index 4)
        
        # Sağa yanaş koşulu: C >= %30 VE SAĞA YANAŞ >= %20
        if c_ratio >= 0.30 and sag_ratio >= 0.20:
            decision = "SAĞA YANAŞ"
        # Sola yanaş koşulu: C >= %30 VE SOLA YANAŞ >= %20
        elif c_ratio >= 0.30 and sol_ratio >= 0.20:
            decision = "SOLA YANAŞ"
        # Eğer yan yanaş koşulları sağlanmıyorsa, en yüksek orana sahip bölgeyi seç
        else:
            best = int(np.argmax(ratios))
            best_ratio = float(ratios[best])
            if best_ratio >= args.area_threshold:
                decision = zones[best].name

        # Görsel gösterim için best bölgesini belirle
        if decision == "SAĞA YANAŞ":
            best = 3  # SAĞA YANAŞ bölgesi
        elif decision == "SOLA YANAŞ":
            best = 4  # SOLA YANAŞ bölgesi
        elif decision != "SERIT YOK":
            # Diğer kararlar için en yüksek orana sahip bölgeyi bul
            best = int(np.argmax(ratios))
        else:
            best = -1  # Hiçbir bölge seçilmedi

        for i, z in enumerate(zones):
            color = (
                CLR_FOCUS
                if i == best and best >= 0
                else CLR_IDLE
            )
            cv2.rectangle(draw, (z.x1, z.y1), (z.x2, z.y2), (0, 0, 0), 6)
            cv2.rectangle(draw, (z.x1, z.y1), (z.x2, z.y2), color, 2)
            txt = f"{z.name}  {ratios[i]*100:.1f}%"
            cv2.putText(
                draw,
                txt,
                (z.x1, max(0, z.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            draw,
            f"KARAR: {decision}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            CLR_TEXT,
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("Takip", draw)
        if args.show_mask:
            cv2.imshow("Mask", mask)

        if alacakart is not None and decision == "DÜZ GİT":
            try:
                alacakart.write(b"HF 40 2000\n")
                print("Komut gönderildi: HF 60 1000")
            except serial.SerialException as e:
                print(f"Seri port hatası: {e}")

        elif alacakart is not None and decision == "SAĞA DÖN":
            try:
                alacakart.write(b"T 90\n")
                print("Komut gönderildi: sağa dön")
            except serial.SerialException as e:
                print(f"Seri port hatası: {e}")

        elif alacakart is not None and decision == "SOLA DÖN":
            try:
                alacakart.write(b"T -90\n")
                print("Komut gönderildi: sola dön")
            except serial.SerialException as e:
                print(f"Seri port hatası: {e}")
                

        elif alacakart is not None and decision == "SOLA YANAŞ":
            try:
                alacakart.write(b"H- 40 500\n")
                print("Komut gönderildi: sola yanaş")
            except serial.SerialException as e:
                print(f"Seri port hatası: {e}")
                
                
        elif alacakart is not None and decision == "SAĞA YANAŞ":
            try:
                alacakart.write(b"H+ 40 500\n")
                print("Komut gönderildi: sağa yanaş")
            except serial.SerialException as e:
                print(f"Seri port hatası: {e}")

    key = cv2.waitKey(10) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord("p"):
        paused = not paused
    elif key == ord("s"):
        cv2.imwrite("frame_save.png", draw)
        cv2.imwrite("mask_save.png", mask)
        print("Kaydedildi: frame_save.png, mask_save.png")
    elif key == ord("1"):
        set_zone({"L_x1%": 5, "L_y1%": 30, "L_x2%": 45, "L_y2%": 70})
    elif key == ord("2"):
        set_zone({"C_x1%": 40, "C_y1%": 10, "C_x2%": 60, "C_y2%": 40})
    elif key == ord("3"):
        set_zone({"R_x1%": 55, "R_y1%": 30, "R_x2%": 95, "R_y2%": 70})
    elif key == ord("r"):
        reset_defaults()

cap.release()
cv2.destroyAllWindows()
