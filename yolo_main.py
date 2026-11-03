import os
import cv2
import numpy as np
import pygame
from djitellopy import Tello
from ultralytics import YOLO
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_SCRIPT_DIR, "yolov8n.pt")


# --- PARAMETERS ---
SPEED = 50
CONF_THRES = 0.4
FLIP_COOLDOWN = 2.0
BATTERY_UPDATE_INTERVAL = 2.0

# --- TRACKING PARAMETERS ---
PERSON_CLASS_ID = 0  # YOLO COCO: person = class 0
TRACKING_SPEED = 40
TRACKING_DEAD_ZONE = 50

# --- DISPLAY ---
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


def draw_box(img, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def load_yolo_model():
    """Folosește yolov8n.pt din folderul proiectului (necesar pe Wi‑Fi Tello fără internet)."""
    if os.path.isfile(_DEFAULT_WEIGHTS):
        print(f"Loading YOLO from {_DEFAULT_WEIGHTS} ...")
        return YOLO(_DEFAULT_WEIGHTS)
    print("Loading YOLO (descărcare yolov8n.pt — necesită internet) ...")
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        raise SystemExit(
            "\nNu s-a putut încărca yolov8n.pt.\n"
            "Pe Wi‑Fi-ul dronei Tello nu ai de obicei internet, deci trebuie să ai deja fișierul.\n\n"
            "Soluție:\n"
            "  1) Conectează-te la un Wi‑Fi cu internet și rulează o dată: python3 yolo_main.py\n"
            "     (sau descarcă manual de la GitHub Ultralytics assets)\n"
            "  2) Copiază fișierul yolov8n.pt în folderul proiectului, lângă yolo_main.py:\n"
            f"     {_DEFAULT_WEIGHTS}\n\n"
            f"Eroare: {e}"
        ) from e


print("Loading YOLO model...")
model = load_yolo_model()
print("YOLO loaded.")

pygame.init()
pygame.display.set_mode((200, 200))
tello = Tello()

try:
    tello.connect()
    battery = tello.get_battery()
    print(f"Baterie: {battery}%")
    if battery < 10:
        print("⚠️  ATENȚIE: Bateria este foarte scăzută!")
    tello.streamon()
    frame_read = tello.get_frame_read()
except Exception as e:
    print(f"❌ Eroare la conectarea dronă: {e}")
    pygame.quit()
    raise SystemExit(1)

running = True
flying = False
tracking_active = False
tracking_yaw_adjustment = 0

last_battery_update = 0.0
battery_cache = tello.get_battery()
last_flip_time = 0.0

while running:
    keys = pygame.key.get_pressed()
    lr = 0
    fb = 0
    ud = 0
    yaw_vel = 0

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            tracking_active = not tracking_active
            print("🎯 Tracking activat" if tracking_active else "⏸️  Tracking dezactivat")
        if event.type == pygame.QUIT:
            running = False

    if keys[pygame.K_LEFT]:
        lr = -SPEED
    elif keys[pygame.K_RIGHT]:
        lr = SPEED
    if keys[pygame.K_UP]:
        fb = SPEED
    elif keys[pygame.K_DOWN]:
        fb = -SPEED
    if keys[pygame.K_w]:
        ud = SPEED
    elif keys[pygame.K_s]:
        ud = -SPEED
    if keys[pygame.K_a]:
        yaw_vel = -SPEED
    elif keys[pygame.K_d]:
        yaw_vel = SPEED

    if keys[pygame.K_SPACE] and not flying:
        try:
            print("🚁 Decolare...")
            tello.takeoff()
            flying = True
            print("✅ Dronă în aer!")
        except Exception as e:
            print(f"⚠️  Eroare la decolare: {e}")
            flying = False
    if keys[pygame.K_l] and flying:
        try:
            print("⬇️  Aterizare...")
            tello.land()
            flying = False
            print("✅ Dronă la sol!")
        except Exception as e:
            print(f"⚠️  Eroare la aterizare: {e}")
    if keys[pygame.K_q]:
        running = False

    if flying:
        current_time = time.time()
        if current_time - last_flip_time >= FLIP_COOLDOWN:
            if keys[pygame.K_f]:
                tello.flip_forward()
                last_flip_time = current_time
            elif keys[pygame.K_b]:
                tello.flip_back()
                last_flip_time = current_time
            elif keys[pygame.K_e]:
                tello.flip_left()
                last_flip_time = current_time
            elif keys[pygame.K_r]:
                tello.flip_right()
                last_flip_time = current_time

    img = frame_read.frame
    if img is not None:
        img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        tracking_yaw_adjustment = 0
        detected_person = None
        best_person_conf = 0.0

        # YOLO inference (single frame)
        results = model.predict(img, conf=CONF_THRES, verbose=False)
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                names = results[0].names
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                    color = (0, 255, 0)
                    if cls_id == PERSON_CLASS_ID and tracking_active:
                        color = (0, 255, 255)
                        if conf > best_person_conf:
                            best_person_conf = conf
                            detected_person = (x1, y1, x2, y2)

                    label = f"{names.get(cls_id, str(cls_id)).upper()} {conf * 100:.1f}%"
                    draw_box(img, (x1, y1, x2, y2), label, color)

        if detected_person is not None and tracking_active and flying:
            x1, y1, x2, y2 = detected_person
            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2
            frame_center_x = DISPLAY_WIDTH // 2
            frame_center_y = DISPLAY_HEIGHT // 2
            offset_x = person_center_x - frame_center_x

            if abs(offset_x) > TRACKING_DEAD_ZONE:
                tracking_yaw_adjustment = int((offset_x / frame_center_x) * TRACKING_SPEED)
                tracking_yaw_adjustment = max(
                    -TRACKING_SPEED, min(TRACKING_SPEED, tracking_yaw_adjustment)
                )

            cv2.line(
                img,
                (frame_center_x, frame_center_y),
                (person_center_x, person_center_y),
                (255, 0, 255),
                2,
            )
            cv2.circle(img, (person_center_x, person_center_y), 8, (255, 0, 255), -1)
            cv2.putText(
                img,
                "TRACKING",
                (person_center_x - 40, max(20, person_center_y - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

        if tracking_active and flying and not (keys[pygame.K_a] or keys[pygame.K_d]):
            yaw_vel = tracking_yaw_adjustment

        now = time.time()
        if now - last_battery_update >= BATTERY_UPDATE_INTERVAL:
            try:
                battery_cache = tello.get_battery()
                last_battery_update = now
            except Exception:
                pass

        try:
            height = tello.get_height() if flying else 0
            yaw = tello.get_yaw() if flying else 0
        except Exception:
            height = 0
            yaw = 0

        cv2.putText(img, f"BAT: {battery_cache}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"ALT: {height} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"YAW: {yaw:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(
            img,
            f"TRACKING: {'ON' if tracking_active else 'OFF'}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Tello + YOLOv8n", img)

    if flying:
        tello.send_rc_control(lr, fb, ud, yaw_vel)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False

print("\n🛑 Oprire program...")
try:
    tello.send_rc_control(0, 0, 0, 0)
    if flying:
        tello.land()
    tello.streamoff()
    tello.end()
except Exception as e:
    print(f"⚠️  Eroare la cleanup: {e}")
finally:
    cv2.destroyAllWindows()
    pygame.quit()
    print("✅ Program terminat corect.")
